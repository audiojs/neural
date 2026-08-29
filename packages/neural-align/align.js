// Forced alignment adapter — a wav2vec2-class CTC acoustic model
// (@huggingface/transformers, Apache-2.0) supplies per-frame log-probabilities;
// ./ctc.js's trellis does the actual alignment. Default model
// Xenova/wav2vec2-base-960h (facebook/wav2vec2-base-960h weights, Apache-2.0,
// English). See README for multilingual alternatives and their licenses.

import { AutoProcessor, AutoTokenizer, AutoModelForCTC, env } from '@huggingface/transformers'
import resample from '@audio/resample-sinc'
import { ctcAlign, mergeWords } from './ctc.js'

const SAMPLE_RATE = 16000       // wav2vec2-class models are trained on 16 kHz mono
const FRAME_DURATION = 0.02     // 320-sample conv-stem stride @ 16 kHz — fixed by the architecture, not derived from audio length (a 'valid'-padding conv stem trims a partial final frame)
const DEFAULT_MODEL = 'Xenova/wav2vec2-base-960h'
const DELIMITER = '|'           // wav2vec2 CTC vocab's word-separator token

const models = new Map()        // "<model>|<dtype>|<device>|<cache>" → Promise<handle>, so repeated align() calls reuse the loaded net

// Node: persist the HF model cache outside node_modules, under the lane's shared
// cache root, so `rm -rf node_modules` (routine after any `npm install`) doesn't
// discard a 360 MB model download. @audio/neural-runtime already keys its own
// (unrelated) ONNX-bytes cache off $AUDIO_NEURAL_CACHE / ~/.cache/audiojs/neural —
// transformers.js' model files go in that root's 'hf/' subdir so the two caches
// never collide. Browser keeps transformers.js' own Cache API default untouched.
const isNode = typeof process !== 'undefined' && process.versions?.node
if (isNode) {
  let [{ join }, os] = await Promise.all([import('node:path'), import('node:os')])
  env.cacheDir = process.env.AUDIO_NEURAL_CACHE
    ? join(process.env.AUDIO_NEURAL_CACHE, 'hf')
    : join(os.homedir(), '.cache', 'audiojs', 'neural', 'hf')
}

function toMono (audio) {
  if (Array.isArray(audio)) {
    let n = audio[0]?.length ?? 0, out = new Float32Array(n)
    for (let c = 0; c < audio.length; c++) for (let i = 0; i < n; i++) out[i] += audio[c][i] / audio.length
    return out
  }
  if (audio instanceof Float64Array) return Float32Array.from(audio)
  if (audio instanceof Float32Array) return audio
  throw new TypeError('align: audio must be a Float32Array, Float64Array, or channel array of either')
}

// Uppercase, drop characters the vocab has no id for (punctuation; digits — spelling
// out numbers is NOT attempted, see README), collapse whitespace to the delimiter.
// normalize:false skips only the case-folding step; unsupported characters are
// always dropped (the vocab has no id to align them to).
function textToTargets (text, vocab, { normalize = true } = {}) {
  let words = (normalize ? text.toUpperCase() : text).trim().split(/\s+/).filter(Boolean)
  let ids = []
  words.forEach((word, i) => {
    for (let ch of word) { let id = vocab[ch]; if (id != null) ids.push(id) }
    if (i < words.length - 1) ids.push(vocab[DELIMITER])
  })
  return Int32Array.from(ids)
}

function logSoftmaxRows (data, T, V) {
  let out = new Float32Array(T * V)
  for (let t = 0; t < T; t++) {
    let base = t * V, max = -Infinity
    for (let v = 0; v < V; v++) if (data[base + v] > max) max = data[base + v]
    let sum = 0
    for (let v = 0; v < V; v++) sum += Math.exp(data[base + v] - max)
    let logSum = max + Math.log(sum)
    for (let v = 0; v < V; v++) out[base + v] = data[base + v] - logSum
  }
  return out
}

async function importSubtitle () {
  try { return await import('@audio/subtitle') }
  catch { throw new Error('align: opts.cues/opts.lrc need @audio/subtitle (npm install @audio/subtitle)') }
}

/** model → { align, logits, free } — loaded once per (model, dtype, device, cache), cached. */
export async function loadModel (model = DEFAULT_MODEL, opts = {}) {
  let key = `${model}|${opts.dtype ?? ''}|${opts.device ?? ''}|${opts.cache ?? ''}`
  if (models.has(key)) return models.get(key)

  let handle = (async () => {
    let loadOpts = {}
    if (opts.dtype) loadOpts.dtype = opts.dtype
    if (opts.device) loadOpts.device = opts.device
    if (opts.progress) loadOpts.progress_callback = opts.progress
    if (opts.cache) loadOpts.cache_dir = opts.cache   // per-call override of env.cacheDir (Node's FileCache; ignored on the browser Cache API path)

    let [processor, tokenizer, net] = await Promise.all([
      AutoProcessor.from_pretrained(model, loadOpts),
      AutoTokenizer.from_pretrained(model, loadOpts),
      AutoModelForCTC.from_pretrained(model, loadOpts),
    ])

    let blank = tokenizer.pad_token_id ?? 0
    let vocab = tokenizer._tokenizerJSON.model.vocab            // char → id
    let id2char = []
    for (let [ch, id] of Object.entries(vocab)) id2char[id] = ch
    for (let id of tokenizer.all_special_ids ?? []) id2char[id] = ''   // <pad>/<s>/</s>/<unk> never form a word

    async function logits (audio, opts2 = {}) {
      let sr = opts2.sampleRate
      if (!sr) throw new RangeError('align: opts.sampleRate required (Hz of the input audio)')
      let mono = toMono(audio)
      let mono16k = sr === SAMPLE_RATE ? mono : resample(mono, { from: sr, to: SAMPLE_RATE })
      let inputs = await processor(mono16k)
      let { logits: out } = await net(inputs)
      let [, T, V] = out.dims
      return { logProbs: logSoftmaxRows(out.data, T, V), T, V, frameDuration: FRAME_DURATION, vocab: id2char.slice() }
    }

    // Long audio: run inference per `chunk`-second window (1 s overlap for encoder
    // context), drop each window's overlapping lead-in frames, concatenate the rest
    // into one T×V log-prob matrix, then run ctcAlign once over the full target —
    // the DP itself is O(T·L) and needs no chunking; only the acoustic model does.
    async function longLogits (mono, sr, chunkSec) {
      let chunkLen = Math.round(chunkSec * sr)
      let overlapLen = Math.round(sr)                     // 1 s
      let step = Math.max(1, chunkLen - overlapLen)
      let overlapFrames = Math.round(1 / FRAME_DURATION)
      let rows = [], V = 0
      for (let start = 0; start < mono.length; start += step) {
        let end = Math.min(start + chunkLen, mono.length)
        let piece = mono.subarray(start, end)
        let r = await logits(piece, { sampleRate: sr })
        V = r.V
        let from = start === 0 ? 0 : Math.min(overlapFrames, r.T)
        for (let t = from; t < r.T; t++) rows.push(r.logProbs.subarray(t * V, (t + 1) * V))
        if (end === mono.length) break
      }
      let T = rows.length
      let logProbs = new Float32Array(T * V)
      rows.forEach((row, t) => logProbs.set(row, t * V))
      return { logProbs, T, V }
    }

    async function alignFn (audio, text, opts2 = {}) {
      let sr = opts2.sampleRate
      if (!sr) throw new RangeError('align: opts.sampleRate required (Hz of the input audio)')
      let mono = toMono(audio)
      let targets = textToTargets(text, vocab, { normalize: opts2.normalize ?? true })
      if (!targets.length) throw new RangeError('align: text has no alignable characters after normalization')

      let chunkSec = opts2.chunk ?? 20
      let totalSec = mono.length / sr
      let { logProbs, T, V } = totalSec <= chunkSec
        ? await logits(mono, { sampleRate: sr })
        : await longLogits(mono, sr, chunkSec)

      let ctc = ctcAlign(logProbs, T, V, targets, { blank })
      let words = mergeWords(ctc.spans, id2char, { delimiter: DELIMITER, frameDuration: FRAME_DURATION })
      let chars = ctc.spans
        .filter(s => id2char[s.token] && id2char[s.token] !== DELIMITER)
        .map(s => ({ text: id2char[s.token], start: s.start * FRAME_DURATION, end: s.end * FRAME_DURATION, score: s.score }))
      let score = words.length ? words.reduce((s, w) => s + w.score, 0) / words.length : 0

      let result = { words, chars, score }
      if (opts2.cues || opts2.lrc) {
        let { fromWords } = await importSubtitle()
        result.cues = fromWords(words)
      }
      if (opts2.lrc) {
        let { default: write } = await importSubtitle()
        result.lrc = write(result.cues, 'lrc', { words: true })
      }
      return result
    }

    return {
      align: alignFn,
      logits,
      free: async () => { models.delete(key); await net.dispose?.() },
    }
  })()

  models.set(key, handle)
  return handle
}

/** audio + known transcript → word/character timestamps. See loadModel for the cached model handle. */
export default async function align (audio, text, opts = {}) {
  let net = await loadModel(opts.model ?? DEFAULT_MODEL, opts)
  return net.align(audio, text, opts)
}

export { ctcAlign, mergeWords }
