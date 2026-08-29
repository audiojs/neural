// @audio/neural-asr — Whisper speech-to-text, running locally in Node and the browser
// via @huggingface/transformers (Apache-2.0), which runs the model through ONNX Runtime
// (onnxruntime-node in Node, onnxruntime-web/wasm/WebGPU in the browser). This is the
// in-browser Whisper path the ecosystem converged on (Xenova's whisper-web, the official
// transformers.js ASR examples) — no bespoke inference code here, only shaping.
//
// Model: Whisper (Radford, Kim, Xu, Brockman, McLeavey, Sutskever — "Robust Speech
// Recognition via Large-Scale Weak Supervision", https://cdn.openai.com/papers/whisper.pdf).
// OpenAI's original weights are MIT (https://github.com/openai/whisper/blob/main/LICENSE);
// the onnx-community/* conversions used here re-export the same MIT weights as ONNX
// (conversion tooling itself is Apache-2.0, transformers.js's license).
//
// Whisper's encoder always expects 16 kHz mono log-mel input — the pipeline does NOT
// resample a raw Float32Array for you (only URL/path/Blob inputs get read_audio()
// treatment), so callers' audio is mixed to mono and resampled here before it ever
// reaches transformers.js. See @audio/resample-sinc for the resampler.

import sinc from '@audio/resample-sinc'
import { pipeline, env } from '@huggingface/transformers'
// node:os / node:path are imported lazily inside defaultCacheDir() below, not here: a static
// top-level import is hoisted and resolved by the module graph regardless of the IS_NODE branch
// that guards its only caller, so a browser bundle would try to fetch 'node:os'/'node:path' as
// URLs and fail (CORS/ERR_FAILED) even though that code path never runs outside Node. A dynamic
// import() is lazy — it only executes when defaultCacheDir() is actually called, which happens
// only when IS_NODE is true.

const SR = 16000  // Whisper's fixed input rate — not a tunable, it's baked into the pretrained encoder
const IS_NODE = typeof process !== 'undefined' && !!process.versions?.node

// Curated model table. All are OpenAI Whisper checkpoints (MIT) converted to ONNX by the
// onnx-community org on the Hugging Face Hub. `size` is the q8 download total (encoder +
// decoder + tokenizer/config) — measured for tiny.en (42 MB, see README), ~1 byte/param
// for q8 quantization plus ~2.5 MB fixed tokenizer/config overhead for the rest (not
// individually measured — download and check `~/.cache/audiojs/neural/hf` to confirm).
export const models = [
	{ id: 'onnx-community/whisper-tiny.en', params: '39M', languages: 'en', size: 42_000_000 },
	{ id: 'onnx-community/whisper-tiny', params: '39M', languages: 'multi', size: 42_000_000 },
	{ id: 'onnx-community/whisper-base.en', params: '74M', languages: 'en', size: 77_000_000 },
	{ id: 'onnx-community/whisper-base', params: '74M', languages: 'multi', size: 77_000_000 },
	{ id: 'onnx-community/whisper-small', params: '244M', languages: 'multi', size: 247_000_000 },
	{ id: 'onnx-community/whisper-large-v3-turbo', params: '809M', languages: 'multi', size: 812_000_000 },
]

/** Mix any channel layout down to mono (equal-weight average). Returns the input unchanged if already mono. */
export function mono(channelData) {
	if (channelData.length === 1) return channelData[0]
	let n = channelData[0].length
	let out = new Float32Array(n)
	for (let c = 0; c < channelData.length; c++) {
		let ch = channelData[c]
		for (let i = 0; i < n; i++) out[i] += ch[i] / channelData.length
	}
	return out
}

/** Resample mono data to Whisper's fixed 16 kHz input rate (no-op if already 16 kHz). */
export function to16k(data, sampleRate) {
	if (sampleRate === SR) return Float32Array.from(data)
	return sinc(data, { from: sampleRate, to: SR })
}

// Round to 3 decimal places (millisecond precision) — matches the task's stated timestamp precision.
let round3 = t => Math.round(t * 1000) / 1000

/** Map transformers.js's Whisper `chunks` ([{text, timestamp:[start,end]}]) to {start,end,text} segments.
 * A null/undefined end timestamp (happens on the final chunk when generation is cut off mid-word) is
 * clamped to `duration` — the end of the audio being transcribed. */
export function toSegments(chunks, duration) {
	if (!chunks?.length) return []
	return chunks.map(({ text, timestamp: [start, end] }) => ({
		start: round3(start ?? 0),
		end: round3(end ?? duration),
		text: text.trim(),
	}))
}

// Sentence-ending punctuation groups word-level chunks into segment-like spans when the
// caller asked for word timestamps — Whisper's own segment boundaries come from timestamp
// tokens emitted only in the non-word decode path, so with `timestamps: 'word'` there is no
// native segmentation to reuse. This is a documented approximation (see README), not the
// model's own segmentation.
function segmentsFromWords(words) {
	let segs = [], cur = null
	for (let w of words) {
		cur = cur ? { start: cur.start, end: w.end, text: cur.text + ' ' + w.text } : { start: w.start, end: w.end, text: w.text }
		if (/[.?!]["')\]]?$/.test(w.text)) { segs.push(cur); cur = null }
	}
	if (cur) segs.push(cur)
	return segs
}

let wsRe = /\s+/g
let normalizeWs = s => s.replace(wsRe, ' ').trim()

// ── VAD pre-segmentation (opts.vad: true) ──────────────────────────────────────────────
// Cuts long silences out before the audio reaches Whisper (speeds up inference on
// sparse recordings and reduces silence-hallucination risk), then remaps every returned
// timestamp back to the original, un-cut timeline.

async function vadTrim(data) {
	let vadMod
	try { vadMod = await import('@audio/vad') }
	catch { throw new Error('neural-asr: opts.vad requires @audio/vad — npm install @audio/vad') }
	let { active, times, hop } = vadMod.vad(data, { fs: SR })
	let pad = 0.3  // seconds of context kept on each side of a detected speech run, so words at the edges aren't clipped
	let total = data.length / SR
	let regions = []
	for (let i = 0; i < active.length;) {
		if (!active[i]) { i++; continue }
		let j = i
		while (j < active.length && active[j]) j++
		let start = Math.max(0, times[i] - pad)
		let end = Math.min(total, times[j - 1] + hop / SR + pad)
		let last = regions[regions.length - 1]
		if (last && start <= last.end) last.end = Math.max(last.end, end)
		else regions.push({ start, end })
		i = j
	}
	if (!regions.length) return { data: new Float32Array(0), regions: [] }
	let out = new Float32Array(regions.reduce((n, r) => n + Math.round((r.end - r.start) * SR), 0))
	let off = 0
	for (let r of regions) {
		let s = Math.round(r.start * SR), e = Math.round(r.end * SR)
		out.set(data.subarray(s, e), off)
		r.dstStart = off / SR
		off += e - s
	}
	return { data: out, regions }
}

// Map a timestamp in the trimmed (concatenated speech-only) buffer back to original-audio time.
function remapTime(t, regions) {
	for (let r of regions) {
		let dstEnd = r.dstStart + (r.end - r.start)
		if (t <= dstEnd) return round3(r.start + (t - r.dstStart))
	}
	let last = regions[regions.length - 1]
	return round3(last.start + (t - last.dstStart))
}

// ── Input normalization ─────────────────────────────────────────────────────────────────

// Shape/type checks only (no mixing) — cheap enough to run before paying for a model
// load, so a malformed call fails immediately instead of after a multi-second download.
function checkAudio(audio, opts) {
	if (audio instanceof Float32Array) {
		if (!(opts.sampleRate > 0)) throw new Error('neural-asr: opts.sampleRate is required for a raw Float32Array input')
	} else if (Array.isArray(audio)) {
		if (!audio.length || !(audio[0] instanceof Float32Array)) throw new Error('neural-asr: array input must be Float32Array[] (one per channel)')
		if (!(opts.sampleRate > 0)) throw new Error('neural-asr: opts.sampleRate is required for a Float32Array[] input')
	} else if (audio && audio.channelData) {
		if (!(audio.sampleRate > 0)) throw new Error('neural-asr: {channelData, sampleRate} input requires a positive sampleRate')
	} else {
		throw new Error('neural-asr: audio must be Float32Array (mono), Float32Array[] (per-channel), or {channelData, sampleRate}')
	}
}

function resolveAudio(audio, opts) {
	checkAudio(audio, opts)
	if (audio instanceof Float32Array) return { data: mono([audio]), sampleRate: opts.sampleRate }
	if (Array.isArray(audio)) return { data: mono(audio), sampleRate: opts.sampleRate }
	return { data: mono(audio.channelData), sampleRate: audio.sampleRate }
}

// ── Pipeline lifecycle ──────────────────────────────────────────────────────────────────

async function defaultCacheDir() {
	// transformers.js defaults env.cacheDir to a `.cache/` folder next to its own installed
	// location — buried in node_modules and wiped (and re-downloaded) on every reinstall.
	// Share the @audio/neural lane's cache root instead (same one @audio/neural-runtime uses
	// for raw ONNX bytes, $AUDIO_NEURAL_CACHE-overridable), under its own `hf/` subdir since
	// transformers.js manages a Hugging-Face-shaped tree, not neural-runtime's sha256 keying.
	let [{ default: os }, { default: path }] = await Promise.all([import('node:os'), import('node:path')])
	let root = process.env.AUDIO_NEURAL_CACHE || path.join(os.homedir(), '.cache', 'audiojs', 'neural')
	return path.join(root, 'hf')
}

async function createPipeline(model, opts = {}) {
	if (opts.cache) env.cacheDir = opts.cache
	else if (IS_NODE) env.cacheDir = await defaultCacheDir()
	let dtype = opts.dtype ?? (IS_NODE ? 'fp32' : 'q8')
	// fp32 in Node: onnxruntime-node's CPU EP has full fp32 kernel coverage and no bandwidth
	// constraint (weights are already on local disk), so full precision costs nothing extra
	// and avoids q8's small accuracy loss. q8 in the browser: the model ships over the wire
	// and the dtype directly sets what gets downloaded — the ~4x smaller transfer and
	// wasm/WebGPU-friendly int8 kernels matter more there than the last bit of accuracy.
	return pipeline('automatic-speech-recognition', opts.model ?? model, {
		dtype,
		device: opts.device ?? 'auto',
		progress_callback: opts.progress,
	})
}

async function runTranscribe(pipe, audio, opts = {}) {
	let { data: mixed, sampleRate } = resolveAudio(audio, opts)
	let data16k = to16k(mixed, sampleRate)
	// *.en checkpoints have no language/task tokens in their vocab at all — the model's own
	// generate() throws if either is passed (even 'transcribe', the otherwise-harmless default).
	let englishOnly = /\.en$/.test(String(opts.model))

	let regions = null
	if (opts.vad) {
		let trimmed = await vadTrim(data16k)
		data16k = trimmed.data
		regions = trimmed.regions
	}
	let clampDuration = round3(data16k.length / SR)

	// VAD found no speech at all — skip inference (also sidesteps Whisper's own
	// silence-hallucination behaviour, see README) and report an empty transcript.
	if (opts.vad && data16k.length === 0) {
		let result = { text: '', language: opts.language ?? (englishOnly ? 'en' : null), segments: [] }
		if (opts.timestamps === 'word') result.words = []
		if (opts.cues) result.cues = await buildCues(result)
		return result
	}

	let timestamps = opts.timestamps ?? 'segment'
	let return_timestamps = timestamps === false ? false : timestamps === 'word' ? 'word' : true

	let raw = await pipe(data16k, {
		chunk_length_s: opts.chunk ?? 30,
		stride_length_s: opts.stride ?? 5,
		return_timestamps,
		// transformers.js's ASR pipeline ships no anti-repetition generation defaults
		// (`_default_generation_config = {}` upstream, greedy decoding otherwise) — on audio
		// the model finds hard (music, non-target-language speech, noise) that reliably
		// degenerates into an infinite repeated-token loop instead of stopping. This is the
		// one generation param OpenAI's own decoder loop gets for free from beam search /
		// temperature fallback that we don't; blocking repeated 3-grams is a cheap, narrow
		// substitute that fixes the loop without changing well-behaved output.
		no_repeat_ngram_size: 3,
		...(englishOnly ? {} : { language: opts.language, task: opts.task ?? 'transcribe' }),
	})

	let words, segments
	if (return_timestamps === 'word') {
		words = toSegments(raw.chunks, clampDuration)
		segments = segmentsFromWords(words)
	} else if (return_timestamps === true) {
		segments = toSegments(raw.chunks, clampDuration)
	} else {
		segments = [{ start: 0, end: clampDuration, text: normalizeWs(raw.text) }]
	}

	if (regions) {
		let remap = seg => ({ ...seg, start: remapTime(seg.start, regions), end: remapTime(seg.end, regions) })
		segments = segments.map(remap)
		if (words) words = words.map(remap)
	}

	let text = normalizeWs(segments.map(s => s.text).join(' '))
	// Whisper-tiny/base are English-only models by vocabulary construction (no language token) —
	// safe to report 'en' without a detection step. Multilingual models: the caller's own
	// `language` opt is echoed back (they told us); on auto-detect, transformers.js's public
	// ASR pipeline (v4.x) does not currently surface the token the model predicted internally
	// (see README — Limitations) so `language` is `null` rather than a guess.
	let language = opts.language ?? (englishOnly ? 'en' : null)

	let result = { text, language, segments }
	if (words) result.words = words

	if (opts.cues) result.cues = await buildCues(result)

	return result
}

async function buildCues({ segments, words }) {
	let subtitle
	try { subtitle = await import('@audio/subtitle') }
	catch { throw new Error('neural-asr: opts.cues requires @audio/subtitle — npm install @audio/subtitle') }
	if (words?.length) return subtitle.fromWords(words)
	return segments.map(({ start, end, text }) => ({ start, end, text }))
}

/** transcribe(audio, opts) → { text, language, segments, words?, cues? } — one-shot: loads
 * the model, transcribes, frees it. For repeated calls against the same model, use
 * `loadModel` to keep the pipeline warm. */
export default async function transcribe(audio, opts = {}) {
	checkAudio(audio, opts)
	let model = opts.model ?? 'onnx-community/whisper-base'
	let pipe = await createPipeline(model, opts)
	try {
		return await runTranscribe(pipe, audio, { ...opts, model })
	} finally {
		await pipe.dispose()
	}
}

/** loadModel(model, opts) → { transcribe(audio, opts), free() } — keeps the pipeline
 * (and its ONNX Runtime session) warm across calls; call free() to release it. */
export async function loadModel(model = 'onnx-community/whisper-base', opts = {}) {
	let pipe = await createPipeline(model, opts)
	let freed = false
	return {
		transcribe: (audio, callOpts = {}) => {
			if (freed) throw new Error('neural-asr: transcribe() called after free()')
			return runTranscribe(pipe, audio, { ...opts, ...callOpts, model })
		},
		free: async () => {
			if (freed) return
			freed = true
			await pipe.dispose()
		},
	}
}
