// @audio/neural-tts — text to speech, running locally in Node and the browser via
// @huggingface/transformers (Apache-2.0) / ONNX Runtime, the same transformers.js pattern
// as @audio/neural-asr, @audio/neural-align, @audio/neural-diarize.
//
// Default model: SpeechT5 (Microsoft, MIT — Ao et al. 2022, "SpeechT5: Unified-Modal
// Encoder-Decoder Pre-Training for Spoken Language Processing",
// https://arxiv.org/abs/2110.07205), an autoregressive text→mel-spectrogram model paired
// with a HiFi-GAN vocoder (Kong et al. 2020, https://arxiv.org/abs/2010.05646) and a
// speaker x-vector (512-d) that selects the voice. Default voice is the CMU ARCTIC "slt"
// speaker embedding transformers.js's own docs/examples use
// (https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/speaker_embeddings.bin,
// derived from the Matthijs/cmu-arctic-xvectors dataset's cmu_us_slt_arctic recording).
//
// This wraps the model classes directly (AutoTokenizer + SpeechT5ForTextToSpeech +
// SpeechT5HifiGan), not @huggingface/transformers' pipeline('text-to-speech', …) — that
// convenience wrapper throws "Missing the following inputs: speaker_embeddings,
// encoder_hidden_states, output_sequence" against Xenova/speecht5_tts on transformers.js
// 4.2.0 (reproduced during development; the direct model-class call in SpeechT5ForTextToSpeech's
// own JSDoc example works correctly). A second, separate issue was found the same way:
// passing device:'auto' explicitly to from_pretrained() (this lane's usual default) corrupts
// a *later* generate_speech() call — see the comment on createModel() below. Both filed as
// known issues — see README.
//
// SpeechT5 runs at a fixed 16 kHz; `sampleRate` resamples the joined output afterwards with
// @audio/resample-sinc, the same post-hoc-resample shape as @audio/neural-asr's input side.

import sinc from '@audio/resample-sinc'
import { AutoTokenizer, SpeechT5ForTextToSpeech, SpeechT5HifiGan, VitsModel, Tensor, env } from '@huggingface/transformers'
import os from 'node:os'
import path from 'node:path'

const IS_NODE = typeof process !== 'undefined' && !!process.versions?.node

const DEFAULT_MODEL = 'Xenova/speecht5_tts'
const DEFAULT_VOCODER = 'Xenova/speecht5_hifigan'
const SPEECHT5_SR = 16000 // fixed native rate, baked into the pretrained model/vocoder pair
// CMU ARCTIC "slt" speaker x-vector — the voice transformers.js's own SpeechT5 docs and
// examples default to. See file header for provenance.
const DEFAULT_VOICE = 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/speaker_embeddings.bin'

// SpeechT5 is autoregressive (one mel frame at a time) and degrades on very long inputs —
// splitting at sentence boundaries keeps each call in the regime the model was trained on.
// 600 is this package's own conservative choice (SpeechT5's own docs don't give a hard
// limit); see README for the measured behavior on a 2000-char input.
const MAX_CHARS = 600
const SILENCE_MS = 150

/** Curated model table — see README for the license/quality notes behind each row. */
export const models = [
	{ id: 'Xenova/speecht5_tts', family: 'speecht5', license: 'MIT', sampleRate: 16000, languages: 'en', voices: '512-d x-vector Float32Array or URL; default the CMU ARCTIC "slt" speaker' },
	{ id: 'Xenova/mms-tts-eng', family: 'vits', license: 'CC-BY-NC-4.0', sampleRate: 16000, languages: 'en', voices: 'single built-in voice, no speaker embedding' },
]

// ── Lane-wide model cache (matches @audio/neural-diarize, @audio/neural-runtime) ────────

function defaultCacheDir() {
	return path.join(process.env.AUDIO_NEURAL_CACHE || path.join(os.homedir(), '.cache', 'audiojs', 'neural'), 'hf')
}
if (IS_NODE) env.cacheDir = defaultCacheDir()

// ── Pure helpers (no model, no network — see test.js's smoke test) ──────────────────────

// Splits on sentence-ending punctuation (. ! ?) followed by whitespace or end-of-string,
// keeping the punctuation with its sentence. A sentence still over `maxChars` after that
// is further broken at word boundaries (never mid-word) so no single call ever exceeds it.
let sentenceRe = /[^.!?]+[.!?]+(?:\s+|$)|[^.!?]+$/g

/** Split text into ≤`maxChars` chunks, first at sentence punctuation, then at word boundaries. */
export function splitSentences(text, maxChars = MAX_CHARS) {
	if (!text) return []
	let sentences = (text.match(sentenceRe) ?? [text]).map(s => s.trim()).filter(Boolean)
	let out = []
	for (let s of sentences) {
		if (s.length <= maxChars) { out.push(s); continue }
		let words = s.split(/\s+/)
		let cur = ''
		for (let w of words) {
			if (cur && cur.length + 1 + w.length > maxChars) { out.push(cur); cur = w }
			else cur = cur ? cur + ' ' + w : w
		}
		if (cur) out.push(cur)
	}
	return out
}

/** Concatenate audio chunks with `gapMs` of silence between each (not before the first or after the last). */
export function joinWithSilence(chunks, sampleRate, gapMs = SILENCE_MS) {
	if (!chunks.length) return new Float32Array(0)
	if (chunks.length === 1) return Float32Array.from(chunks[0])
	let gap = Math.round(sampleRate * gapMs / 1000)
	let total = chunks.reduce((s, c) => s + c.length, 0) + gap * (chunks.length - 1)
	let out = new Float32Array(total)
	let off = 0
	for (let i = 0; i < chunks.length; i++) {
		out.set(chunks[i], off)
		off += chunks[i].length
		if (i < chunks.length - 1) off += gap
	}
	return out
}

// ── Model lifecycle ───────────────────────────────────────────────────────────────────

async function createModel(model, opts = {}) {
	if (opts.cache) env.cacheDir = opts.cache
	else if (IS_NODE) env.cacheDir = defaultCacheDir()
	let dtype = opts.dtype ?? (IS_NODE ? 'fp32' : 'q8') // same fp32-in-Node / q8-in-browser split as neural-asr/neural-diarize
	let info = models.find(m => m.id === model)
	let family = info?.family ?? 'speecht5' // unknown model ids assumed SpeechT5-shaped — same checkpoint architecture, different weights

	// `device` is forwarded to from_pretrained() only when the caller explicitly asks for one.
	// Passing device:'auto' explicitly — the seemingly-harmless default this lane's other atoms
	// use — was observed to corrupt a *later* generate_speech() call in this transformers.js
	// 4.2.0 / onnxruntime-node combo: "Non-zero status code ... Invalid dimension ...
	// SizeToDimension" deep inside a MatMul, on the second sentence of the very first speak()
	// call, reproducible outside tst with this package's own direct (non-pipeline) loading of
	// three onnxruntime-node sessions (SpeechT5 encoder + decoder_model_merged + vocoder).
	// Omitting `device` — letting transformers.js apply its own ambient default — does not hit
	// this, confirmed by isolating the difference against a minimal repro script; explicit
	// device:'cpu' also works. Filed as a known issue — see README.
	let loadOpts = { dtype, progress_callback: opts.progress }
	if (opts.device) loadOpts.device = opts.device

	if (family === 'vits') {
		let tokenizer = await AutoTokenizer.from_pretrained(model)
		let vits = await VitsModel.from_pretrained(model, loadOpts)
		return { family, tokenizer, vits, sampleRate: vits.config.sampling_rate }
	}

	let tokenizer = await AutoTokenizer.from_pretrained(model)
	let t2s = await SpeechT5ForTextToSpeech.from_pretrained(model, loadOpts)
	let vocoder = await SpeechT5HifiGan.from_pretrained(opts.vocoder ?? DEFAULT_VOCODER, loadOpts)
	return { family, tokenizer, t2s, vocoder, sampleRate: SPEECHT5_SR }
}

async function disposeModel(m) {
	if (m.family === 'vits') await m.vits.dispose?.()
	else await Promise.all([m.t2s.dispose?.(), m.vocoder.dispose?.()])
}

let voiceCache = new Map() // URL → Tensor, across calls on the same loaded model

async function resolveVoice(voice) {
	if (voice == null) voice = DEFAULT_VOICE
	if (voice instanceof Float32Array) return new Tensor('float32', voice, [1, voice.length])
	if (voice instanceof Tensor) return voice
	if (typeof voice === 'string') {
		let cached = voiceCache.get(voice)
		if (cached) return cached
		let data = new Float32Array(await (await fetch(voice)).arrayBuffer())
		let t = new Tensor('float32', data, [1, data.length])
		voiceCache.set(voice, t)
		return t
	}
	throw new Error('neural-tts: opts.voice must be a Float32Array (512-d x-vector), a Tensor, or a URL string')
}

async function synthOne(m, text, opts) {
	if (m.family === 'vits') {
		let inputs = m.tokenizer(text)
		let { waveform } = await m.vits(inputs) // dims [1, N]
		return Float32Array.from(waveform.data)
	}
	let voice = await resolveVoice(opts.voice)
	let { input_ids } = m.tokenizer(text)
	let { waveform } = await m.t2s.generate_speech(input_ids, voice, { vocoder: m.vocoder }) // dims [N]
	return Float32Array.from(waveform.data)
}

async function runSpeak(m, text, opts = {}) {
	if (typeof text !== 'string' || !text.trim()) throw new Error('neural-tts: text must be a non-empty string')
	let sentences = splitSentences(text, opts.maxChars ?? MAX_CHARS)
	let chunks = []
	for (let i = 0; i < sentences.length; i++) {
		chunks.push(await synthOne(m, sentences[i], opts))
		opts.progress?.({ status: 'speak', index: i, total: sentences.length })
	}
	let joined = joinWithSilence(chunks, m.sampleRate, opts.silenceGap ?? SILENCE_MS)
	let sampleRate = opts.sampleRate ?? m.sampleRate
	let data = sampleRate === m.sampleRate ? joined : sinc(joined, { from: m.sampleRate, to: sampleRate })
	return { channelData: [data], sampleRate, duration: data.length / sampleRate }
}

/** tts(text, opts) → { channelData: [Float32Array], sampleRate, duration } — one-shot:
 * loads the model, speaks, frees it. For repeated calls, use `loadModel` instead. */
export default async function tts(text, opts = {}) {
	let model = opts.model ?? DEFAULT_MODEL
	let m = await createModel(model, opts)
	try {
		return await runSpeak(m, text, opts)
	} finally {
		await disposeModel(m)
	}
}

/** loadModel(model, opts) → { speak(text, opts), free() } — keeps the model (and its ONNX
 * Runtime sessions) warm across calls; call free() to release it. */
export async function loadModel(model = DEFAULT_MODEL, opts = {}) {
	let m = await createModel(model, opts)
	let freed = false
	return {
		speak: (text, callOpts = {}) => {
			if (freed) throw new Error('neural-tts: speak() called after free()')
			return runSpeak(m, text, { ...opts, ...callOpts })
		},
		free: async () => {
			if (freed) return
			freed = true
			await disposeModel(m)
		},
	}
}
