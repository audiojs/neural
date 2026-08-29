/** Text to speech — SpeechT5 (default) and VITS/mms-tts-eng, running locally via @huggingface/transformers (ONNX Runtime). */

export interface TtsProgressInfo {
	status: string
	name?: string
	file?: string
	progress?: number
	loaded?: number
	total?: number
	index?: number
	total_?: number
}

export interface TtsOptions {
	/** HF model id or local path. Default 'Xenova/speecht5_tts'. See `models` for the curated table. */
	model?: string
	/** SpeechT5 only: 512-d speaker x-vector (Float32Array), a URL/path to fetch one from, or a raw Tensor.
	 * Default the CMU ARCTIC "slt" speaker embedding transformers.js's own docs use. Ignored for VITS models
	 * (Xenova/mms-tts-eng), which have one built-in voice. */
	voice?: Float32Array | string
	/** SpeechT5 only: override the vocoder model id. Default 'Xenova/speecht5_hifigan'. */
	vocoder?: string
	/** Resample the joined output to this rate with @audio/resample-sinc. Default the model's native rate (16000). */
	sampleRate?: number
	/** Max characters per model call before splitting at a further word boundary. Default 600. */
	maxChars?: number
	/** Silence gap inserted between synthesized sentences, milliseconds. Default 150. */
	silenceGap?: number
	/** ONNX Runtime numeric precision. Default 'fp32' in Node, 'q8' in the browser. */
	dtype?: 'auto' | 'fp32' | 'fp16' | 'q8' | 'int8' | 'q4' | 'q4f16' | 'bnb4'
	/** Left unset by default — transformers.js's own ambient default is used (cpu in Node). Passing 'auto' explicitly
	 * has been observed to corrupt a later `speak()` call in this package's direct (non-pipeline) model loading —
	 * see README. Pass 'cpu'/'wasm'/'webgpu' explicitly if you need to force a backend. */
	device?: 'cpu' | 'wasm' | 'webgpu' | 'gpu' | string
	/** Model download / load progress, and one call per synthesized sentence ({status:'speak', index, total}). */
	progress?: (info: TtsProgressInfo) => void
	/** Override the on-disk model cache directory (Node only). Default `$AUDIO_NEURAL_CACHE/hf` or `~/.cache/audiojs/neural/hf`. */
	cache?: string
}

export interface TtsResult {
	channelData: [Float32Array]
	sampleRate: number
	/** seconds */
	duration: number
}

export interface TtsModelInfo {
	id: string
	family: 'speecht5' | 'vits'
	license: string
	/** native output sample rate, Hz */
	sampleRate: number
	languages: string
	/** human-readable note on how voice selection works for this model */
	voices: string
}

/** Curated table of supported models — see README for the full license/quality notes. */
export const models: TtsModelInfo[]

/** Split text into ≤`maxChars` chunks, first at sentence punctuation, then at word boundaries. Pure, no model. */
export function splitSentences(text: string, maxChars?: number): string[]

/** Concatenate audio chunks with `gapMs` of silence between each. Pure, no model. */
export function joinWithSilence(chunks: Float32Array[], sampleRate: number, gapMs?: number): Float32Array

export interface TtsModel {
	speak(text: string, opts?: TtsOptions): Promise<TtsResult>
	free(): Promise<void>
}

/** Load the model once and reuse it (keeps the ONNX Runtime session(s) warm) across calls. */
export function loadModel(model?: string, opts?: TtsOptions): Promise<TtsModel>

/** tts(text, opts) → one-shot: loads the model, speaks, frees it.
 * For repeated calls, use `loadModel` instead. Throws on an empty/whitespace-only `text`. */
export default function tts(text: string, opts?: TtsOptions): Promise<TtsResult>
