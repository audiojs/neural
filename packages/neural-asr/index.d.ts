/** Whisper speech-to-text, running locally via @huggingface/transformers (ONNX Runtime). */

/** Mono Float32Array; per-channel Float32Array[]; or a decode()-shaped {channelData, sampleRate}. */
export type AsrAudioInput =
	| Float32Array
	| Float32Array[]
	| { channelData: Float32Array[], sampleRate: number }

export interface AsrSegment {
	/** seconds, 3-decimal precision */
	start: number
	/** seconds, 3-decimal precision */
	end: number
	text: string
}

export interface AsrProgressInfo {
	status: string
	name?: string
	file?: string
	progress?: number
	loaded?: number
	total?: number
}

export interface AsrOptions {
	/** required unless `audio` is the {channelData, sampleRate} form */
	sampleRate?: number
	/** HF model id or local path. Default 'onnx-community/whisper-base'. See `models` for a curated list. */
	model?: string
	/** Whisper language name/code (e.g. 'en', 'german'). NOT auto-detected if omitted — transformers.js's ASR
	 * pipeline silently assumes English when `language` is unset (see README — Limitations); pass it explicitly
	 * for non-English audio on a multilingual model. */
	language?: string
	task?: 'transcribe' | 'translate'
	/** 'segment' (default): Whisper's own chunk timestamps. 'word': word-level timestamps (segments are then
	 * reconstructed by grouping words at sentence-ending punctuation — see README). false: no timestamps, one
	 * segment spanning the whole input. */
	timestamps?: 'segment' | 'word' | false
	/** long-audio chunk window, seconds. Default 30 (Whisper's own training window). */
	chunk?: number
	/** overlap between chunk windows, seconds. Default 5. */
	stride?: number
	/** ONNX Runtime numeric precision. Default 'fp32' in Node, 'q8' in the browser — see README. */
	dtype?: 'auto' | 'fp32' | 'fp16' | 'q8' | 'int8' | 'q4' | 'q4f16' | 'bnb4'
	/** Default 'auto' (transformers.js picks cpu/wasm/webgpu per environment). */
	device?: 'auto' | 'cpu' | 'wasm' | 'webgpu' | 'gpu' | string
	/** Model download / load progress. */
	progress?: (info: AsrProgressInfo) => void
	/** Override the on-disk model cache directory (Node only). Default `~/.cache/audiojs/neural/hf`
	 * (`$AUDIO_NEURAL_CACHE`-overridable — the @audio/neural lane's shared cache root). */
	cache?: string
	/** Pre-cut long silences with @audio/vad before transcribing (optional peer dep; dynamically imported),
	 * then remap returned timestamps back to the original timeline. Default false. */
	vad?: boolean
	/** Also return `cues`: @audio/subtitle Cue[] built from `words` (if requested) or `segments` (optional peer dep). */
	cues?: boolean
}

export interface AsrResult {
	/** normalized-whitespace join of `segments[].text` */
	text: string
	/** echoed `opts.language`, 'en' for *.en models, else null (see README — Limitations) */
	language: string | null
	segments: AsrSegment[]
	/** present only when `timestamps: 'word'` */
	words?: AsrSegment[]
	/** present only when `opts.cues` is true */
	cues?: Array<{ start: number, end: number, text: string, words?: AsrSegment[] }>
}

/** Mix any channel layout down to mono (equal-weight average). Returns the input unchanged if already mono. */
export function mono(channelData: Float32Array[]): Float32Array

/** Resample mono data to Whisper's fixed 16 kHz input rate (no-op if already 16 kHz). */
export function to16k(data: Float32Array, sampleRate: number): Float32Array

/** Map transformers.js's Whisper `chunks` to {start,end,text} segments; null end timestamps clamp to `duration`. */
export function toSegments(chunks: Array<{ text: string, timestamp: [number | null, number | null] }>, duration: number): AsrSegment[]

export interface AsrModel {
	transcribe(audio: AsrAudioInput, opts?: AsrOptions): Promise<AsrResult>
	free(): Promise<void>
}

/** Load a Whisper pipeline once and reuse it (keeps the ONNX Runtime session warm) across calls. */
export function loadModel(model?: string, opts?: AsrOptions): Promise<AsrModel>

export interface AsrModelInfo {
	id: string
	params: string
	languages: 'en' | 'multi'
	/** approximate ONNX download size, bytes — see README for how each row was measured/estimated */
	size: number
}

/** Curated table of onnx-community Whisper conversions (all MIT-licensed OpenAI weights). */
export const models: AsrModelInfo[]

/** transcribe(audio, opts) → one-shot: loads the model, transcribes, frees it.
 * For repeated calls against the same model, use `loadModel` instead. */
export default function transcribe(audio: AsrAudioInput, opts?: AsrOptions): Promise<AsrResult>
