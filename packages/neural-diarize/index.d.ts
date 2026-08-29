/** Speaker diarization — who spoke when, via WavLM x-vector embeddings and agglomerative clustering. */

/** Mono Float32Array; per-channel Float32Array[]; or a decode()-shaped {channelData, sampleRate}. */
export type DiarizeAudioInput =
	| Float32Array
	| Float32Array[]
	| { channelData: Float32Array[], sampleRate: number }

export interface DiarizeSegment {
	/** seconds */
	start: number
	/** seconds */
	end: number
	/** 'S0', 'S1', … — S0 is whichever speaker's first window comes first chronologically */
	speaker: string
	/** cosine similarity of this segment's windows to their assigned cluster centroid, averaged, in [0,1] — a confidence signal, not a probability */
	score?: number
}

export interface DiarizeProgressInfo {
	status: string
	name?: string
	file?: string
	progress?: number
	loaded?: number
	total?: number
	index?: number
	total_?: number
}

export interface DiarizeOptions {
	/** required unless `audio` is the {channelData, sampleRate} form */
	sampleRate?: number
	/** HF model id or local path. Default 'Xenova/wavlm-base-plus-sv'. */
	model?: string
	/** exact speaker count — when given, clustering cuts the dendrogram at this many clusters instead of using `threshold` */
	speakers?: number
	/** lower bound on cluster count when `speakers` is not given. Default 1. */
	minSpeakers?: number
	/** upper bound on cluster count when `speakers` is not given — clustering keeps merging past `threshold` to respect this. Default 8. */
	maxSpeakers?: number
	/** cosine-similarity stop threshold for clustering (ignored when `speakers` is given). Default 0.75 — see README for how this was chosen. */
	threshold?: number
	/** embedding window length, seconds. Default 1.5. */
	window?: number
	/** embedding window hop, seconds. Default 0.75. */
	hop?: number
	/** minimum output segment length, seconds — also the minimum length a hangover-merged VAD region must reach to count as speech. Default 0.5. */
	minSegment?: number
	/** include the raw per-window embeddings in the result. Default false. */
	embeddings?: boolean
	/** ONNX Runtime numeric precision. Default 'fp32' in Node, 'q8' in the browser. */
	dtype?: 'auto' | 'fp32' | 'fp16' | 'q8' | 'int8' | 'q4' | 'q4f16' | 'bnb4'
	/** Default 'auto' (transformers.js picks cpu/wasm/webgpu per environment). */
	device?: 'auto' | 'cpu' | 'wasm' | 'webgpu' | 'gpu' | string
	/** Model download / load progress, and one call per embedded window ({status:'embed', index, total}). */
	progress?: (info: DiarizeProgressInfo) => void
	/** Override the on-disk model cache directory (Node only). Default `$AUDIO_NEURAL_CACHE/hf` or `~/.cache/audiojs/neural/hf`. */
	cache?: string
}

export interface DiarizeResult {
	segments: DiarizeSegment[]
	/** distinct speaker count actually found */
	speakers: number
	/** present only when `opts.embeddings` is true — one 512-d Float32Array per embedding window (not per segment) */
	embeddings?: Float32Array[]
}

/** Mix any channel layout down to mono (equal-weight average). Returns the input unchanged if already mono. */
export function mono(channelData: Float32Array[]): Float32Array

/** Resample mono data to WavLM's fixed 16 kHz input rate (no-op if already 16 kHz). */
export function to16k(data: Float32Array, sampleRate: number): Float32Array

export interface ClusterOptions {
	threshold?: number
	speakers?: number
	minSpeakers?: number
	maxSpeakers?: number
}

/**
 * Agglomerative hierarchical clustering (average linkage / UPGMA on cosine distance) over
 * a plain array of embeddings — no model involved, deterministic, directly testable.
 */
export function cluster(embeddings: Float32Array[], opts?: ClusterOptions): Int32Array

export interface DiarizeWindow {
	start: number
	end: number
}

export interface ToSegmentsOptions {
	minSegment?: number
	/** per-window confidence (same length/order as `labels`/`windows`), averaged into each output segment's `score` */
	scores?: number[]
}

/**
 * Collapse per-window speaker labels into segments: median-filters isolated single-window
 * flips, merges consecutive same-speaker windows, splits speaker changes at the midpoint
 * between window centers, and merges any segment shorter than `minSegment` into its longer
 * same-run neighbour.
 */
export function toSegments(labels: ArrayLike<number>, windows: DiarizeWindow[], opts?: ToSegmentsOptions): DiarizeSegment[]

export interface SubtitleCue {
	start: number
	end: number
	text: string
	[key: string]: unknown
}

/**
 * Prefix each cue's text with a WebVTT voice span (`<v S0>…</v>`) naming the diarization
 * segment it overlaps most, by time overlap. `cues` is any `{start, end, text}[]` —
 * @audio/neural-asr's `segments`/`words`, @audio/subtitle's `Cue[]`, or your own.
 */
export function toSubtitles<T extends SubtitleCue>(segments: DiarizeSegment[], cues: T[]): T[]

export interface DiarizeModel {
	diarize(audio: DiarizeAudioInput, opts?: DiarizeOptions): Promise<DiarizeResult>
	free(): Promise<void>
}

/** Load the WavLM x-vector model once and reuse it (keeps the ONNX Runtime session warm) across calls. */
export function loadModel(model?: string, opts?: DiarizeOptions): Promise<DiarizeModel>

/** diarize(audio, opts) → one-shot: loads the model, diarizes, frees it.
 * For repeated calls, use `loadModel` instead. */
export default function diarize(audio: DiarizeAudioInput, opts?: DiarizeOptions): Promise<DiarizeResult>
