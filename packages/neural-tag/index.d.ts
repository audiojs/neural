/**
 * What a sound is, in AudioSet's 521 classes: Google's YAMNet (tensorflow/models, Apache-2.0) through
 * @audio/neural-runtime, its log-mel features in JS as upstream's features.py computes them.
 */

export interface Options {
	/** seconds between patches (0.48), whole 10 ms frames: a longer hop scores fewer patches of a long sound */
	hop?: number
	/** the model: bytes, a URL, or in Node a directory or file path holding yamnet.fp16.onnx; default the neural cache's copy, else the hosted one (once REVISION is set) */
	weights?: string | URL | Uint8Array | ArrayBuffer
	/** @audio/neural-runtime's backend: 'auto' (default), 'node', 'wasm', 'webgpu', 'coreml' */
	backend?: string
	/** ONNX Runtime session options, e.g. { intraOpNumThreads: 1 } */
	sessionOptions?: Record<string, unknown>
	progress?: (p: { loaded: number, total: number | null }) => void
	fetch?: typeof fetch
}

export interface Tag {
	/** the class's name in yamnet_class_map.csv, e.g. 'Speech' */
	name: string
	/** its AudioSet ontology id, e.g. '/m/09x0r' */
	id: string
	/** its place in YAMNet's output, 0 to 520 */
	index: number
	/** its mean score over the sound's patches, 0 to 1: a sigmoid, not a probability calibrated across classes */
	score: number
}

/** a mono Float32Array, or one array per channel (mixed down to their mean) */
export type Samples = Float32Array | ArrayLike<number> | Float32Array[]

/** The classes of `x` ranked by their mean score over its patches, best first: `top` of them (5; Infinity for all 521) */
export function tag(x: Samples, sampleRate?: number, opts?: Options & { top?: number }): Promise<Tag[]>

/** Every patch's scores: one Float32Array(521) a patch; each patch hears 0.975 s from its time */
export function scores(x: Samples, sampleRate?: number, opts?: Options): Promise<{ scores: Float32Array[], hop: number, times: number[] }>

/** YAMNet's input from 16 kHz mono samples: log-mel patches of 96 frames × 64 bands, frame-major */
export function features(x: Float32Array, opts?: { hop?: number }): { data: Float32Array, patches: number }

/** The 521 classes in output order: AudioSet ontology ids and names (yamnet_class_map.csv) */
export const CLASSES: { id: string, name: string }[]

/** YAMNet's rate, and a patch's frames and mel bands */
export const SAMPLE_RATE: 16000
export const FRAMES: 96
export const BANDS: 64

/** The commit of audiojs/yamnet this version reads ('' until the file is uploaded), and the file's SHA-256 */
export const REVISION: string
export const SHA256: string
/** The hosted model at REVISION, or null until it is set */
export const MODEL: URL | null
