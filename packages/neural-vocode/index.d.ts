/**
 * A sound made again from its mel spectrogram by Vocos (charactr/vocos-mel-24khz, MIT), through @audio/neural-runtime:
 * the features and the iSTFT in JS as Vocos computes them, the network in ONNX.
 */

export interface Options {
	/** rate of the samples given and returned (24000): others are resampled to Vocos's 24 kHz and back */
	sampleRate?: number
	/** the model: bytes, a URL, or in Node a directory or file path holding vocos.fp16.onnx; default the neural cache's copy, else the hosted one (REVISION), fetched into the cache */
	weights?: string | URL | Uint8Array | ArrayBuffer
	/** @audio/neural-runtime's backend: 'auto' (default), 'node', 'wasm', 'webgpu', 'coreml' */
	backend?: string
	/** frames a run (1024), each with 32 frames of context either side: the same frames as one run of all */
	chunk?: number
	progress?: (p: { loaded: number, total: number | null }) => void
	fetch?: typeof fetch
}

/** Vocos's rate, frame, hop and mel bands */
export const SAMPLE_RATE: 24000
export const N: 1024
export const HOP: 256
export const MELS: 100

/** The commit of audiojs/vocos-mel-24khz this version reads, and its file's SHA-256 */
export const REVISION: string
export const SHA256: string
/** The hosted model, at REVISION */
export const MODEL: URL

/** `x` (mono) made again by Vocos, at `sampleRate` and as long */
export default function vocode(x: Float32Array | ArrayLike<number>, opts?: Options): Promise<Float32Array>

/** Vocos's features of 24 kHz mono samples: log of a 100-band HTK mel spectrogram, band-major */
export function mel(x: Float32Array): { data: Float32Array, frames: number }

/** The complex STFT frames Vocos predicts from mel(): bin-major, 513 bins */
export function frames(mel: { data: Float32Array, frames: number }, opts?: Options): Promise<{ re: Float32Array, im: Float32Array }>

/** torch.istft of such frames: Hann of 1024, hop 256, centered; `length` samples (default 256 · (frames − 1)) */
export function istft(re: Float32Array, im: Float32Array, frames: number, length?: number): Float32Array
