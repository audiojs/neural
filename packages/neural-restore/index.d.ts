export const SAMPLE_RATE: 48000
export const N: 2048
export const HOP: 512
export const BINS: 1025
export const LEVEL: -20
export const FILE: string
export const REVISION: string
export const SHA256: string
/** The hosted model at REVISION, or null while REVISION is empty */
export const MODEL: URL | null
export interface RestoreOptions {
  /** the input's rate, Hz (48000) */
  sampleRate?: number
  /** the model: bytes, a URL, or in Node a file or directory; default the neural cache's restore/restore.fp16.onnx when of SHA256, else the hosted one (MODEL), fetched into the cache */
  weights?: string | URL | Uint8Array | ArrayBuffer
  /** 'auto' (WebGPU in a page that has it, else wasm; Node's runtime in Node), 'wasm', 'webgpu', 'node', 'coreml' */
  backend?: string
  progress?: (p: { done?: number, total?: number, loaded?: number }) => void
  fetch?: typeof fetch
}
export function stft(x: Float32Array): { data: Float32Array, frames: number }
export function istft(spec: Float32Array, frames: number, length: number): Float32Array
export function session(opts?: RestoreOptions): Promise<{ backend: string, run(spec: Float32Array, frames: number): Promise<Float32Array> }>
export default function restore(x: Float32Array | ArrayLike<number>, opts?: RestoreOptions): Promise<Float32Array>
