/** ONNX inference adapter — one interface over onnxruntime-node and onnxruntime-web. */

export type Backend = 'auto' | 'node' | 'wasm' | 'webgpu' | 'webgl'

export type TensorType = 'float32' | 'int32' | 'int64' | 'uint8' | 'bool' | 'string'

/** Plain tensor — no ORT dependency at call time; converted to ort.Tensor inside run(). */
export interface Tensor {
  data: Float32Array | Int32Array | BigInt64Array | Uint8Array | string[]
  dims: number[]
  type: TensorType
}

export interface ValueInfo {
  name: string
  /** absent when the installed onnxruntime version doesn't expose shape/type metadata */
  dims?: (number | string)[]
  type?: string
}

export interface Progress {
  loaded: number
  /** null when the server didn't send a Content-Length */
  total: number | null
}

export interface FetchOptions {
  /** Node: ~/.cache/audiojs/neural/<sha256(url)> (override with $AUDIO_NEURAL_CACHE); browser: Cache API store 'audio-neural'. Default true. */
  cache?: boolean
  progress?: (p: Progress) => void
  /** default: the ambient fetch */
  fetch?: typeof fetch
}

export interface LoadOptions extends FetchOptions {
  /** default 'auto': 'node' in Node, 'wasm' in a browser/worker/worklet */
  backend?: Backend
  /** onnxruntime-web wasm backend only */
  threads?: number
  /** passthrough to ort.InferenceSession.create's executionProviders */
  executionProviders?: string[]
  graphOptimizationLevel?: 'all' | 'basic' | 'disabled'
  /** passthrough to ort.env.logLevel */
  logLevel?: 'verbose' | 'info' | 'warning' | 'error' | 'fatal'
  /** passthrough to ort.env.wasm.wasmPaths — where onnxruntime-web loads its .wasm from; required in a worklet, which can't resolve a relative default */
  wasmPaths?: string | Record<string, string>
}

export interface Session {
  run(feeds: Record<string, Tensor>, outputNames?: string[]): Promise<Record<string, Tensor>>
  inputs: ValueInfo[]
  outputs: ValueInfo[]
  backend: Backend
  /** releases the underlying ORT session; idempotent. run() after free() throws. */
  free(): void
}

/** Load an ONNX model and get back a ready-to-run session. */
export function load(model: string | Uint8Array | ArrayBuffer, opts?: LoadOptions): Promise<Session>

/** Build a plain tensor; infers `type` from `data`'s constructor when omitted (Float32Array → 'float32', Int32Array → 'int32', BigInt64Array → 'int64', Uint8Array → 'uint8', string[] → 'string'). */
export function tensor(data: Tensor['data'], dims: number[], type?: TensorType): Tensor

/** Backends this environment can run: Node → ['node']; browser → ['wasm'] (+ 'webgpu' when navigator.gpu exists). */
export function backends(): Promise<Backend[]>

/** Cached fetch with progress. Also the primitive fetchJson uses for tokenizer/config siblings. */
export function fetchModel(url: string, opts?: FetchOptions): Promise<Uint8Array>

/** Cached fetch + JSON.parse, same cache/progress path as fetchModel. */
export function fetchJson(url: string, opts?: FetchOptions): Promise<unknown>
