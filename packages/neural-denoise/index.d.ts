/** Neural speech enhancement: RNNoise (pure JS, weights bundled) and DeepFilterNet3 (ONNX, fetched and cached). */

export type Audio = Float32Array | Float32Array[] | { channelData: Float32Array[], sampleRate: number }

export interface Model {
  model: 'rnnoise' | 'deepfilternet3'
  /** algorithmic delay in samples at 48 kHz: 960 (RNNoise), 1440 (DeepFilterNet3); offline results are aligned */
  latency: number
  /** releases the ONNX sessions (DeepFilterNet3); no-op for RNNoise */
  free(): void
}

export interface LoadOptions {
  /** RNNoise: bytes of an upstream weight blob (dump_weights_blob), default the bundled model. DeepFilterNet3: URL or bytes of an upstream ONNX export (.tar.gz), default MODEL */
  weights?: string | Uint8Array | ArrayBuffer
  /** DeepFilterNet3: @audio/neural-runtime backend, 'node' | 'wasm' | 'webgpu'; default 'auto' */
  device?: 'auto' | 'node' | 'wasm' | 'webgpu'
  /** DeepFilterNet3: ort.InferenceSession options passed through, e.g. { intraOpNumThreads: 4 } */
  sessionOptions?: Record<string, unknown>
  /** DeepFilterNet3: replaces @audio/neural-runtime's load() for each graph (a custom ORT setup, a test double) */
  session?: (bytes: Uint8Array) => Promise<unknown>
  /** fetch progress, as @audio/neural-runtime reports it */
  progress?: (p: { loaded: number, total: number | null }) => void
}

export interface DenoiseOptions extends LoadOptions {
  /** Hz; required unless audio is { channelData, sampleRate } */
  sampleRate?: number
  /** 'rnnoise' (default), 'deepfilternet3', or a handle from load() to reuse across calls */
  model?: 'rnnoise' | 'deepfilternet3' | Model
  /** attenuation limit in dB, the most the noise drops: mixes the input back in at 10^(−limit/20); default 20 for RNNoise, 18 for DeepFilterNet3 (README, API); 0 for none. DeepFilterNet3 also hears its input with the speech at −20 dBFS and keeps sustained voicing it would remove (held sung notes) */
  limit?: number
  /** DeepFilterNet3: frames (10 ms) per model run, default 1000 */
  chunk?: number
  /** DeepFilterNet3: frames of context run before each chunk after the first, default 300 */
  warmup?: number
  /** DeepFilterNet3: frames over which consecutive chunks crossfade, default 50 */
  fade?: number
}

/** Denoise speech; returns the same shape as `audio`, same length, aligned with it. */
export default function denoise<T extends Audio>(audio: T, opts?: DenoiseOptions): Promise<T>

/** Load a model once to reuse it across denoise() calls. */
export function load(model?: 'rnnoise' | 'deepfilternet3', opts?: LoadOptions): Promise<Model>

/** The bundled RNNoise weights (3,544,320 bytes), e.g. for the worklet's processorOptions.weights. */
export function weights(): Promise<Uint8Array>

/** RNNoise frame by frame, as rnnoise_process_frame: 480 samples at 48 kHz and int16 scale in and out; returns the frame's voice probability. Output lags input by 960 samples. */
export function rnnoise(weights: Uint8Array | ArrayBuffer | object): { process(input: Float32Array, output: Float32Array): number, latency: number }

/** RNNoise frame length: 480 samples, 10 ms at 48 kHz. */
export const FRAME: 480

/** Upstream's DeepFilterNet3 ONNX export at Rikorose/DeepFilterNet d375b2d, fetched on first use and cached. */
export const MODEL: string
