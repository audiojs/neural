/** Neural speech enhancement: RNNoise (pure JS, weights bundled) and DeepFilterNet3 (ONNX, fetched and cached); music passes through untouched. */

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
  /** attenuation limit in dB, the most the noise drops: mixes the input back in at 10^(−limit/20) (DeepFilterNet3: less where the noise stands within `floor` of the voice); default 16 for RNNoise, 18 for DeepFilterNet3 (README, API); 0 takes nothing off (DeepFilterNet3: but what `floor` takes in the voice's pauses), `Infinity` is no limit; a value that is no number is a TypeError. DeepFilterNet3 also hears its input with the speech at −20 dBFS and, above a band-limited input's edge, a white noise floor 20 dB under it, and keeps sustained voicing it would remove (held sung notes) */
  limit?: number
  /** DeepFilterNet3: dB under the voice to which noise the limit would leave louder drops (mixback()), frame by frame, in the voice's pauses: where the model's output stays more than 30 dB under the voice, which on real noise it mostly does not, and the noise there (what the model took, over 0.5 s) stands less than 6 dB over the bed about it, the quietest tenth of it within 10 s either side (more: a voice it took whole, or a noise that comes and goes), so there the limit holds (README, DeepFilterNet3's floor); under the voice the limit stands. Default 40, 0 for none (before 0.5: every noise down by the limit only) */
  floor?: number
  /** DeepFilterNet3: true takes the floor under the voice too, what the model took back at one gain: 0.7.0's mix, sample for sample; the cleanest pauses, and the voice the model took goes with the noise. Default false */
  everywhere?: boolean
  /** 'pass' (default): speech and noise are enhanced, music (songs included) passes through untouched, as inaSpeechSegmenter's speech/music/noise CNN segments the input: DeepFilterNet3 with the whole input in view, RNNoise as its stream decides (from what has arrived: music's first second or so is denoised); the gain ramps over 200 ms at each switch. 'enhance': everything, as before 0.4 */
  music?: 'pass' | 'enhance'
  /** DeepFilterNet3: frames (10 ms) per model run, default 1000 */
  chunk?: number
  /** DeepFilterNet3: frames of context run before each chunk after the first, default 300 */
  warmup?: number
  /** DeepFilterNet3: frames over which consecutive chunks crossfade, default 50 */
  fade?: number
}

/** Denoise speech; returns the same shape as `audio`, same length, aligned with it. */
export default function denoise<T extends Audio>(audio: T, opts?: DenoiseOptions): Promise<T>

/** The input x mixed back into a model's output y, y + a·(x − y): the noise drops by `limit` dB (default 18), or to `floor` dB under the voice (default 40) where it stands higher in the voice's pauses (where y stays more than 30 dB under the voice and x − y, over 0.5 s, less than 6 dB over its quietest tenth within 10 s either side), per 10 ms frame; `everywhere`: the floor under the voice too (0.7.0's mix); `limit` Infinity: y; `limit` 0: nothing taken but what the floor takes in the voice's pauses; `floor` 0: a constant 10^(−limit/20) (with `limit` 0, x). denoise() applies it to DeepFilterNet3. */
export function mixback(x: Float32Array, y: Float32Array, opts?: { limit?: number, floor?: number, everywhere?: boolean, sampleRate?: number }): Float32Array

/** Per sample of x, the gain an enhancement of it takes: 1 where it is speech or noise, 0 where it is music (inaSpeechSegmenter's segments over the whole input, as denoise() decides), ramped over 200 ms; blend g·y + (1 − g)·x. */
export function guard(x: Float32Array, sampleRate?: number): Promise<Float32Array>

/** What x is, stretch by stretch, as inaSpeechSegmenter's speech/music/noise CNN segments the whole input (no prior for speech, which is the guard's): in order, end to end from 0 to its length, seconds; none for an input of digital silence. */
export function segments(x: Float32Array, sampleRate?: number): Promise<{ kind: 'speech' | 'music' | 'noise', start: number, end: number }[]>

/** The kinds segments() names, as the classifier orders them. */
export const KINDS: ['speech', 'music', 'noise']

/** Load a model once to reuse it across denoise() calls. */
export function load(model?: 'rnnoise' | 'deepfilternet3', opts?: LoadOptions): Promise<Model>

/** The bundled RNNoise weights (3,544,320 bytes), e.g. for the worklet's processorOptions.weights. */
export function weights(): Promise<Uint8Array>

/** RNNoise frame by frame, as rnnoise_process_frame: 480 samples at 48 kHz and int16 scale in and out; returns the frame's voice probability. Output lags input by 960 samples. `run` is `process` with no number returned, the probability left in `voice[0]`: real-time loops call it, since a float a call returns is a new object where V8 does not inline the call. */
export function rnnoise(weights: Uint8Array | ArrayBuffer | object): { process(input: Float32Array, output: Float32Array): number, run(input: Float32Array, output: Float32Array): void, voice: Float32Array, latency: number }

/** RNNoise frame length: 480 samples, 10 ms at 48 kHz. */
export const FRAME: 480

/** Upstream's DeepFilterNet3 ONNX export at Rikorose/DeepFilterNet d375b2d, fetched on first use and cached. */
export const MODEL: string
