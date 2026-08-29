/** Forced alignment — audio + known transcript → word/character timestamps, via a wav2vec2-class CTC acoustic model. */

export { ctcAlign, mergeWords } from './ctc.js'

export interface Word {
  text: string
  /** seconds */
  start: number
  /** seconds */
  end: number
  /** confidence in [0,1] — geometric mean of the word's per-frame probabilities */
  score: number
}

export interface Char {
  text: string
  /** seconds */
  start: number
  /** seconds */
  end: number
  score: number
}

export interface AlignResult {
  words: Word[]
  chars: Char[]
  /** mean of words[].score */
  score: number
  /** present only when opts.cues is truthy — needs @audio/subtitle installed */
  cues?: unknown[]
  /** present only when opts.lrc is truthy — enhanced LRC text, needs @audio/subtitle installed */
  lrc?: string
}

export interface AlignOptions {
  /** Hz of the input audio — required */
  sampleRate: number
  /** HF hub id of a wav2vec2-class CTC model, default 'Xenova/wav2vec2-base-960h' */
  model?: string
  /** documentation/future-proofing only — this build has no per-language adapter switching, see README */
  language?: string
  /** uppercase + strip characters the vocab has no id for, default true */
  normalize?: boolean
  /** seconds per inference window for long audio (1 s overlap, stitched), default 20 */
  chunk?: number
  /** onnxruntime dtype override, e.g. 'fp32' | 'q8' */
  dtype?: string
  /** onnxruntime device override, e.g. 'wasm' | 'webgpu' | 'cpu' */
  device?: string
  /** Node only: override the model cache directory for this call (default: $AUDIO_NEURAL_CACHE/hf or ~/.cache/audiojs/neural/hf — see README Cache section) */
  cache?: string
  /** attach @audio/subtitle Cue[] to the result */
  cues?: boolean
  /** attach an enhanced-LRC string to the result (implies cues) */
  lrc?: boolean
  /** @huggingface/transformers model-loading progress callback */
  progress?: (info: unknown) => void
}

/** audio + known transcript → word/character timestamps. Loads (and caches) the model on first call. */
export default function align(
  audio: Float32Array | Float64Array | ArrayLike<number>[],
  text: string,
  opts: AlignOptions
): Promise<AlignResult>

export interface Logits {
  /** T×V row-major, log-softmax */
  logProbs: Float32Array
  T: number
  V: number
  /** seconds per frame — 0.02 for wav2vec2-class models */
  frameDuration: number
  /** id → character, index === token id */
  vocab: string[]
}

export interface ModelHandle {
  align(audio: Float32Array | Float64Array | ArrayLike<number>[], text: string, opts?: Omit<AlignOptions, 'sampleRate' | 'model'> & { sampleRate: number }): Promise<AlignResult>
  logits(audio: Float32Array | Float64Array | ArrayLike<number>[], opts: { sampleRate: number }): Promise<Logits>
  /** releases the underlying onnxruntime session and drops the module-level cache entry */
  free(): Promise<void>
}

/** Load (and cache by model+dtype+device+cache) a CTC acoustic model. */
export function loadModel(model?: string, opts?: Pick<AlignOptions, 'dtype' | 'device' | 'cache' | 'progress'>): Promise<ModelHandle>
