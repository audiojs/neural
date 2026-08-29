/** Staged device identifier — dry/wet pair → cheapest audiojs formula that nulls. */

export type Rung = 'lti' | 'hammerstein' | 'wh' | 'wh-tail'

/** Biquad section, a0-normalized (`@audio/biquad` convention). */
export interface BiquadCoefs {
  b0: number
  b1: number
  b2: number
  a1: number
  a2: number
}

export interface CaptureOptions {
  /** stop escalating once the null reaches this depth (dB), default -60 */
  targetDb?: number
  /** highest rung to try, default 'wh-tail' */
  maxRung?: Rung
  /** wh-tail: alternation rounds (wh fit ↔ LS tail solve), default 2 */
  tailRounds?: number
  /** GCC-PHAT latency detection + trim, default true; the lag lands in result.latency */
  align?: boolean
  /** fixed IR length in samples; default auto-detects memory depth */
  irLength?: number
  /** auto-detection IR length cap, default 65536 (also capped at signal/4) */
  maxIrLength?: number
  /** trim the polished IR tail this far below its peak (dB), default -80 */
  floorDb?: number
  /** Tikhonov regularization relative to input energy, default 1e-8 */
  reg?: number
  /** conjugate-gradient iteration cap, default 50 */
  cgIters?: number
  /** conjugate-gradient relative residual tolerance, default 1e-8 */
  cgTol?: number
  /** Chebyshev order of the static-nonlinearity curve, default 11 */
  order?: number
  /** Hammerstein alternating iterations, default 3 */
  iterations?: number
  /** wh: biquad sections per filter (pre and post each), default 2 */
  whSections?: number
  /** wh: Adam iteration cap, default 1500 */
  whIters?: number
  /** wh: Adam initial learning rate (cosine-decayed), default 0.1 */
  whLr?: number
  /** wh: skip the rung when the identified IR's 99%-energy depth exceeds this, default 2048 */
  whMaxMemory?: number
  /** wh: skip the rung when a cheaper rung already nulls this deep (device essentially linear), default -40 */
  whSkipLinearDb?: number
  /** wh: per-iteration callback (iteration, loss as ESR) */
  onProgress?: (iteration: number, loss: number) => void
  /** wh: warm-start filters/curve (e.g. a previous capture's pre/post/curve) */
  init?: { pre?: BiquadCoefs[]; post?: BiquadCoefs[]; curve?: ChebyshevCurve; ir?: ArrayLike<number> }
}

export interface ChebyshevCurve {
  type: 'chebyshev'
  /** coefficients of T₁..T_order applied to x/scale */
  coeffs: number[]
  /** input normalization — the peak |dry| the curve was fitted on */
  scale: number
}

/** Chain-recipe stage (audiojs `@audio/chain` vocabulary) — JSON-safe. */
export interface Stage {
  atom: string
  name: string
  params: Record<string, unknown>
  why: string
}

export interface Attempt {
  rung: Rung
  nullDb?: number
  esr?: number
  /** present when the rung was skipped, with the reason */
  skipped?: string
}

export interface CaptureResult {
  /** cheapest rung that reached the target (or the deepest-nulling attempt) */
  rung: Rung
  /** null depth 10·log10(esr), dB — lower is better */
  nullDb: number
  /** error-to-signal ratio ∑(wet−pred)²/∑wet² */
  esr: number
  /** device lag trimmed from the pair (samples; negative = wet led dry) */
  latency: number
  /** identified impulse response of the linear stage (lti/hammerstein rungs) */
  ir?: Float32Array
  /** static nonlinearity, present from the hammerstein rung up */
  curve?: ChebyshevCurve
  /** wh rung: input/output biquad cascades */
  pre?: BiquadCoefs[]
  post?: BiquadCoefs[]
  /** runnable chain recipe — cascade/waveshaper/convolve stages of shipped atoms */
  stages: Stage[]
  /** every rung tried, in order */
  attempts: Attempt[]
}

/** Identify the device behind a dry/wet pair, escalating rungs until the null test passes. */
export default function capture(
  dry: ArrayLike<number>,
  wet: ArrayLike<number>,
  opts?: CaptureOptions
): CaptureResult

/** Run a capture result over a dry signal (identification-domain: no oversampling). */
export function render(result: Pick<CaptureResult, 'stages'>, dry: ArrayLike<number>): Float32Array

/** Error-to-signal ratio ∑(wet−pred)²/∑wet². */
export function esr(wet: ArrayLike<number>, pred: ArrayLike<number>): number

/** Null depth 10·log10(esr), dB. */
export function nullDepth(wet: ArrayLike<number>, pred: ArrayLike<number>): number
