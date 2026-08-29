/** Sound matching — configure a knob-synth to reproduce a target sound. */

export interface MatchOptions {
  /** one [min, max] per knob, real units — required */
  bounds: [number, number][]
  /** total render evaluations across all stages, default 2000 */
  budget?: number
  /** independent CMA-ES starts, default 3 */
  restarts?: number
  /** warm-start param vectors (real units), e.g. from an inverse-model predictor */
  seeds?: ArrayLike<number>[]
  /** custom loss (target, candidate) → number; default multi-scale spectral */
  loss?: (a: ArrayLike<number>, b: ArrayLike<number>) => number
}

export interface MatchResult {
  /** best knob settings, real units */
  params: number[]
  /** loss at params */
  loss: number
  /** render evaluations spent */
  evals: number
}

/** Find knob settings for `render` that reproduce `target`. */
export default function match(
  render: (params: number[]) => ArrayLike<number>,
  target: ArrayLike<number>,
  opts: MatchOptions
): MatchResult

export interface SpecLossOptions {
  /** analysis resolutions, default [2048, 512, 128] */
  fftSizes?: number[]
  /** sample rate for the mel warp, default 44100 */
  fs?: number
  /** mel bands per scale, default 48 */
  melBands?: number
}

/** Multi-scale spectral distance: ≥ 0, 0 iff spectra match. */
export function specLoss(a: ArrayLike<number>, b: ArrayLike<number>, opts?: SpecLossOptions): number

export interface CmaesOptions {
  /** evaluation budget, default 100·n² */
  maxEvals?: number
  /** RNG seed (deterministic), default 1 */
  seed?: number
  /** population size, default 4+⌊3·ln n⌋ */
  lambda?: number
  /** stop when σ·max(D) falls below this, default 1e-12 */
  tolX?: number
  /** stop when the best f falls to this, default -Infinity */
  tolFun?: number
}

/** CMA-ES minimizer (weighted recombination, rank-1 + rank-μ, CSA). */
export function cmaes(
  f: (x: Float64Array) => number,
  x0: ArrayLike<number>,
  sigma0: number,
  opts?: CmaesOptions
): { x: Float64Array; fx: number; evals: number }
