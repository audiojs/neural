/** Standardized capture excitation — the dry signal to play through a device. */

export interface ProbeOptions {
  /** sample rate, default 48000 */
  fs?: number
  /** ESS sweep length in seconds, default 4 */
  sweepDuration?: number
  /** per-level noise segment length in seconds, default 2 */
  noiseDuration?: number
  /** amplitude-ramp triangle length in seconds, default 3 */
  rampDuration?: number
  /** noise segment peak levels, default [0.06, 0.25, 0.5, 0.95] */
  levels?: number[]
}

export interface ProbeSegment {
  type: 'silence' | 'click' | 'sweep' | 'noise' | 'ramp'
  /** offset in samples */
  start: number
  /** length in samples */
  duration: number
  /** peak level (noise/ramp segments) */
  level?: number
  /** sweep band / ramp tone frequency */
  f0?: number
  f1?: number
  f?: number
}

export interface Probe {
  signal: Float32Array
  fs: number
  layout: ProbeSegment[]
}

/** Deterministic: the same options always produce the identical signal. */
export default function probe(opts?: ProbeOptions): Probe
