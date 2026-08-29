/** Differentiable Wiener–Hammerstein rung — usually reached via capture(); direct use for custom fits. */

import type { BiquadCoefs, CaptureOptions, ChebyshevCurve } from './index.js'

export interface WhFit {
  rung: 'wh'
  pre: BiquadCoefs[]
  curve: ChebyshevCurve
  post: BiquadCoefs[]
  /** model output over the dry signal (identification-domain) */
  pred: Float64Array
}

export default function fitWh(
  dry: ArrayLike<number>,
  wet: ArrayLike<number>,
  opts?: CaptureOptions
): WhFit
