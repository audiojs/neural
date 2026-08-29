// Sound matching (automatic synthesizer programming): given a black-box synth —
// a knob vector in, audio out — find knob settings that reproduce a target
// sound. Derivative-free: CMA-ES over normalized parameters against a
// multi-scale spectral loss, with seeded restarts (the practical recipe of the
// field — SpiegeLib/Sound2Synth lineage; a neural inverse model can later warm
// the seeds, same interface).

import specLoss from './loss.js'
import cmaes from './cmaes.js'

// match(render, target, opts) → { params, loss, evals }
//   render(params: number[]) → audio (ArrayLike) — the synth, params in real units
//   target: the sound to reproduce
//   opts.bounds: [[min, max], …] — one pair per knob (required)
//   opts.budget: total render evaluations, default 2000
//   opts.restarts: independent CMA-ES starts, default 3
//   opts.seeds: warm-start param vectors (real units), e.g. from a predictor
//   opts.loss: custom (a, b) → number, default multi-scale spectral
export default function match(render, target, opts = {}) {
  let bounds = opts.bounds
  if (!bounds?.length) throw new RangeError('match: opts.bounds required — one [min, max] per knob')
  let n = bounds.length
  let budget = opts.budget ?? 2000
  let restarts = opts.restarts ?? 3
  let lossFn = opts.loss ?? specLoss

  let toReal = u => bounds.map(([lo, hi], i) => lo + (hi - lo) * Math.min(1, Math.max(0, u[i])))
  let evals = 0
  let f = u => {
    // quadratic penalty outside the unit box keeps CMA-ES informed, render sees clipped
    let pen = 0
    for (let i = 0; i < n; i++) {
      let d = u[i] < 0 ? -u[i] : u[i] > 1 ? u[i] - 1 : 0
      pen += d * d
    }
    evals++
    return lossFn(target, render(toReal(u))) + 10 * pen
  }

  let starts = []
  if (opts.seeds) for (let s of opts.seeds)
    starts.push(Float64Array.from(s.map((v, i) => (v - bounds[i][0]) / (bounds[i][1] - bounds[i][0]))))
  // scatter cold starts over the unit box (deterministic low-discrepancy-ish)
  for (let si = 0; starts.length < restarts; si++)
    starts.push(Float64Array.from({ length: n }, (_, i) => ((si * 7 + i * 13 + 5) * 0.6180339887) % 1))

  // explore: restarts share 70% of the budget; polish: the winner gets the rest
  let best = null
  let explore = Math.floor(budget * 0.7 / starts.length)
  starts.forEach((start, si) => {
    let r = cmaes(f, start, opts.seeds && si < opts.seeds.length ? 0.1 : 0.3, { maxEvals: explore, seed: si + 1 })
    if (!best || r.fx < best.fx) best = r
  })
  let polish = cmaes(f, best.x, 0.15, { maxEvals: Math.floor(budget * 0.3), seed: 777 })
  if (polish.fx < best.fx) best = polish
  return { params: toReal(best.x), loss: best.fx, evals }
}

export { specLoss, cmaes }
