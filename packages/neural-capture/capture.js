// Staged device identifier — dry/wet pair → cheapest formula that nulls.
//   rung 1 'lti':         Wiener deconvolution → impulse response
//   rung 2 'hammerstein': static nonlinearity (Chebyshev curve) → IR, iterated LS
// Escalates only when the null test fails; reports ESR + null depth per attempt.
// Output stages speak the audiojs chain-recipe vocabulary ({ atom, name, params, why })
// so the capture result is an inspectable, dependency-free pipeline of shipped atoms
// (@audio/reverb-convolution, @audio/saturate-waveshaper) — not a weights blob.
//
// Later rungs (differentiable Wiener–Hammerstein, causal TCN) land behind the same
// interface; see research.md.

import { ifft } from 'fourier-transform'
import { nextPow2, fftCopy, convolve, chebyEval, chebyBasis, lsFit } from './math.js'
import fitWh from './wh.js'

// --- spectral helpers -------------------------------------------------------

// Wiener deconvolution: H = Y·X* / (|X|² + reg·max|X|²) → rough causal IR.
// Biased by the record-edge truncated tail — used only to seed/size the LS solve.
function wienerIr(x, y, reg) {
  let N = nextPow2(2 * Math.max(x.length, y.length))
  let [xr, xi] = fftCopy(x, N)
  let [yr, yi] = fftCopy(y, N)
  let pmax = 0
  for (let k = 0; k < xr.length; k++) pmax = Math.max(pmax, xr[k] * xr[k] + xi[k] * xi[k])
  let eps = reg * pmax + 1e-300
  for (let k = 0; k < xr.length; k++) {
    let p = xr[k] * xr[k] + xi[k] * xi[k] + eps
    let re = (yr[k] * xr[k] + yi[k] * xi[k]) / p
    let im = (yi[k] * xr[k] - yr[k] * xi[k]) / p
    xr[k] = re
    xi[k] = im
  }
  return ifft(xr, xi, new Float64Array(N)) // full: causal in [0,N/2), negative lags wrap to the top
}

// Exact FIR least squares: min ‖y − x∗h‖² + λ‖h‖² over L taps, solved by
// conjugate gradient with FFT Toeplitz matvecs. Unbiased for truncated records
// (rows [0, n) of the convolution are complete); noise excitation makes the
// normal matrix ≈ σ²·I, so CG converges in a handful of iterations.
function cgIr(x, y, h0, L, { reg = 1e-8, cgIters = 50, cgTol = 1e-8 } = {}) {
  let n = y.length
  let N = nextPow2(n + L)
  let [xr, xi] = fftCopy(x, N)
  let lambda = 0
  for (let i = 0; i < x.length; i++) lambda += x[i] * x[i]
  lambda *= reg

  // X v: causal convolution truncated to n rows
  let fwd = v => {
    let [vr, vi] = fftCopy(v, N)
    for (let k = 0; k < vr.length; k++) {
      let re = vr[k] * xr[k] - vi[k] * xi[k]
      vi[k] = vr[k] * xi[k] + vi[k] * xr[k]
      vr[k] = re
    }
    return ifft(vr, vi, new Float64Array(N)).subarray(0, n)
  }
  // Xᵀu: cross-correlation lags [0, L)
  let corr = u => {
    let [ur, ui] = fftCopy(u, N)
    for (let k = 0; k < ur.length; k++) {
      let re = ur[k] * xr[k] + ui[k] * xi[k]
      ui[k] = ui[k] * xr[k] - ur[k] * xi[k]
      ur[k] = re
    }
    return ifft(ur, ui, new Float64Array(N)).subarray(0, L)
  }
  let matvec = v => {
    let c = corr(fwd(v))
    for (let j = 0; j < L; j++) c[j] += lambda * v[j]
    return c
  }
  let dot = (a, b) => {
    let s = 0
    for (let i = 0; i < a.length; i++) s += a[i] * b[i]
    return s
  }

  let h = Float64Array.from(h0.subarray(0, L))
  let b = corr(y)
  let r = matvec(h), p
  for (let j = 0; j < L; j++) r[j] = b[j] - r[j]
  p = r.slice()
  let rs = dot(r, r), b2 = dot(b, b) + 1e-300
  for (let it = 0; it < cgIters && rs > cgTol * cgTol * b2; it++) {
    let Ap = matvec(p)
    let alpha = rs / (dot(p, Ap) + 1e-300)
    for (let j = 0; j < L; j++) {
      h[j] += alpha * p[j]
      r[j] -= alpha * Ap[j]
    }
    let rs2 = dot(r, r)
    let beta = rs2 / rs
    rs = rs2
    for (let j = 0; j < L; j++) p[j] = r[j] + beta * p[j]
  }
  return h
}

// Rough Wiener seed → memory-depth detection → exact LS polish → tail trim.
function estimateIr(x, y, opts = {}) {
  let reg = opts.reg ?? 1e-8
  let h0 = wienerIr(x, y, Math.max(reg, 1e-6))
  h0 = h0.subarray(0, h0.length >> 1) // causal half — anti-wrap
  // L capped at n/4 — L → n makes the convolution matrix square/invertible and
  // the "IR" overfits whatever the linear model can't explain
  let L, cap = Math.min(h0.length, y.length >> 2, opts.maxIrLength ?? 1 << 16)
  if (opts.irLength) L = Math.min(opts.irLength, cap)
  else {
    // memory depth: block-RMS envelope vs its own median (= estimation noise
    // floor) — keeps decaying tails and late echoes, rejects flat seed noise
    let W = 32, nb = Math.ceil(cap / W)
    let env = new Float64Array(nb)
    for (let b = 0; b < nb; b++) {
      let s = 0, i0 = b * W, i1 = Math.min(i0 + W, cap)
      for (let i = i0; i < i1; i++) s += h0[i] * h0[i]
      env[b] = Math.sqrt(s / (i1 - i0))
    }
    let floor = Float64Array.from(env).sort()[nb >> 1]
    let lastB = 0
    for (let b = 0; b < nb; b++) if (env[b] > 3 * floor) lastB = b
    L = Math.max(16, Math.min((lastB + 1) * W, cap))
  }
  let h = cgIr(x, y, h0, L, opts)
  if (!opts.irLength) {
    // the seed's bias floor hides tail below ~−30 dB — extend while it pays:
    // double L as long as the polished null deepens by > 3 dB
    let e = esr(y, convolve(x, h, y.length))
    while (L < cap) {
      let L2 = Math.min(2 * L, cap)
      let h2 = cgIr(x, y, h0, L2, opts)
      let e2 = esr(y, convolve(x, h2, y.length))
      if (!(e2 < e * 0.5)) break
      h = h2
      e = e2
      L = L2
    }
  }
  if (opts.irLength) return h
  let peak = 0
  for (let i = 0; i < L; i++) peak = Math.max(peak, Math.abs(h[i]))
  let floor = peak * 10 ** ((opts.floorDb ?? -80) / 20)
  let last = 15
  for (let i = 0; i < L; i++) if (Math.abs(h[i]) > floor) last = i
  return h.subarray(0, last + 1)
}

// Device lag = onset of the (linearized) impulse response. The Wiener IR is the
// right correlogram for this: the spectral division whitens the excitation
// (sweeps and tones smear plain cross-correlation) while regularization keeps
// magnitude weighting, so a causal device yields a causal h — unlike GCC-PHAT,
// whose phase-only correlogram scatters on dense IRs. Negative lags (wet leads:
// misaligned recording) wrap to the top half. Positive lag = wet delayed;
// trimming it makes the formula zero-added-latency (the honest ceiling), and
// the measured lag is reported as result.latency.
function measureLag(dry, wet) {
  let n = Math.min(dry.length, wet.length)
  let h = wienerIr(dry, wet, 1e-6)
  let N = h.length
  let maxLag = Math.min(n - 1, N >> 1)
  let peak = 0
  for (let i = 0; i < N; i++) peak = Math.max(peak, Math.abs(h[i]))
  let thr = 0.3 * peak
  for (let m = -maxLag; m < maxLag; m++) {
    if (Math.abs(h[(m + N) % N]) < thr) continue
    // a gradually-rising IR (lowpass) crosses the threshold a few samples into
    // its support — walk back over the contiguous leading slope
    while (m > -maxLag && Math.abs(h[(m - 1 + N) % N]) > 0.05 * peak) m--
    return m
  }
  return 0
}

function alignPair(dry, wet) {
  let lag = measureLag(dry, wet)
  if (lag > 0) return { dry: dry.subarray(0, dry.length - lag), wet: wet.subarray(lag), lag }
  if (lag < 0) return { dry: dry.subarray(-lag), wet: wet.subarray(0, wet.length + lag), lag }
  return { dry, wet, lag }
}

// --- metrics ----------------------------------------------------------------

// Error-to-signal ratio: ∑(wet−pred)² / ∑wet². Null depth = 10·log10(esr) dB.
export function esr(wet, pred) {
  let n = Math.min(wet.length, pred.length)
  let e = 0, s = 0
  for (let i = 0; i < n; i++) {
    let d = wet[i] - pred[i]
    e += d * d
    s += wet[i] * wet[i]
  }
  return s > 0 ? e / s : e > 0 ? Infinity : 0
}

export const nullDepth = (wet, pred) => 10 * Math.log10(Math.max(esr(wet, pred), 1e-30))

// --- rungs ------------------------------------------------------------------

function fitLti(dry, wet, opts) {
  let ir = estimateIr(dry, wet, opts)
  let pred = convolve(dry, ir, wet.length)
  return { rung: 'lti', ir, pred }
}

function fitHammerstein(dry, wet, opts) {
  let order = opts.order ?? 11
  let iterations = opts.iterations ?? 3
  let scale = 0
  for (let i = 0; i < dry.length; i++) scale = Math.max(scale, Math.abs(dry[i]))
  if (!scale) scale = 1

  let basis = chebyBasis(dry, scale, order)
  let coeffs = new Array(order).fill(0)
  coeffs[0] = 1 // identity init: f(x) = T₁(x/s)·1 → x/s (gain absorbed by LS/IR)
  let u = Float64Array.from(dry), ir, pred

  for (let it = 0; it < iterations; it++) {
    ir = estimateIr(u, wet, opts)
    // filtered basis zₖ = ir * Tₖ; least squares wet ≈ Σ cₖ·zₖ
    let z = basis.map(bk => convolve(bk, ir, wet.length))
    coeffs = lsFit(z, wet)
    for (let i = 0; i < dry.length; i++) {
      let s = 0
      for (let k = 0; k < order; k++) s += coeffs[k] * basis[k][i]
      u[i] = s
    }
    pred = convolve(u, ir, wet.length)
  }
  return { rung: 'hammerstein', ir, curve: { type: 'chebyshev', coeffs, scale }, pred }
}

// --- staged capture ---------------------------------------------------------

// Long-tail factorization: alternate a short-memory wh fit against a fixed
// linear tail with an exact LS tail solve — the amp shape (drive → cab/room).
function fitWhTail(dry, wet, opts, prev) {
  let init = prev?.pre ? { pre: prev.pre, post: prev.post } : undefined
  let tail = null, fit
  for (let round = 0; round < (opts.tailRounds ?? 2); round++) {
    fit = fitWh(dry, wet, { ...opts, whIters: opts.whIters ?? 800, tail, init })
    tail = Float64Array.from(estimateIr(fit.preTail, wet, opts))
    init = { pre: fit.pre, post: fit.post }
  }
  let pred = convolve(fit.preTail, tail, wet.length)
  return { rung: 'wh-tail', pre: fit.pre, curve: fit.curve, post: fit.post, ir: tail, pred }
}

const RUNGS = {
  lti: fitLti,
  hammerstein: fitHammerstein,
  wh: (dry, wet, opts, prev) =>
    fitWh(dry, wet, prev?.curve ? { ...opts, init: { curve: prev.curve, ir: prev.ir } } : opts),
  'wh-tail': fitWhTail,
}
const LADDER = ['lti', 'hammerstein', 'wh', 'wh-tail']

// capture(dry, wet, opts) → { rung, nullDb, esr, ir, curve?, stages, attempts }
// Tries the cheapest rung, escalates while nullDb > targetDb, keeps the deepest null.
export default function capture(dry, wet, opts = {}) {
  if (!dry?.length || !wet?.length) throw new RangeError('capture: dry/wet pair required')
  if (!dry.subarray) dry = Float64Array.from(dry)
  if (!wet.subarray) wet = Float64Array.from(wet)
  let latency = 0
  if (opts.align !== false) ({ dry, wet, lag: latency } = alignPair(dry, wet))
  let targetDb = opts.targetDb ?? -60
  let ladder = LADDER.slice(0, 1 + LADDER.indexOf(opts.maxRung ?? LADDER[LADDER.length - 1]))

  let attempts = [], best = null
  let att = r => attempts.find(a => a.rung === r)
  for (let rung of ladder) {
    if (rung === 'wh' || rung === 'wh-tail') {
      // a deep linear null means nonlinearity < 1% of signal — the exact IR
      // already is the formula; biquad fitting can only lose
      if (best && best.nullDb <= (opts.whSkipLinearDb ?? -40)) {
        attempts.push({ rung, skipped: `device essentially linear (${best.nullDb.toFixed(1)} dB by ${best.rung})` })
        continue
      }
    }
    if (rung === 'wh' && best?.ir) {
      // biquads can't express long memory — that's wh-tail's job
      let total = 0
      for (let v of best.ir) total += v * v
      let acc = 0, depth = 0
      for (let i = 0; i < best.ir.length; i++) {
        acc += best.ir[i] * best.ir[i]
        if (acc >= 0.99 * total) { depth = i + 1; break }
      }
      if (depth > (opts.whMaxMemory ?? 2048)) {
        attempts.push({ rung, skipped: `memory depth ${depth} > ${opts.whMaxMemory ?? 2048}` })
        continue
      }
    }
    let fit = RUNGS[rung](dry, wet, opts, best)
    let e = esr(wet, fit.pred)
    let nullDb = 10 * Math.log10(Math.max(e, 1e-30))
    attempts.push({ rung, nullDb, esr: e })
    // a costlier rung must be materially better (> 1 dB) to displace a cheaper formula
    if (!best || nullDb < best.nullDb - 1) best = { ...fit, nullDb, esr: e }
    if (nullDb <= targetDb) break
  }

  let { rung, nullDb, esr: e, ir, curve, pre, post } = best
  let stages = []
  let cascadeStage = (secs, role) => ({
    atom: '@audio/biquad',
    name: 'cascade',
    params: { coefs: secs.map(c => ({ ...c })) },
    why: `${role} — ${secs.length} biquad section${secs.length > 1 ? 's' : ''}`,
  })
  if (pre) stages.push(cascadeStage(pre, 'pre filter (input tone shaping)'))
  if (curve) stages.push({
    atom: '@audio/saturate-waveshaper',
    name: 'waveshaper',
    params: { curve: { type: 'chebyshev', coeffs: [...curve.coeffs], scale: curve.scale }, oversample: 1 },
    why: `static nonlinearity — Chebyshev order ${curve.coeffs.length}, fitted on |x| ≤ ${curve.scale.toFixed(3)}`,
  })
  if (post) stages.push(cascadeStage(post, 'post filter (output tone shaping)'))
  if (ir) stages.push({
    atom: '@audio/reverb-convolution',
    name: 'convolve',
    params: { ir: [...ir] },
    why: `linear memory — ${ir.length}-tap IR, null ${nullDb.toFixed(1)} dB`,
  })
  return {
    rung, nullDb, esr: e, latency,
    ir: ir && Float32Array.from(ir), curve, pre, post,
    stages, attempts,
  }
}

// render(result, dry) → prediction of the wet signal (identification-domain: no
// oversampling — matches how the formula was fitted and how the null was measured).
// Re-applies result.latency so the output sits on the original wet timeline.
export function render(result, dry) {
  let lat = result.latency ?? 0
  let x = new Float64Array(dry.length)
  if (lat >= 0) for (let i = 0; i + lat < dry.length; i++) x[i + lat] = dry[i]
  else for (let i = 0; i < dry.length + lat; i++) x[i] = dry[i - lat]
  for (let s of result.stages) {
    if (s.name === 'waveshaper') {
      let { coeffs, scale } = s.params.curve
      for (let i = 0; i < x.length; i++) x[i] = chebyEval(coeffs, scale, x[i])
    } else if (s.name === 'convolve') {
      x = convolve(x, s.params.ir, x.length)
    } else if (s.name === 'cascade') {
      for (let { b0, b1, b2, a1, a2 } of s.params.coefs) {
        let x1 = 0, x2 = 0, y1 = 0, y2 = 0
        for (let i = 0; i < x.length; i++) {
          let y = b0 * x[i] + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
          x2 = x1; x1 = x[i]; y2 = y1; y1 = y
          x[i] = y
        }
      }
    } else throw new Error(`render: unknown stage ${s.name}`)
  }
  return Float32Array.from(x)
}
