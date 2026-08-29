// Differentiable Wiener–Hammerstein rung: biquad cascade → static Chebyshev
// curve → biquad cascade, fit by Adam over exact analytic gradients. The
// compact-formula lane (Kuznetsov/Parker/Esqueda, DAFx-20): a real device's
// drive-dependent tone lands in dozens of coefficients, all shipped atoms.
//
// Gradients, per biquad section y = (B/A)x (direct form, a0 = 1):
//   ∂L/∂b_j = ⟨e, z⁻ʲ(1/A)x⟩     ∂L/∂a_j = −⟨e, z⁻ʲ(1/A)y⟩
//   adjoint into the input: e ← reverse(biquad(reverse(e)))  (Aᵀ = time-reversal)
// Poles parameterized r = 0.999·σ(ρ), angle θ: a1 = −2r·cosθ, a2 = r² — stable
// by construction, no projection step.

// --- primitives -------------------------------------------------------------

function biquadFwd(x, c, out = new Float64Array(x.length)) {
  let { b0, b1, b2, a1, a2 } = c
  let x1 = 0, x2 = 0, y1 = 0, y2 = 0
  for (let n = 0; n < x.length; n++) {
    let y = b0 * x[n] + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
    x2 = x1; x1 = x[n]; y2 = y1; y1 = y
    out[n] = y
  }
  return out
}

// 1/A only — the sensitivity filter
function polesFwd(x, c, out = new Float64Array(x.length)) {
  let { a1, a2 } = c
  let y1 = 0, y2 = 0
  for (let n = 0; n < x.length; n++) {
    let y = x[n] - a1 * y1 - a2 * y2
    y2 = y1; y1 = y
    out[n] = y
  }
  return out
}

function biquadRev(e, c) {
  let r = Float64Array.from(e).reverse()
  biquadFwd(r, c, r)
  return r.reverse()
}

const sigmoid = z => 1 / (1 + Math.exp(-z))

import { chebyBasis, lsFit, convolve } from './math.js'

// Adjoint of convolution with h: correlate — reverse, convolve, reverse.
function correlate(e, h) {
  let r = Float64Array.from(e).reverse()
  return Float64Array.from(convolve(r, h, r.length)).reverse()
}

function coefs(sec) {
  let r = 0.999 * sigmoid(sec.rho)
  return { b0: sec.b[0], b1: sec.b[1], b2: sec.b[2], a1: -2 * r * Math.cos(sec.theta), a2: r * r }
}

// --- model ------------------------------------------------------------------

function forward(model, x, keep) {
  let sig = x
  let acts = []
  for (let sec of model.pre) {
    if (keep) acts.push(sig)
    sig = biquadFwd(sig, coefs(sec))
  }
  if (keep) acts.push(sig)
  let t = new Float64Array(sig.length)
  let u = new Float64Array(sig.length)
  let P = model.curve.length, s = model.scale
  for (let n = 0; n < sig.length; n++) {
    let tt = sig[n] / s, tp = 1, tc = tt, y = 0
    for (let k = 0; k < P; k++) {
      y += model.curve[k] * tc
      let tn = 2 * tt * tc - tp
      tp = tc; tc = tn
    }
    t[n] = tt
    u[n] = y
  }
  sig = u
  for (let sec of model.post) {
    if (keep) acts.push(sig)
    sig = biquadFwd(sig, coefs(sec))
  }
  let preTail = sig
  if (model.tail) sig = convolve(sig, model.tail, sig.length)
  return keep ? { pred: sig, preTail, acts, t } : { pred: sig, preTail }
}

function sectionGrads(sec, x, y, e, g) {
  let c = coefs(sec)
  let w = polesFwd(x, c)
  let v = polesFwd(y, c)
  let gb0 = 0, gb1 = 0, gb2 = 0, ga1 = 0, ga2 = 0
  for (let n = 0; n < e.length; n++) {
    gb0 += e[n] * w[n]
    if (n > 0) { gb1 += e[n] * w[n - 1]; ga1 -= e[n] * v[n - 1] }
    if (n > 1) { gb2 += e[n] * w[n - 2]; ga2 -= e[n] * v[n - 2] }
  }
  let sg = sigmoid(sec.rho), r = 0.999 * sg, drdRho = 0.999 * sg * (1 - sg)
  g.b[0] += gb0; g.b[1] += gb1; g.b[2] += gb2
  g.rho += (ga1 * -2 * Math.cos(sec.theta) + ga2 * 2 * r) * drdRho
  g.theta += ga1 * 2 * r * Math.sin(sec.theta)
  return biquadRev(e, c)
}

// --- fit --------------------------------------------------------------------

function initSections(K, spread) {
  return Array.from({ length: K }, (_, i) => {
    let sec = { rho: 0, theta: Math.PI * (i + 1) / (K + 1) * spread }
    let r = 0.999 * sigmoid(sec.rho)
    // zeros = poles → B = A → exact identity at init
    sec.b = [1, -2 * r * Math.cos(sec.theta), r * r]
    return sec
  })
}

// {b0,b1,b2,a1,a2} → optimizer parameterization (b, rho, theta)
function sectionFromCoefs(c) {
  let r = Math.min(Math.sqrt(Math.max(c.a2, 1e-12)), 0.998)
  let cos = Math.max(-1, Math.min(1, -c.a1 / (2 * r)))
  let s = r / 0.999
  return { b: [c.b0, c.b1, c.b2], rho: Math.log(s / (1 - s)), theta: Math.acos(cos) }
}

function paramList(model) {
  let params = []
  for (let sec of [...model.pre, ...model.post]) params.push([sec, 'b', 3], [sec, 'rho', 1], [sec, 'theta', 1])
  params.push([model, 'curve', model.curve.length])
  return params
}

// Given the filters, the model is linear in curve coefficients:
// pred = Σ cₖ·post(Tₖ(pre(x)/s)) — solve them exactly. Keeping the curve at its
// LS optimum every iteration means filter gradients see a clean objective
// (envelope theorem) instead of fighting curve error.
function curveLS(model, x, wet) {
  let v = x
  for (let sec of model.pre) v = biquadFwd(v, coefs(sec))
  let basis = chebyBasis(v, model.scale, model.curve.length)
  let z = basis.map(bk => {
    let u = bk
    for (let sec of model.post) u = biquadFwd(u, coefs(sec))
    if (model.tail) u = convolve(u, model.tail, wet.length)
    return u
  })
  model.curve.set(lsFit(z, wet))
}

// ESR loss + full analytic gradient. Exported for the finite-difference test.
export function lossAndGrads(model, params, x, wet, wetE) {
  let K = model.pre.length
  let { pred, preTail, acts, t } = forward(model, x, true)
  let n = Math.min(pred.length, wet.length)
  let loss = 0
  let e = new Float64Array(pred.length)
  for (let i = 0; i < n; i++) {
    let d = pred[i] - wet[i]
    loss += d * d
    e[i] = 2 * d / wetE
  }
  loss /= wetE

  // backward — acts layout: [pre inputs ×K][curve input][post inputs ×K]
  let grads = params.map(([, , d]) => new Float64Array(d))
  let gOf = (o, key) => grads[params.findIndex(([po, pk]) => po === o && pk === key)]
  if (model.tail) e = correlate(e, model.tail)
  for (let si = model.post.length - 1; si >= 0; si--) {
    let sec = model.post[si]
    let inp = acts[K + 1 + si]
    let out = si === model.post.length - 1 ? preTail : acts[K + 2 + si]
    let packed = { b: gOf(sec, 'b'), rho: 0, theta: 0 }
    e = sectionGrads(sec, inp, out, e, packed)
    gOf(sec, 'rho')[0] += packed.rho
    gOf(sec, 'theta')[0] += packed.theta
  }
  // curve node
  {
    let gc = grads[grads.length - 1]
    let P = model.curve.length
    let eIn = new Float64Array(e.length)
    for (let i = 0; i < e.length; i++) {
      let tt = t[i], tp = 1, tc = tt, up = 1, uc = 2 * tt, dfdx = 0
      for (let k = 0; k < P; k++) {
        gc[k] += e[i] * tc
        dfdx += model.curve[k] * (k + 1) * up
        let tn = 2 * tt * tc - tp
        tp = tc; tc = tn
        let un = 2 * tt * uc - up
        up = uc; uc = un
      }
      eIn[i] = e[i] * dfdx / model.scale
    }
    e = eIn
  }
  for (let si = model.pre.length - 1; si >= 0; si--) {
    let sec = model.pre[si]
    let packed = { b: gOf(sec, 'b'), rho: 0, theta: 0 }
    e = sectionGrads(sec, acts[si], acts[si + 1], e, packed)
    gOf(sec, 'rho')[0] += packed.rho
    gOf(sec, 'theta')[0] += packed.theta
  }
  return { loss, grads }
}

export { forward as _forward, paramList as _paramList, initSections as _initSections }

export default function fitWh(dry, wet, opts = {}) {
  let order = opts.order ?? 11
  let K = opts.whSections ?? 2
  let iters = opts.whIters ?? 1500
  let lr0 = opts.whLr ?? 0.1
  let scale = 0
  for (let i = 0; i < dry.length; i++) scale = Math.max(scale, Math.abs(dry[i]))
  if (!scale) scale = 1

  let model = {
    pre: opts.init?.pre ? opts.init.pre.map(sectionFromCoefs) : initSections(K, 1),
    post: opts.init?.post ? opts.init.post.map(sectionFromCoefs) : initSections(K, 0.85),
    curve: new Float64Array(order),
    scale,
    tail: opts.tail ?? null,
  }
  let x = Float64Array.from(dry)
  let wetE = 0
  for (let i = 0; i < wet.length; i++) wetE += wet[i] * wet[i]
  wetE += 1e-30

  // Adam state per parameter group
  let params = paramList(model)
  let m = params.map(([, , d]) => new Float64Array(d))
  let v = params.map(([, , d]) => new Float64Array(d))
  let b1 = 0.9, b2 = 0.999, eps = 1e-8
  let best = null, bestLoss = Infinity, sinceBest = 0
  let targetEsr = 10 ** ((opts.targetDb ?? -60) / 10)

  for (let it = 1; it <= iters; it++) {
    curveLS(model, x, wet)
    let { loss, grads } = lossAndGrads(model, params, x, wet, wetE)

    opts.onProgress?.(it, loss)
    if (!Number.isFinite(loss)) {
      if (best) restore(model, best)
      break
    }
    if (loss < bestLoss) { bestLoss = loss; best = snapshot(model); sinceBest = 0 } else sinceBest++
    if (loss <= targetEsr || sinceBest > 200) break

    // Adam step, cosine-decayed lr
    // curve group (last) is LS-solved, not gradient-stepped
    let lr = lr0 * (0.5 + 0.5 * Math.cos(Math.PI * it / iters))
    let bc1 = 1 - b1 ** it, bc2 = 1 - b2 ** it
    for (let p = 0; p < params.length - 1; p++) {
      let [o, key, d] = params[p]
      let val = o[key]
      for (let j = 0; j < d; j++) {
        let g = grads[p][j]
        m[p][j] = b1 * m[p][j] + (1 - b1) * g
        v[p][j] = b2 * v[p][j] + (1 - b2) * g * g
        let stepv = lr * (m[p][j] / bc1) / (Math.sqrt(v[p][j] / bc2) + eps)
        if (d === 1) o[key] = val - stepv
        else val[j] -= stepv
      }
    }
  }
  if (best) restore(model, best)

  let { pred, preTail } = forward(model, x)
  return {
    rung: 'wh',
    pre: model.pre.map(coefs),
    curve: { type: 'chebyshev', coeffs: [...model.curve], scale: model.scale },
    post: model.post.map(coefs),
    pred,
    preTail,
  }
}

function snapshot(model) {
  return {
    pre: model.pre.map(s => ({ b: [...s.b], rho: s.rho, theta: s.theta })),
    post: model.post.map(s => ({ b: [...s.b], rho: s.rho, theta: s.theta })),
    curve: Float64Array.from(model.curve),
  }
}
function restore(model, snap) {
  model.pre.forEach((s, i) => Object.assign(s, { b: [...snap.pre[i].b], rho: snap.pre[i].rho, theta: snap.pre[i].theta }))
  model.post.forEach((s, i) => Object.assign(s, { b: [...snap.post[i].b], rho: snap.post[i].rho, theta: snap.post[i].theta }))
  model.curve.set(snap.curve)
}
