import test, { almost, ok, is, throws } from 'tst'
import capture, { render, esr, nullDepth } from './capture.js'
import { lossAndGrads, _paramList, _initSections } from './wh.js'
import probe from './probe.js'

// deterministic pseudo-noise (Lehmer LCG) — full-band excitation, reproducible
function noise(n, amp = 0.9) {
  let seed = 123456789
  let x = new Float32Array(n)
  for (let i = 0; i < n; i++) {
    seed = (seed * 48271) % 2147483647
    x[i] = amp * (seed / 2147483647 * 2 - 1)
  }
  return x
}

const TAPS = [[0, 1], [220, 0.5], [800, -0.25]]
function fir(x, taps = TAPS) {
  let y = new Float32Array(x.length)
  for (let [d, g] of taps) for (let i = 0; i + d < x.length; i++) y[i + d] += g * x[i]
  return y
}

const drive = 2.5
const tanhCurve = x => Math.tanh(drive * x) / Math.tanh(drive)

test('lti — recovers a known multi-tap system, nulls deep', () => {
  let dry = noise(1 << 15)
  let wet = fir(dry)
  let r = capture(dry, wet)
  is(r.rung, 'lti')
  ok(r.nullDb <= -60, 'null ' + r.nullDb.toFixed(1) + ' dB')
  is(r.attempts.length, 1, 'no escalation')
  for (let [d, g] of TAPS) almost(r.ir[d], g, 0.02, 'tap @' + d + ' = ' + r.ir[d].toFixed(3))
})

test('lti — identity device yields unit impulse', () => {
  let dry = noise(1 << 14)
  let r = capture(dry, Float32Array.from(dry))
  is(r.rung, 'lti')
  almost(r.ir[0], 1, 0.01, 'δ height ' + r.ir[0].toFixed(4))
  ok(r.nullDb <= -80, 'null ' + r.nullDb.toFixed(1) + ' dB')
})

test('lti — reverb-scale decaying tail survives depth detection', () => {
  let dry = noise(1 << 15)
  // noisy exponential decay, 3000 taps, −60 dB at the end
  let h = new Float32Array(3000)
  let seed = 42
  for (let i = 0; i < h.length; i++) {
    seed = (seed * 48271) % 2147483647
    h[i] = (seed / 2147483647 * 2 - 1) * Math.exp(-i / 434)
  }
  h[0] = 1
  let wet = new Float32Array(dry.length)
  for (let i = 0; i < h.length; i++) for (let j = 0; i + j < dry.length; j++) wet[i + j] += h[i] * dry[j]
  let r = capture(dry, wet)
  is(r.rung, 'lti')
  ok(r.ir.length >= 2500, 'tail kept: ' + r.ir.length + ' taps')
  ok(r.nullDb <= -60, 'null ' + r.nullDb.toFixed(1) + ' dB')
})

test('hammerstein — waveshaper→FIR device: escalates, nulls deep', () => {
  let dry = noise(1 << 15)
  let wet = fir(Float32Array.from(dry, tanhCurve))
  let r = capture(dry, wet)
  is(r.rung, 'hammerstein')
  ok(r.nullDb <= -40, 'null ' + r.nullDb.toFixed(1) + ' dB')
  is(r.attempts.length, 2, 'escalated once')
  ok(r.attempts[0].nullDb > -30, 'lti rung insufficient: ' + r.attempts[0].nullDb.toFixed(1) + ' dB')
  is(r.stages.length, 2)
  is(r.stages[0].atom, '@audio/saturate-waveshaper')
  is(r.stages[1].atom, '@audio/reverb-convolution')
})

test('hammerstein — pure static nonlinearity: IR collapses to delta', () => {
  let dry = noise(1 << 14)
  let wet = Float32Array.from(dry, tanhCurve)
  let r = capture(dry, wet)
  is(r.rung, 'hammerstein')
  ok(r.nullDb <= -40, 'null ' + r.nullDb.toFixed(1) + ' dB')
  // linear stage carries no memory: energy concentrated at lag 0
  let e0 = r.ir[0] * r.ir[0], et = 0
  for (let v of r.ir) et += v * v
  ok(e0 / et > 0.99, 'delta-like IR (' + (100 * e0 / et).toFixed(1) + '% @0)')
})

test('maxRung — caps the ladder', () => {
  let dry = noise(1 << 14)
  let wet = fir(Float32Array.from(dry, tanhCurve))
  let r = capture(dry, wet, { maxRung: 'lti' })
  is(r.rung, 'lti')
  is(r.attempts.length, 1)
})

test('render — reproduces the reported null', () => {
  let dry = noise(1 << 15)
  let wet = fir(Float32Array.from(dry, tanhCurve))
  let r = capture(dry, wet)
  let d = nullDepth(wet, render(r, dry))
  almost(d, r.nullDb, 0.5, 'render null ' + d.toFixed(1) + ' vs reported ' + r.nullDb.toFixed(1))
})

test('stages — JSON-safe chain recipe', () => {
  let dry = noise(1 << 13)
  let r = capture(dry, fir(dry))
  let json = JSON.parse(JSON.stringify(r.stages))
  is(json[json.length - 1].params.ir.length, r.ir.length)
  ok(typeof json[json.length - 1].why === 'string')
})

const biquad = (x, c) => {
  let x1 = 0, x2 = 0, y1 = 0, y2 = 0
  let out = new Float64Array(x.length)
  for (let i = 0; i < x.length; i++) {
    let y = c.b0 * x[i] + c.b1 * x1 + c.b2 * x2 - c.a1 * y1 - c.a2 * y2
    x2 = x1; x1 = x[i]; y2 = y1; y1 = y
    out[i] = y
  }
  return out
}

test('wh — biquad→tanh→biquad device: lower rungs fail, wh nulls deep', () => {
  let dry = noise(1 << 13)
  let pre = { b0: .6, b1: .3, b2: .1, a1: -.4, a2: .2 }
  let post = { b0: .9, b1: -.5, b2: .2, a1: -.3, a2: .15 }
  let wet = biquad(Float64Array.from(biquad(dry, pre), tanhCurve), post)
  let r = capture(dry, wet, { targetDb: -50 })
  is(r.rung, 'wh')
  ok(r.attempts[1].nullDb > -30, 'hammerstein insufficient: ' + r.attempts[1].nullDb.toFixed(1) + ' dB')
  ok(r.nullDb <= -50, 'null ' + r.nullDb.toFixed(1) + ' dB')
  is(r.stages.length, 3)
  is(r.stages[0].name, 'cascade')
  is(r.stages[1].name, 'waveshaper')
  is(r.stages[2].name, 'cascade')
  let d = nullDepth(wet, render(r, dry))
  almost(d, r.nullDb, 0.5, 'render null ' + d.toFixed(1) + ' vs reported ' + r.nullDb.toFixed(1))
})

test('wh-tail — biquad→tanh→long-FIR device: factorizes drive + tail', () => {
  let dry = noise(1 << 13)
  let pre = { b0: .6, b1: .3, b2: .1, a1: -.4, a2: .2 }
  let u = Float64Array.from(biquad(dry, pre), tanhCurve)
  let wet = new Float64Array(dry.length)
  for (let [d, g] of [[0, 1], [130, 0.5], [500, -0.25]]) for (let i = 0; i + d < dry.length; i++) wet[i + d] += g * u[i]
  let r = capture(dry, wet, { targetDb: -45 })
  is(r.rung, 'wh-tail')
  ok(r.attempts[1].nullDb > -25, 'hammerstein insufficient: ' + r.attempts[1].nullDb.toFixed(1) + ' dB')
  ok(r.nullDb <= -45, 'null ' + r.nullDb.toFixed(1) + ' dB')
  is(r.stages[r.stages.length - 1].name, 'convolve')
  let d = nullDepth(wet, render(r, dry))
  almost(d, r.nullDb, 0.5, 'render null ' + d.toFixed(1) + ' vs reported ' + r.nullDb.toFixed(1))
})

test('wh — analytic gradients match finite differences', () => {
  let seed = 7
  let rnd = () => { seed = (seed * 48271) % 2147483647; return seed / 2147483647 * 2 - 1 }
  let n = 256
  let x = Float64Array.from({ length: n }, () => 0.8 * rnd())
  let wet = Float64Array.from({ length: n }, () => 0.5 * rnd())
  let model = { pre: _initSections(1, 1), post: _initSections(1, 0.85), curve: Float64Array.from([0.7, 0.1, -0.2]), scale: 0.8 }
  model.pre[0].b = [0.9, 0.2, -0.1]; model.pre[0].rho = 0.3; model.pre[0].theta = 1.1
  model.post[0].b = [1.1, -0.3, 0.05]; model.post[0].rho = -0.2; model.post[0].theta = 2.0
  let wetE = wet.reduce((s, v) => s + v * v, 0)
  let params = _paramList(model)
  let { grads } = lossAndGrads(model, params, x, wet, wetE)
  let h = 1e-6
  for (let p = 0; p < params.length; p++) {
    let [o, key, d] = params[p]
    for (let j = 0; j < d; j++) {
      let orig = d === 1 ? o[key] : o[key][j]
      let set = val => { if (d === 1) o[key] = val; else o[key][j] = val }
      set(orig + h)
      let lp = lossAndGrads(model, params, x, wet, wetE).loss
      set(orig - h)
      let lm = lossAndGrads(model, params, x, wet, wetE).loss
      set(orig)
      let num = (lp - lm) / (2 * h)
      let rel = Math.abs(num - grads[p][j]) / (Math.abs(num) + Math.abs(grads[p][j]) + 1e-12)
      ok(rel < 1e-5, key + '[' + j + '] rel err ' + rel.toExponential(1))
    }
  }
})

test('probe — deterministic, layout matches signal', () => {
  let a = probe({ fs: 8000 })
  let b = probe({ fs: 8000 })
  is(a.signal.length, b.signal.length)
  let same = true
  for (let i = 0; i < a.signal.length; i++) if (a.signal[i] !== b.signal[i]) { same = false; break }
  ok(same, 'deterministic')
  let end = a.layout[a.layout.length - 1]
  is(a.signal.length, end.start + end.duration, 'layout covers signal')
  let ramp = a.layout.find(s => s.type === 'ramp')
  let peak = 0
  for (let i = ramp.start; i < ramp.start + ramp.duration; i++) peak = Math.max(peak, Math.abs(a.signal[i]))
  ok(peak > 0.9 && peak <= 0.95, 'ramp reaches full level: ' + peak.toFixed(3))
  let click = a.layout.find(s => s.type === 'click')
  almost(a.signal[click.start], 0.9, 1e-6, 'click amplitude')
})

test('probe — captures a nonlinear device through the full ladder', () => {
  let { signal: dry } = probe({ fs: 8000, sweepDuration: 1, noiseDuration: 0.5, rampDuration: 1 })
  let wet = Float32Array.from(dry, tanhCurve)
  let r = capture(dry, wet, { maxRung: 'hammerstein' })
  ok(r.nullDb <= -40, 'null ' + r.nullDb.toFixed(1) + ' dB')
})

test('align — positive latency trimmed and reported', () => {
  let dry = noise(1 << 14)
  let wet = new Float32Array(dry.length)
  for (let i = 0; i + 137 < dry.length; i++) wet[i + 137] = 0.7 * dry[i]
  let r = capture(dry, wet)
  is(r.latency, 137)
  is(r.rung, 'lti')
  almost(r.ir[0], 0.7, 0.01, 'gain at lag 0: ' + r.ir[0].toFixed(3))
  ok(r.nullDb <= -60, 'null ' + r.nullDb.toFixed(1) + ' dB')
})

test('align — negative latency (wet leads) still captures', () => {
  let dry = noise(1 << 14)
  // wet[i] = 0.7·dry[i+90]: device output recorded 90 samples early
  let wet = new Float32Array(dry.length)
  for (let i = 0; i + 90 < dry.length; i++) wet[i] = 0.7 * dry[i + 90]
  let r = capture(dry, wet)
  is(r.latency, -90)
  ok(r.nullDb <= -60, 'null ' + r.nullDb.toFixed(1) + ' dB')
})

test('metrics — esr/nullDepth basics', () => {
  let a = Float32Array.from([1, -1, 1, -1])
  is(esr(a, a), 0)
  almost(nullDepth(a, Float32Array.from([0.9, -0.9, 0.9, -0.9])), -20, 0.01)
})

test('guards — empty input throws', () => {
  throws(() => capture(new Float32Array(0), new Float32Array(0)))
})
