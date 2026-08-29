import test, { almost, ok, is, throws } from 'tst'
import { readFileSync, existsSync } from 'node:fs'
import nam, { parse, process } from './nam.js'

// MIT-licensed example model (random weights — format/parse tests only)
const STANDARD = readFileSync(new URL('./fixtures/wavenet_a1_standard.nam', import.meta.url), 'utf8')
// Real trained capture (community model, no explicit license — fetched, not
// committed; see fixtures/README.md). Behavior tests skip when absent.
const REAL_PATH = new URL('./fixtures/5150.nam', import.meta.url)
const REAL = existsSync(REAL_PATH) ? readFileSync(REAL_PATH, 'utf8') : null
const realTest = REAL ? test : test.skip

// Minimal hand-computable model: 1 array, 1 layer, 1 channel, kernel 2,
// dilation 2 — pins the weight order and the causal time alignment.
// forward: y0 = r·x; z[i] = w0·y0[i−2] + w1·y0[i] + b + m·x[i]; a = tanh(z);
// head_rc = hr·a + hb; out = s·head_rc; residual (unused: single layer) = y0 + w2·a + b2
const MICRO = {
  version: '0.5.0',
  architecture: 'WaveNet',
  config: {
    layers: [{
      input_size: 1, condition_size: 1, head_size: 1, channels: 1,
      kernel_size: 2, dilations: [2], activation: 'Tanh', gated: false, head_bias: true,
    }],
    head: null,
    head_scale: 0.5,
  },
  //        rech  conv w0,w1  b     mixin  1x1 w  1x1 b  headrc hb    scale
  weights: [2.0,  0.3, 0.7,   0.1,  0.4,   0.9,   0.05,  1.5,   0.2,  0.5],
}

test('nam — micro model matches hand computation', () => {
  let amp = nam(MICRO)
  is(amp.receptiveField, 3)
  let x = Float32Array.from([0.1, -0.2, 0.3, 0.05])
  let expected = x.map((_, i) => {
    let y0 = i => (i < 0 ? 0 : 2.0 * x[i])
    let z = 0.3 * y0(i - 2) + 0.7 * y0(i) + 0.1 + 0.4 * x[i]
    return 0.5 * (1.5 * Math.tanh(z) + 0.2)
  })
  let out = amp(Float32Array.from(x))
  for (let i = 0; i < x.length; i++) almost(out[i], expected[i], 1e-6, `sample ${i}: ${out[i]} vs ${expected[i]}`)
})

test('nam — standard model parses with exact weight consumption', () => {
  let model = parse(STANDARD)
  is(model.arrays.length, 2)
  is(model.receptiveField, 4093)
  almost(model.headScale, 0.02, 1e-6, 'head_scale')
})

realTest('nam — real capture: silence in, near-silence out past warmup', () => {
  let amp = nam(REAL)
  let out = amp(new Float32Array(16384))
  let peak = 0
  for (let i = amp.receptiveField; i < out.length; i++) peak = Math.max(peak, Math.abs(out[i]))
  ok(peak < 0.01, 'steady idle level ' + peak.toFixed(5))
})

realTest('nam — real capture: sine grows harmonics, output bounded', () => {
  let amp = nam(REAL)
  let n = 1 << 14, fs = 48000, f = 220
  let x = new Float32Array(n)
  for (let i = 0; i < n; i++) x[i] = 0.25 * Math.sin(2 * Math.PI * f * i / fs)
  let out = amp(x)
  let peak = 0
  for (let v of out) peak = Math.max(peak, Math.abs(v))
  ok(peak > 0.01 && peak < 4, 'output level ' + peak.toFixed(3))
  // Goertzel energy at f, 2f, 3f over the steady tail
  let g = freq => {
    let re = 0, im = 0
    for (let i = n >> 1; i < n; i++) {
      let ph = 2 * Math.PI * freq * i / fs
      re += out[i] * Math.cos(ph)
      im -= out[i] * Math.sin(ph)
    }
    return Math.hypot(re, im)
  }
  let h1 = g(f), h2 = g(2 * f), h3 = g(3 * f)
  ok(h1 > 0, 'fundamental present')
  ok((h2 + h3) / h1 > 1e-3, 'harmonic distortion present: ' + ((h2 + h3) / h1).toFixed(4))
})

test('nam — deterministic, mutates in place', () => {
  let amp = nam(STANDARD)
  let x = Float32Array.from({ length: 4096 }, (_, i) => 0.3 * Math.sin(i / 30))
  let a = amp(Float32Array.from(x))
  let b = amp(Float32Array.from(x))
  let same = true
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) { same = false; break }
  ok(same, 'deterministic')
  let y = Float32Array.from(x)
  is(amp(y), y, 'in place')
})

test('nam — unsupported variants throw', () => {
  throws(() => parse({ architecture: 'LSTM', config: {}, weights: [] }))
  let gated = JSON.parse(JSON.stringify(MICRO))
  gated.config.layers[0].gated = true
  throws(() => parse(gated))
})
