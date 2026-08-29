import test, { almost, ok, is, throws } from 'tst'
import match, { specLoss, cmaes } from './match.js'

test('cmaes — minimizes rosenbrock and sphere', () => {
  let sphere = x => x.reduce((s, v) => s + v * v, 0)
  let r = cmaes(sphere, [0.8, -0.5, 0.3], 0.3, { maxEvals: 2000 })
  ok(r.fx < 1e-10, 'sphere min ' + r.fx.toExponential(2))
  let rosen = x => (1 - x[0]) ** 2 + 100 * (x[1] - x[0] * x[0]) ** 2
  let r2 = cmaes(rosen, [-1, 1], 0.5, { maxEvals: 4000 })
  ok(r2.fx < 1e-8, 'rosenbrock min ' + r2.fx.toExponential(2))
  almost(r2.x[0], 1, 1e-3, 'x0')
  almost(r2.x[1], 1, 1e-3, 'x1')
})

test('cmaes — deterministic under seed', () => {
  let f = x => (x[0] - 0.3) ** 2 + (x[1] + 0.7) ** 2
  let a = cmaes(f, [0, 0], 0.3, { maxEvals: 500, seed: 42 })
  let b = cmaes(f, [0, 0], 0.3, { maxEvals: 500, seed: 42 })
  is(a.fx, b.fx)
  ok(a.x.every((v, i) => v === b.x[i]), 'same solution')
})

test('specLoss — zero iff same, grows with difference', () => {
  let fs = 22050, n = 1 << 13
  let tone = f => Float32Array.from({ length: n }, (_, i) => 0.5 * Math.sin(2 * Math.PI * f * i / fs))
  is(specLoss(tone(440), tone(440)), 0)
  let near = specLoss(tone(440), tone(466))
  let far = specLoss(tone(440), tone(880))
  ok(near > 0, 'nonzero for different tones')
  ok(far > near, 'farther pitch, larger loss: ' + near.toFixed(3) + ' < ' + far.toFixed(3))
})

// subtractive synth: saw → RBJ lowpass → tanh drive; knobs [pitch, cutoff, drive]
const FS = 22050, LEN = 1 << 13
function subSynth([pitch, cutoff, drive]) {
  let out = new Float32Array(LEN)
  let ph = 0
  for (let i = 0; i < LEN; i++) {
    ph += pitch / FS
    ph -= Math.floor(ph)
    out[i] = 2 * ph - 1
  }
  let w = 2 * Math.PI * cutoff / FS, cw = Math.cos(w), alpha = Math.sin(w) / (2 * 0.707)
  let a0 = 1 + alpha
  let b0 = (1 - cw) / 2 / a0, b1 = (1 - cw) / a0, b2 = b0, a1 = -2 * cw / a0, a2 = (1 - alpha) / a0
  let x1 = 0, x2 = 0, y1 = 0, y2 = 0
  for (let i = 0; i < LEN; i++) {
    let y = b0 * out[i] + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
    x2 = x1; x1 = out[i]; y2 = y1; y1 = y
    out[i] = Math.tanh(drive * y) / Math.tanh(drive)
  }
  return out
}

test('match — recovers hidden subtractive-synth patch', () => {
  let hidden = [220, 1200, 3]
  let target = subSynth(hidden)
  let r = match(subSynth, target, {
    bounds: [[80, 800], [300, 6000], [1, 8]],
    budget: 2400,
    loss: (a, b) => specLoss(a, b, { fs: FS }),
  })
  ok(r.loss < 0.05, 'matched: loss ' + r.loss.toFixed(4))
  almost(r.params[0], 220, 2, 'pitch ' + r.params[0].toFixed(1))
  almost(r.params[1], 1200, 60, 'cutoff ' + r.params[1].toFixed(0))
  almost(r.params[2], 3, 0.3, 'drive ' + r.params[2].toFixed(2))
})

test('match — FM synth: recovers carrier, ratio, index', () => {
  let fm = ([fc, ratio, index]) => {
    let out = new Float32Array(LEN)
    for (let i = 0; i < LEN; i++) {
      let t = i / FS
      out[i] = 0.5 * Math.sin(2 * Math.PI * fc * t + index * Math.sin(2 * Math.PI * fc * ratio * t))
    }
    return out
  }
  let target = fm([330, 2, 1.5])
  let r = match(fm, target, {
    bounds: [[100, 900], [0.5, 4], [0, 5]],
    budget: 3000,
    loss: (a, b) => specLoss(a, b, { fs: FS }),
  })
  ok(r.loss < 0.05, 'FM matched: loss ' + r.loss.toFixed(4))
  almost(r.params[0], 330, 3, 'carrier ' + r.params[0].toFixed(1))
  almost(r.params[1], 2, 0.05, 'ratio ' + r.params[1].toFixed(3))
})

test('match — guards', () => {
  throws(() => match(() => new Float32Array(64), new Float32Array(64), {}))
})
