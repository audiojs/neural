// Standardized capture excitation — the dry signal to play through a device.
// Covers the identification manifold in ~16 s: a click (sync/alignment), an ESS
// sweep (linear response, Farina 2000), noise at four levels (level-dependent
// behavior: compression, saturation depth), and an amplitude-ramped triangle
// (dense static-curve coverage of the full input range). Deterministic: the same
// options always produce the identical signal, so any wet recording pairs with a
// regenerated dry. NAM's ~3 min training file is the prior art; this is the
// compact audiojs equivalent.

import chirp from '@audio/synth-chirp'

const LEVELS = [0.06, 0.25, 0.5, 0.95]

export default function probe(opts = {}) {
  let fs = opts.fs ?? 48000
  let sweepDur = opts.sweepDuration ?? 4
  let noiseDur = opts.noiseDuration ?? 2
  let rampDur = opts.rampDuration ?? 3
  let levels = opts.levels ?? LEVELS

  let layout = []
  let parts = []
  let pos = 0
  let push = (type, samples, fill, extra) => {
    layout.push({ type, start: pos, duration: samples, ...extra })
    let seg = new Float32Array(samples)
    if (fill) fill(seg)
    parts.push(seg)
    pos += samples
  }
  let silence = sec => push('silence', Math.round(sec * fs))

  silence(0.25)
  push('click', 1, seg => (seg[0] = 0.9))
  silence(0.25)

  let f0 = 20, f1 = fs / 2 * 0.95
  push('sweep', Math.round(sweepDur * fs), seg =>
    seg.set(chirp({ f0, f1, duration: sweepDur, amp: 0.5, fs })), { f0, f1 })
  silence(0.25)

  let seed = 123456789
  for (let level of levels) {
    push('noise', Math.round(noiseDur * fs), seg => {
      for (let i = 0; i < seg.length; i++) {
        seed = (seed * 48271) % 2147483647
        seg[i] = level * (seed / 2147483647 * 2 - 1)
      }
    }, { level })
    silence(0.1)
  }

  // 100 Hz triangle, amplitude 0 → 0.95: every input level visited densely
  push('ramp', Math.round(rampDur * fs), seg => {
    let n = seg.length
    for (let i = 0; i < n; i++) {
      let ph = (i * 100 / fs) % 1
      let tri = ph < 0.5 ? 4 * ph - 1 : 3 - 4 * ph
      seg[i] = 0.95 * (i / n) * tri
    }
  }, { f: 100, level: 0.95 })
  silence(0.5)

  let signal = new Float32Array(pos)
  let off = 0
  for (let seg of parts) { signal.set(seg, off); off += seg.length }
  return { signal, fs, layout }
}
