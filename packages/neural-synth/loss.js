// Multi-scale spectral distance — the perceptual objective for sound matching
// (Engel et al. 2020, DDSP; the standard loss of the synth-matching literature).
// Frame both signals at several FFT sizes, compare magnitude spectra linearly
// and log-scaled: coarse windows judge tone, fine windows judge transients.

import { fft } from 'fourier-transform'

let _hann = new Map()
function hann(N) {
  let w = _hann.get(N)
  if (w) return w
  w = new Float64Array(N)
  for (let i = 0; i < N; i++) w[i] = 0.5 * (1 - Math.cos(2 * Math.PI * i / N))
  _hann.set(N, w)
  return w
}

function mags(x, pos, N, out) {
  let w = hann(N)
  let f = new Float64Array(N)
  for (let i = 0; i < N; i++) f[i] = (x[pos + i] || 0) * w[i]
  let [re, im] = fft(f)
  let half = N >> 1
  for (let k = 0; k <= half; k++) out[k] = Math.hypot(re[k], im[k])
}

// Triangular mel filterbank (HTK mel), rows normalized to unit sum.
let _mel = new Map()
function melBank(half, bands, fs) {
  let key = half + ':' + bands + ':' + fs
  let bank = _mel.get(key)
  if (bank) return bank
  let mel = f => 2595 * Math.log10(1 + f / 700)
  let imel = m => 700 * (10 ** (m / 2595) - 1)
  let lo = mel(20), hi = mel(fs / 2)
  let centers = Float64Array.from({ length: bands + 2 }, (_, i) => imel(lo + (hi - lo) * i / (bands + 1)))
  bank = []
  for (let b = 0; b < bands; b++) {
    let [fl, fc, fr] = [centers[b], centers[b + 1], centers[b + 2]]
    let row = new Float64Array(half + 1)
    let sum = 0
    for (let k = 0; k <= half; k++) {
      let f = k * fs / (2 * half)
      let w = f <= fl || f >= fr ? 0 : f <= fc ? (f - fl) / (fc - fl) : (fr - f) / (fr - fc)
      row[k] = w
      sum += w
    }
    if (sum > 0) for (let k = 0; k <= half; k++) row[k] /= sum
    bank.push(row)
  }
  _mel.set(key, bank)
  return bank
}

// specLoss(a, b, { fftSizes = [2048, 512, 128], fs = 44100, melBands = 48 }) →
// scalar ≥ 0, 0 iff spectra match. Two terms per scale: per-frame linear
// spectral convergence (fine detail) + log-mel band distance on the averaged
// spectrum (wide, smooth basins — what makes pitch/cutoff optimizable at all).
export default function specLoss(a, b, opts = {}) {
  let sizes = opts.fftSizes ?? [2048, 512, 128]
  let fs = opts.fs ?? 44100
  let bands = opts.melBands ?? 48
  let n = Math.min(a.length, b.length)
  let total = 0
  for (let N of sizes) {
    if (N > n) continue
    let hop = N >> 2
    let half = N >> 1
    let ma = new Float64Array(half + 1), mb = new Float64Array(half + 1)
    let avgA = new Float64Array(half + 1), avgB = new Float64Array(half + 1)
    let lin = 0, ref = 0, frames = 0
    for (let pos = 0; pos + N <= n; pos += hop) {
      mags(a, pos, N, ma)
      mags(b, pos, N, mb)
      for (let k = 0; k <= half; k++) {
        let d = ma[k] - mb[k]
        lin += d * d
        ref += ma[k] * ma[k]
        avgA[k] += ma[k]
        avgB[k] += mb[k]
      }
      frames++
    }
    if (!frames) continue
    let bank = melBank(half, bands, fs)
    let mel = 0
    for (let row of bank) {
      let ea = 0, eb = 0
      for (let k = 0; k <= half; k++) {
        ea += row[k] * avgA[k]
        eb += row[k] * avgB[k]
      }
      mel += Math.abs(Math.log(ea / frames + 1e-7) - Math.log(eb / frames + 1e-7))
    }
    total += Math.sqrt(lin / (ref + 1e-12)) + mel / bands
  }
  return total
}
