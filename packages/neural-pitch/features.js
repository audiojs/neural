// Model input: one variable-Q spectrum per frame, centred on the frame's time, at any sample rate.
//
// Bins 3 a semitone (36 an octave). Bin k has bandwidth α·f + γ, α = 2^(1/36) − 1, γ = 12 Hz: the
// variable-Q transform of Schörkhuber, Klapuri, Holighaus & Dörfler (AES 53rd Conference, 2014), with
// window lengths fs / (α·f + γ) as librosa.vqt sets them. Constant-Q above ≈ 600 Hz, where a voice's
// harmonics sit; near-constant bandwidth below, so the longest window, 84 ms, fits a 4096-point FFT
// at 48 kHz. Hann windows, one FFT a frame and a sparse spectral kernel a bin (Brown & Puckette,
// "An efficient algorithm for the calculation of a constant Q transform", JASA 92(5), 1992). Windows
// are set in seconds, so a sound gives the same spectrum at every sample rate up to its Nyquist.
//
// Model input: dB relative to the loudest bin, floor 80 dB below it, scaled to 0…1; `level`, the
// loudest bin in dBFS (a full-scale sine reads −6), is the one absolute quantity it sees.

import { fft, cfft } from 'fourier-transform'

export const PER_SEMITONE = 3
export const GAMMA = 12
export const LOW = 23             // MIDI of input bin 0 (B0, 30.87 Hz)
export const BINS = 288           // 96 semitones, to MIDI 118⅔ (7.6 kHz)
export const MARGIN = 15          // bins computed beyond both ends for training's transpositions
export const TOP = 80             // dB kept below the loudest bin
const FLOOR = 1e-7                // amplitude added before the log (−140 dBFS): silence stays finite
const ALPHA = 2 ** (1 / (12 * PER_SEMITONE)) - 1

export const hz = midi => 440 * 2 ** ((midi - 69) / 12)
const width = (f, fs) => 2 * Math.round(fs / (ALPHA * f + GAMMA) / 2) + 1   // odd window length
const cache = new Map()

/**
 * Analyzer at sample rate `fs` for `bins` bins from `first` (a bin index, negative into the margin).
 * `n`: FFT size, the samples a frame reads, centred on sample n/2; the same at every range, so the
 * kernels of a bin never change. `db(x, centre, out)`: dB of each bin around x[centre], zeros outside x.
 */
export function analyzer(fs, first = 0, bins = BINS) {
	let key = `${fs} ${first} ${bins}`
	if (cache.has(key)) return cache.get(key)
	if (!(fs >= 4000)) throw new RangeError(`neural-pitch: sample rate ${fs} below 4000`)
	let n = 2 ** Math.ceil(Math.log2(width(hz(LOW - MARGIN / PER_SEMITONE), fs))), half = n >> 1
	let freq = Float64Array.from({ length: bins }, (_, k) => hz(LOW + (first + k) / PER_SEMITONE))

	// kernel k: Hann(L) · e^{2πi f (t − centre)} / Σ Hann, its spectrum conjugated over n where above 1e-5
	// of its peak (−100 dB, under the input's floor); for m > n/2, X[m] = conj X[n − m], so those taps go to a mirrored list
	let pos = [], neg = [], re = new Float64Array(n), im = new Float64Array(n), none = [new Int32Array(0), new Float64Array(0), new Float64Array(0)]
	for (let k = 0; k < bins; k++) {
		// a bin whose band reaches past Nyquist would alias: it stays empty, at the floor
		if (freq[k] + ALPHA * freq[k] + GAMMA > fs / 2) { pos.push(none); neg.push(none); continue }
		let L = Math.min(width(freq[k], fs), n - 1), h = (L - 1) >> 1, sum = 0
		re.fill(0); im.fill(0)
		for (let j = 0; j < L; j++) sum += Math.sin(Math.PI * (j + 1) / (L + 1)) ** 2
		for (let j = 0; j < L; j++) {
			let w = Math.sin(Math.PI * (j + 1) / (L + 1)) ** 2 / sum, ph = 2 * Math.PI * freq[k] * (j - h) / fs
			re[half - h + j] = w * Math.cos(ph); im[half - h + j] = w * Math.sin(ph)
		}
		cfft(re, im)
		let peak = 0
		for (let m = 0; m < n; m++) peak = Math.max(peak, re[m] * re[m] + im[m] * im[m])
		let p = [[], [], []], q = [[], [], []]
		for (let m = 0; m < n; m++) {
			if (re[m] * re[m] + im[m] * im[m] < 1e-10 * peak) continue
			let [idx, cr, ci] = m <= half ? p : q
			idx.push(m <= half ? m : n - m); cr.push(re[m] / n); ci.push(-im[m] / n)
		}
		pos.push(p.map((a, i) => i ? Float64Array.from(a) : Int32Array.from(a)))
		neg.push(q.map((a, i) => i ? Float64Array.from(a) : Int32Array.from(a)))
	}

	let buf = new Float64Array(n), X = [new Float64Array(half + 1), new Float64Array(half + 1)]
	function db(x, centre, out = new Float64Array(bins)) {
		let start = centre - half
		for (let i = 0; i < n; i++) { let j = start + i; buf[i] = j >= 0 && j < x.length ? x[j] : 0 }
		fft(buf, X)
		let [Xr, Xi] = X
		for (let k = 0; k < bins; k++) {
			// Σ X[m]·conj K[m] over the positive taps, conj X[m]·conj K[n − m] over the mirrored ones
			let r = 0, s = 0, [pi, pr, pc] = pos[k], [ni, nr, nc] = neg[k]
			for (let t = 0; t < pi.length; t++) { let a = Xr[pi[t]], b = Xi[pi[t]]; r += a * pr[t] - b * pc[t]; s += a * pc[t] + b * pr[t] }
			for (let t = 0; t < ni.length; t++) { let a = Xr[ni[t]], b = -Xi[ni[t]]; r += a * nr[t] - b * nc[t]; s += a * nc[t] + b * nr[t] }
			out[k] = 20 * Math.log10(Math.sqrt(r * r + s * s) + FLOOR)
		}
		return out
	}
	let a = { fs, first, bins, n, freq, db }
	cache.set(key, a)
	return a
}

/** dB spectrum → model input in 0…1 (from `offset`, `out.length` bins); returns the level (dBFS). */
export function normalize(db, out, offset = 0) {
	let max = -Infinity
	for (let k = 0; k < out.length; k++) if (db[offset + k] > max) max = db[offset + k]
	for (let k = 0; k < out.length; k++) out[k] = Math.max(db[offset + k] - max, -TOP) / TOP + 1
	return max
}
