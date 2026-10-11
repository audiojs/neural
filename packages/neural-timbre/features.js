// Timbre features: one note at 44.1 kHz, cut at its onset, as a level-normalized constant-Q
// spectrogram. The same function prepares training views and queries, so the encoder never sees a
// frontend it was not trained on. Chosen over a log-mel spectrogram by measurement (README,
// Features): same data, encoder and steps, 71.7% against 61.0% development top-1.
//
// Constant-Q transform by Brown & Puckette, "An efficient algorithm for the calculation of a
// constant Q transform", JASA 92(5), 1992: one FFT a frame, multiplied by a sparse spectral kernel
// per bin. Bins 12 an octave from E1 (41.2 Hz) to 9956 Hz, Q = 1/(2^(1/12) - 1) (Brown, "Calculation
// of a constant Q spectral transform", JASA 89(1), 1991), Hann windows of Q periods, capped at the
// 8192-sample FFT, so bins below 90 Hz keep 186 ms resolution. Power in dB relative to the loudest
// cell with an 80 dB floor, as librosa.power_to_db(ref=np.max, top_db=80) but with no absolute
// floor (its amin): the features of a note and of any scaled copy are the same.

import { fft, cfft } from 'fourier-transform'
import resample from '@audio/resample-sinc'

// The encoder's input contract. A catalog binds its hash (featuresSha256) beside the encoder's.
// Frame t is centred `center` samples after t hops from the onset.
export const FEATURES = Object.freeze({ kind: 'cqt', sampleRate: 44100, n: 8192, hop: 512, bands: 96, fmin: 41.2034, perOctave: 12, frames: 128, top: 80, center: 1024 })

let kernel = null
// Per bin: the FFT bins where its kernel's spectrum exceeds 0.54% of its peak (Brown & Puckette's
// sparsity threshold), with the conjugate kernel over n.
function kernels({ sampleRate: fs, n, bands, fmin, perOctave } = FEATURES) {
	if (kernel) return kernel
	let Q = 1 / (2 ** (1 / perOctave) - 1)
	kernel = []
	for (let k = 0; k < bands; k++) {
		let f = fmin * 2 ** (k / perOctave), len = Math.min(n, Math.round(Q * fs / f))
		let re = new Float64Array(n), im = new Float64Array(n), start = (n >> 1) - (len >> 1)
		for (let j = 0; j < len; j++) {
			let w = (0.5 - 0.5 * Math.cos(2 * Math.PI * j / len)) / len, ph = 2 * Math.PI * f * (j - (len >> 1)) / fs
			re[start + j] = w * Math.cos(ph); im[start + j] = w * Math.sin(ph)
		}
		cfft(re, im)
		let peak = 0
		for (let j = 0; j <= n / 2; j++) peak = Math.max(peak, Math.hypot(re[j], im[j]))
		let idx = [], kr = [], ki = []
		for (let j = 0; j <= n / 2; j++) if (Math.hypot(re[j], im[j]) > 0.0054 * peak) { idx.push(j); kr.push(re[j] / n); ki.push(-im[j] / n) }
		kernel.push([Int32Array.from(idx), Float64Array.from(kr), Float64Array.from(ki)])
	}
	return kernel
}

// Mono Float32Array at the features' rate, from any rate; stereo or more averages the channels.
export function toRate(audio, sampleRate, f = FEATURES) {
	let x = Array.isArray(audio) ? mix(audio) : audio
	if (!(sampleRate > 0)) throw new RangeError('neural-timbre: sampleRate required')
	return sampleRate === f.sampleRate ? Float32Array.from(x) : resample(Float32Array.from(x), { from: sampleRate, to: f.sampleRate })
}

function mix(channels) {
	if (channels.length === 1) return channels[0]
	let out = new Float32Array(channels[0].length)
	for (let c of channels) for (let i = 0; i < out.length; i++) out[i] += c[i] / channels.length
	return out
}

// First sample of the note: the hop before the first hop within 60 dB of the loudest one. Leading
// silence or room tone would otherwise shift every frame the encoder reads.
export function onset(x, f = FEATURES) {
	let hop = f.hop, e = [], max = 0
	for (let i = 0; i + hop <= x.length; i += hop) {
		let s = 0
		for (let j = i; j < i + hop; j++) s += x[j] * x[j]
		e.push(s); if (s > max) max = s
	}
	if (!(max > 0)) return 0
	let t = e.findIndex(s => s >= max * 1e-6)
	return Math.max(0, (t - 1) * hop)
}

// features(audio, { sampleRate }) → Float32Array(bands × frames), band-major, values in [-1, 1]:
// dB below the loudest cell over 40, plus 1. Shorter notes pad with silence (the floor).
export function features(audio, { sampleRate = FEATURES.sampleRate, trim = true } = {}) {
	let x = toRate(audio, sampleRate)
	return cqt(trim ? x.subarray(onset(x)) : x)
}

// Constant-Q power of a note that starts at x[0], in dB re its loudest cell, scaled to [-1, 1].
export function cqt(x, f = FEATURES) {
	let { n, hop, bands, frames, top, center } = f, K = kernels(f), frame = new Float64Array(n)
	let out = new Float32Array(bands * frames), max = 0
	for (let t = 0; t < frames; t++) {
		let at = t * hop + center - (n >> 1)
		for (let i = 0; i < n; i++) frame[i] = x[at + i] || 0
		let [re, im] = fft(frame)
		for (let b = 0; b < bands; b++) {
			let [idx, kr, ki] = K[b], sr = 0, si = 0
			for (let j = 0; j < idx.length; j++) { let m = idx[j]; sr += re[m] * kr[j] - im[m] * ki[j]; si += re[m] * ki[j] + im[m] * kr[j] }
			let p = sr * sr + si * si
			out[b * frames + t] = p
			if (p > max) max = p
		}
	}
	for (let i = 0; i < out.length; i++) { let db = out[i] > 0 ? 10 * Math.log10(out[i] / max) : -top; out[i] = Math.max(db, -top) / (top / 2) + 1 }
	return out
}
