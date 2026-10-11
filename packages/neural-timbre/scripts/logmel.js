// Log-mel spectrogram for the feature pilot (README, Features): the frontend the constant-Q one was
// measured against, on the same framing, level normalization and frame count. As librosa computes
// it: Slaney mel scale and area normalization (librosa.filters.mel), power spectrum of a
// periodic-Hann STFT without centering (librosa.stft(center=False)), dB re the loudest cell with an
// 80 dB floor (librosa.power_to_db(ref=np.max, top_db=80)); it matched librosa 1.0.0 to 3e-6 dB.
import { fft } from 'fourier-transform'

export const LOGMEL = Object.freeze({ kind: 'logmel', sampleRate: 44100, n: 2048, hop: 512, bands: 96, fmin: 20, fmax: 11025, frames: 128, top: 80 })

const hann = n => Float64Array.from({ length: n }, (_, i) => 0.5 - 0.5 * Math.cos(2 * Math.PI * i / n))

// Slaney's Auditory Toolbox mel scale, librosa.hz_to_mel(htk=False): linear below 1 kHz, log above.
const F_SP = 200 / 3, MIN_LOG_HZ = 1000, MIN_LOG_MEL = MIN_LOG_HZ / F_SP, LOGSTEP = Math.log(6.4) / 27
const hzToMel = f => f < MIN_LOG_HZ ? f / F_SP : MIN_LOG_MEL + Math.log(f / MIN_LOG_HZ) / LOGSTEP
const melToHz = m => m < MIN_LOG_MEL ? m * F_SP : MIN_LOG_HZ * Math.exp(LOGSTEP * (m - MIN_LOG_MEL))

// Triangular filters on the FFT bin frequencies, each scaled to unit area (norm='slaney').
function melBank({ sampleRate: fs, n, bands, fmin, fmax }) {
	let lo = hzToMel(fmin), hi = hzToMel(fmax)
	let edges = Float64Array.from({ length: bands + 2 }, (_, i) => melToHz(lo + (hi - lo) * i / (bands + 1)))
	return Array.from({ length: bands }, (_, b) => {
		let [l, c, r] = [edges[b], edges[b + 1], edges[b + 2]], norm = 2 / (r - l)
		return Float64Array.from({ length: n / 2 + 1 }, (_, k) => { let f = k * fs / n; return Math.max(0, Math.min((f - l) / (c - l), (r - f) / (r - c))) * norm })
	})
}

let bank = null, window = null
export function logmel(x, f = LOGMEL) {
	let { n, hop, bands, frames, top } = f, frame = new Float64Array(n)
	bank ??= melBank(f); window ??= hann(n)
	let out = new Float32Array(bands * frames), max = 0
	for (let t = 0; t < frames; t++) {
		for (let i = 0; i < n; i++) frame[i] = (x[t * hop + i] || 0) * window[i]
		let [re, im] = fft(frame)
		for (let b = 0; b < bands; b++) {
			let w = bank[b], s = 0
			for (let k = 0; k <= n / 2; k++) if (w[k]) s += w[k] * (re[k] * re[k] + im[k] * im[k])
			out[b * frames + t] = s
			if (s > max) max = s
		}
	}
	let ref = Math.max(max, 1e-10)
	for (let i = 0; i < out.length; i++) { let db = 10 * Math.log10(Math.max(out[i], 1e-10) / ref); out[i] = Math.max(db, -top) / (top / 2) + 1 }
	return out
}
