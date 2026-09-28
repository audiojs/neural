// Frame-wise pitch posterior and voicing from a small transposition-equivariant network (model.js),
// and a stage 1 for @audio/pitch-pyin: `candidates` turns each frame's posterior into pYIN's
// candidate list, so its pitch HMM, Viterbi decoding and Tony's note model run unchanged on it.

import { analyzer, normalize, hz, LOW, PER_SEMITONE, BINS } from './features.js'
import { network, CONFIG } from './model.js'

export { analyzer, network, CONFIG }

const PEAKS = 5, SPREAD = 2   // candidates kept a frame; bins either side of a peak that belong to it
const KAPPA = 1               // weight of the network's voicing log-odds in pYIN's HMM

/** Pitch posterior: softmax over the logits, in place. */
function softmax(y) {
	let max = -Infinity, sum = 0
	for (let j = 0; j < y.length; j++) if (y[j] > max) max = y[j]
	for (let j = 0; j < y.length; j++) sum += y[j] = Math.exp(y[j] - max)
	for (let j = 0; j < y.length; j++) y[j] /= sum
}

/** Fractional bin of the mass within ±SPREAD bins of j, and that mass. */
function centroid(p, j) {
	let m = 0, s = 0
	for (let i = Math.max(0, j - SPREAD); i <= Math.min(p.length - 1, j + SPREAD); i++) { m += p[i]; s += i * p[i] }
	return [s / m, m]
}

const binHz = b => hz(LOW + b / PER_SEMITONE)

/**
 * Analyzer of one frame at sample rate fs: `frame(x, centre)` → { posterior (Float32Array(BINS), bin k at
 * MIDI 23 + k/3), voicing (0…1), level (dBFS) } for the spectrum around x[centre]. Buffers are reused.
 */
export function frame(fs) {
	let a = analyzer(fs), db = new Float64Array(BINS), input = new Float32Array(BINS), net = network()
	return (x, centre) => {
		a.db(x, centre, db)
		let level = normalize(db, input)
		let { logits, voicing } = net(input, level)
		softmax(logits)
		return { posterior: logits, voicing, level }
	}
}

/**
 * Frame-wise f0 without a pitch HMM: `pitch(samples, { fs, hopSize })` → { times, f0, voicing },
 * frame i at i·hopSize/fs (hopSize default ≈ 5.8 ms, as @audio/pitch-pyin), f0 the posterior's peak
 * refined over ±2 bins (every frame has one), voicing the network's probability.
 */
export default function pitch(data, { fs = 44100, hopSize } = {}) {
	hopSize ??= Math.round(fs * 256 / 44100)
	let run = frame(fs), n = Math.floor(data.length / hopSize) + 1
	let times = new Float64Array(n), f0 = new Float32Array(n), voicing = new Float32Array(n)
	for (let i = 0; i < n; i++) {
		let { posterior, voicing: v } = run(data, i * hopSize), j = 0
		for (let k = 1; k < BINS; k++) if (posterior[k] > posterior[j]) j = k
		times[i] = i * hopSize / fs; f0[i] = binHz(centroid(posterior, j)[0]); voicing[i] = v
	}
	return { times, f0, voicing }
}

/**
 * Stage 1 for @audio/pitch-pyin, its `candidates` contract: `candidates(N, fs, minFreq, maxFreq)` →
 * `run(frame)` → count, filling run.freq and run.prob (the posterior's peaks in range, strongest first,
 * each with its mass within ±2 bins, times the voicing weight) and run.rms (the level of the N/2 samples
 * centred on the frame, as pYIN's YIN window, which Tony's note onsets read).
 * run.frameSize and run.lead: the samples it reads, centred on the frame's time (the FFT window).
 *
 * Voicing weight: pYIN's HMM gives a voiced state half its candidate's probability p and each of its n
 * unvoiced states (1 − p/2)/n, so a frame decodes voiced once p exceeds about 2/n: YIN's candidate mass
 * is near 0 on noise, a network's voicing probability v is not. The weight (v / (1 − v))^κ · 2/(n + 1),
 * at most 1, makes that ratio the network's voicing odds (to the power κ), so the HMM adds up the
 * network's log-odds frame by frame against its switching cost.
 *
 *   track(samples, { fs, candidates })     notes(samples, { fs, candidates })
 */
export function candidates(N, fs, minFreq, maxFreq, { kappa = KAPPA } = {}) {
	let run = frame(fs), W = N >> 1, size = Math.max(analyzer(fs).n, W), lead = size >> 1
	let lo = minFreq, hi = maxFreq, used = new Uint8Array(BINS)
	let n = Math.floor(120 * Math.log2(maxFreq / minFreq)) + 1           // pYIN's 0.1-semitone states
	function analyze(frame) {
		let e = 0
		for (let i = lead - (W >> 1); i < lead - (W >> 1) + W; i++) e += frame[i] * frame[i]
		analyze.rms = Math.sqrt(e / W)
		let { posterior: p, voicing } = run(frame, lead), m = 0, max = 0
		let v = Math.min(1 - 1e-9, Math.max(1e-9, voicing)), weight = Math.min(1, (v / (1 - v)) ** kappa * 2 / (n + 1))
		for (let j = 0; j < BINS; j++) if (p[j] > max) max = p[j]
		used.fill(0)
		// peaks by height; each claims ±SPREAD bins (no mass counted twice)
		while (m < PEAKS) {
			let j = -1
			for (let k = 0; k < BINS; k++) if (!used[k] && (j < 0 || p[k] > p[j])) j = k
			if (j < 0 || p[j] < max * 0.01) break
			let ok = (j === 0 || p[j] >= p[j - 1]) && (j === BINS - 1 || p[j] >= p[j + 1])
			for (let i = Math.max(0, j - SPREAD); i <= Math.min(BINS - 1, j + SPREAD); i++) used[i] = 1
			if (!ok) continue
			let [b, mass] = centroid(p, j), f = binHz(b)
			if (f < lo || f > hi) continue
			analyze.freq[m] = f; analyze.prob[m++] = mass * weight
		}
		return m
	}
	analyze.freq = new Float64Array(PEAKS)
	analyze.prob = new Float64Array(PEAKS)
	analyze.rms = 0
	analyze.frameSize = size
	analyze.lead = lead
	return analyze
}
