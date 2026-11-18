// The contract atom's tail when `limit` moves at the input's stop (README, After the input stops). A host renders
// latency + tail frames after the input, the tail read from the values in force when it asks, at the end (contract 1.0,
// Latency, tail); the share of the input glides to a new value over 20 ms, and until it arrives RNNoise's part of the
// output, (1 − share) of it, rings on. So a move toward 0, whose tail is none, shortly before the stop or after it
// leaves output over 60 dB under the input's RMS past what the host renders. Inputs: test.js's noisy speech, a 16.3 Hz
// tone of 0.002 (the tail's loudest) at four stop phases and a DC step at −46 dBFS, each stopping at 0.6 s, music
// 'enhance', calls of 128, the value jumping d ms from the stop (d < 0: before it). Printed per rate and move: for each d,
// the longest stretch over that level past latency + max(tail, 128 frames), ms, over the inputs (0: none); then the same
// with the tail read at the larger of the two values, which covers the ring, (1 − share) falling along a glide toward 0.
//
//   node scripts/limit-stop.mjs

import { rnnoise } from '../audio.js'
import { inputs } from './rnnoise-reference.mjs'

const BLOCK = 128, W = 64, D = [-5, 0, 1, 5, 10, 25]
const noisy = inputs()['lena+noise'].map(v => v / 32768)
const rms = (a, from, to) => { let s = 0; for (let i = from; i < to; i++) s += a[i] * a[i]; return Math.sqrt(s / (to - from)) }

// the atom from `from` dB, the value `to` written at frame `at` (the call split there), calls of BLOCK, the input, the
// latency's frames and `more` frames of silence; its output aligned with the input
function render(x, sr, from, to, at, more) {
	let params = { limit: Float32Array.of(from), music: 'enhance' }, ctx = { sampleRate: sr, maxBlockSize: BLOCK, maxChannels: 1, params: { limit: Float32Array.of(from), music: 'enhance' } }
	let L = rnnoise.latency(ctx), p = rnnoise(ctx), n = x.length + L + more, inp = new Float32Array(n), out = new Float32Array(n)
	inp.set(x)
	for (let i = 0, k; i < n; i += k) {
		if (i === at) params.limit[0] = to
		k = Math.min(BLOCK, n - i, i < at ? at - i : Infinity)
		p([[inp.subarray(i, i + k)]], [[out.subarray(i, i + k)]], params)
	}
	return out.subarray(L)
}

for (let sr of [44100, 48000]) {
	let S = Math.round(.6 * sr), more = Math.round(.3 * sr)
	let sigs = [...[0, 4, 8, 12].map(q => Float32Array.from({ length: S }, (_, i) => .002 * Math.sin(2 * Math.PI * 16.3 * (i - S) / sr + Math.PI * q / 8))), new Float32Array(S).fill(.005), noisy.subarray(0, S)]
	// frames a host renders past the input for a tail read at `limit`
	const waits = limit => Math.max(BLOCK, Math.ceil(rnnoise.tail({ sampleRate: sr, params: { limit: Float32Array.of(limit), music: 'enhance' } }) * sr))
	for (let [from, to] of [[16, 0], [100, 0], [16, .01], [0, 16]]) {
		let wait = waits(to), larger = waits(Math.max(from, to)), worst = 0
		let row = D.map(d => {
			let short = 0
			for (let x of sigs) {
				let y = render(x, sr, from, to, S + Math.round(d * sr / 1000), more), floor = 1e-3 * rms(x, 0, S), last = S
				for (let i = S; i + W <= y.length; i += W) if (rms(y, i, i + W) > floor) last = i + W
				short = Math.max(short, last - S - wait); worst = Math.max(worst, last - S - larger)
			}
			return `${d} ms ${(short / sr * 1000).toFixed(1)}`
		})
		console.log(`${sr / 1000} kHz, limit ${from} → ${to} (the host renders ${(wait / sr * 1000).toFixed(1)} ms): ${row.join(', ')}; with the tail read at the larger value, ${(larger / sr * 1000).toFixed(1)} ms: ${(Math.max(0, worst) / sr * 1000).toFixed(1)} at most`)
	}
}
