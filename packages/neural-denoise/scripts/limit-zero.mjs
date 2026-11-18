// What `limit: 0` gives at each host rate (README, The limit's scale): the contract atom at 0, music 'enhance', 2 s of
// each input in calls of 128, against the input, the outer quarter seconds left out. At 48 kHz the output is the input
// itself; at other rates it is the input as resample-sinc carries it to 48 kHz and back. Printed per rate: each tone's
// level against its input's (dB; a tone over 24 kHz comes back as its alias under it and its image over it, both
// counted), at fractions of the band (of 24 kHz at 88.2 and 96 kHz) and there at 24.5, 30 and 40 kHz; white noise's and
// test.js's noisy speech's SNR from the input (dB).
//
//   node scripts/limit-zero.mjs

import { rnnoise } from '../audio.js'
import { inputs } from './rnnoise-reference.mjs'

const BLOCK = 128

// the atom as a host runs it: built at limit 0, calls of BLOCK, then the latency's frames of silence; its output
// aligned with the input
function render(x, sr) {
	let params = { limit: Float32Array.of(0), music: 'enhance' }, ctx = { sampleRate: sr, maxBlockSize: BLOCK, maxChannels: 1, params }
	let L = rnnoise.latency(ctx), p = rnnoise(ctx), n = x.length + L, inp = new Float32Array(n), out = new Float32Array(n)
	inp.set(x)
	for (let i = 0; i < n; i += BLOCK) p([[inp.subarray(i, i + BLOCK)]], [[out.subarray(i, i + BLOCK)]], params)
	return out.subarray(L)
}

const rms = (a, from, to) => { let s = 0; for (let i = from; i < to; i++) s += a[i] * a[i]; return Math.sqrt(s / (to - from)) }
const snr = (x, y, from, to) => { let s = 0, e = 0; for (let i = from; i < to; i++) { s += x[i] ** 2; e += (x[i] - y[i]) ** 2 } return e ? 10 * Math.log10(s / e) : Infinity }
const noisy = inputs()['lena+noise'].map(v => v / 32768)

for (let sr of [8000, 16000, 22050, 44100, 48000, 88200, 96000]) {
	// fractions of the band, of 48 kHz's above it, where the mix is made; there also 24.5, 30 and 40 kHz
	let n = 2 * sr, a = Math.round(.25 * sr), b = n - a, hz = [.5, .8, .9, .95, .98].map(f => f * Math.min(sr, 48000) / 2), tones = []
	if (sr > 48000) hz.push(24500, 30000, 40000)
	for (let f of hz) {
		let x = Float32Array.from({ length: n }, (_, i) => .25 * Math.sin(2 * Math.PI * f * i / sr))
		tones.push(`${Math.round(f)} Hz ${(20 * Math.log10(rms(render(x, sr), a, b) / rms(x, a, b))).toFixed(1)}`)
	}
	let s = 9, white = Float32Array.from({ length: n }, () => .3 * ((s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 2 ** 32 - .5))
	let speech = Float32Array.from({ length: n }, (_, i) => noisy[i % noisy.length])
	let row = [white, speech].map(x => snr(x, render(x, sr), a, b)).map(v => v === Infinity ? 'the input itself' : `${v.toFixed(1)} dB`)
	console.log(`${sr / 1000} kHz: ${tones.join(', ')}; SNR from the input: white noise ${row[0]}, noisy speech ${row[1]}`)
}
