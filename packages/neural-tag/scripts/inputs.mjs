// The waveforms test.js and scripts/export.py give YAMNet, mono float32 at 16 kHz: lena (audio-lena: speech, 44.1 kHz)
// resampled by @audio/resample-sinc as tag() resamples it; plucked strings (Karplus–Strong, a seeded LCG's noise); a
// swept tone; white noise; a 0.5 s beep, shorter than a patch; 2 s of digital silence; nothing. export.py runs this file
// and gives the Keras model those very bytes; test.js checks their SHA-256 against fixtures/reference.json first.
//
//   node scripts/inputs.mjs <dir>       writes <dir>/<name>.f32 for each

import { writeFileSync, mkdirSync } from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import lena from 'audio-lena/raw'
import resample from '@audio/resample-sinc'

const lcg = (s = 1) => () => (s = (s * 1664525 + 1013904223) % 4294967296, s / 4294967296 - .5)

// eight notes of an A minor arpeggio, 0.375 s apart, each left ringing: a string's period of noise, each sample
// replaced by 0.996 of the mean of it and the next as it is read
export function pluck(sr = 16000, seconds = 3) {
	let rnd = lcg(7), y = new Float64Array(Math.round(sr * seconds)), notes = [220, 261.63, 329.63, 440, 329.63, 261.63, 220, 164.81]
	notes.forEach((f, k) => {
		let p = Math.round(sr / f), buf = Float64Array.from({ length: p }, rnd)
		for (let n = Math.round(k * seconds / notes.length * sr), i = 0; n < y.length; n++, i = (i + 1) % p) {
			let v = buf[i]
			y[n] += v
			buf[i] = 0.996 * 0.5 * (v + buf[(i + 1) % p])
		}
	})
	let peak = y.reduce((m, v) => Math.max(m, Math.abs(v)), 0)
	return Float32Array.from(y, v => 0.5 * v / peak)
}

// a tone swept 600 to 1200 Hz and back every 2 s (a wail), with its third harmonic; the same continuous signal at any rate
export function sweep(sr = 16000, seconds = 3) {
	return Float32Array.from({ length: Math.round(sr * seconds) }, (_, n) => {
		let t = n / sr, phi = 2 * Math.PI * (900 * t - 300 / Math.PI * Math.cos(Math.PI * t))
		return 0.3 * Math.sin(phi) + 0.1 * Math.sin(3 * phi)
	})
}

export function noise(sr = 16000, seconds = 3) {
	let rnd = lcg(3)
	return Float32Array.from({ length: Math.round(sr * seconds) }, () => 0.2 * rnd())
}

// 1 kHz for 0.5 s, 10 ms raised-cosine ends
export function beep(sr = 16000, seconds = 0.5) {
	let n = Math.round(sr * seconds), r = Math.round(0.01 * sr)
	return Float32Array.from({ length: n }, (_, i) => {
		let g = i < r ? 0.5 - 0.5 * Math.cos(Math.PI * i / r) : i >= n - r ? 0.5 - 0.5 * Math.cos(Math.PI * (n - 1 - i) / r) : 1
		return 0.3 * g * Math.sin(2 * Math.PI * 1000 * i / sr)
	})
}

export const LENA_RATE = 44100
export const lena44 = () => new Float32Array(lena)

export function inputs() {
	return {
		lena: resample(lena44(), { from: LENA_RATE, to: 16000 }),
		pluck: pluck(), sweep: sweep(), noise: noise(), beep: beep(),
		silence: new Float32Array(32000), empty: new Float32Array(0),
	}
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
	let dir = process.argv[2]
	if (!dir) { console.error('usage: node scripts/inputs.mjs <dir>'); process.exit(1) }
	mkdirSync(dir, { recursive: true })
	for (let [name, x] of Object.entries(inputs())) writeFileSync(path.join(dir, `${name}.f32`), new Uint8Array(x.buffer, x.byteOffset, x.byteLength))
}
