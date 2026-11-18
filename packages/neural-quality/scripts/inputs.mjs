// The waveforms test.js and scripts/reference.py give DNSMOS, mono float32: lena (audio-lena: speech from a film scene,
// 44.1 kHz) as it is, and at 16 and 8 kHz through resampy.js (at 16 kHz the very samples the reference makes of it);
// lena under white noise (a seeded LCG) 5 dB down at 16 kHz; its first 0.1 s; 2 s of digital silence; and ramps
// (sample i = i · 2^-21, exact in float32) whose values tell reference.py which samples each of the reference's windows
// holds. reference.py runs this file and gives the reference those very bytes; test.js checks their SHA-256 against
// fixtures/reference.json first.
//
//   node scripts/inputs.mjs <dir>       writes <dir>/<name>.f32 for each, and <dir>/rates.json

import { writeFileSync, mkdirSync } from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import lena from 'audio-lena/raw'
import resampy from '../resampy.js'

export const LENA_RATE = 44100
const lcg = (s = 1) => () => (s = (s * 1664525 + 1013904223) % 4294967296, s / 4294967296 - .5)
const rms = x => Math.sqrt(x.reduce((m, v) => m + v * v, 0) / x.length)

export const lena44 = () => new Float32Array(lena)
export const lena16 = () => Float32Array.from(resampy(lena44(), LENA_RATE, 16000))

// lena at 16 kHz plus white noise at `snr` dB under its RMS
export function noisy(snr = 5) {
	let x = lena16(), rnd = lcg(5), w = Float64Array.from(x, () => rnd()), g = rms(x) / rms(w) * 10 ** (-snr / 20)
	return Float32Array.from(x, (v, i) => v + g * w[i])
}

// the lengths whose windows the fixtures hold (samples at 16 kHz): around the window (144160) and a doubling, and
// clips long enough to reach the hops the reference's float product drops (7 to 23, 119 to 122)
export const RAMPS = [1600, 32000, 144159, 144160, 160159, 160160, 16000 * 17, 16000 * 30, 16000 * 34, 16000 * 132]
export const ramp = n => Float32Array.from({ length: n }, (_, i) => i * 2 ** -21)

/** name → [samples, sample rate] */
export function inputs() {
	let l16 = lena16()
	return {
		'lena@16k': [l16, 16000],
		'lena-white5@16k': [noisy(5), 16000],
		'lena@44.1k': [lena44(), LENA_RATE],
		'lena@8k': [Float32Array.from(resampy(lena44(), LENA_RATE, 8000)), 8000],
		'lena-0.1s@16k': [l16.slice(0, 1600), 16000],
		'silence@16k': [new Float32Array(32000), 16000],
		...Object.fromEntries(RAMPS.map(n => [`ramp-${n}`, [ramp(n), 16000]])),
	}
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
	let dir = process.argv[2]
	if (!dir) { console.error('usage: node scripts/inputs.mjs <dir>'); process.exit(1) }
	mkdirSync(dir, { recursive: true })
	let rates = {}
	for (let [name, [x, sr]] of Object.entries(inputs())) {
		writeFileSync(path.join(dir, `${name}.f32`), new Uint8Array(x.buffer, x.byteOffset, x.byteLength))
		rates[name] = sr
	}
	writeFileSync(path.join(dir, 'rates.json'), JSON.stringify(rates))
}
