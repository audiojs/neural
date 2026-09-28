// Feature shards for training: `node scripts/factory.js synth <name> <clips> <seed>` renders labelled
// clips (scripts/data.js); `node scripts/factory.js real <name> <list> <seed>` reads unlabelled
// recordings (one path per line) as pairs of views, the recording and a degraded copy, labelled by
// @audio/pitch-pyin's track on the recording (a teacher, used only where it is sure).
//
// Per worker w: <name>-<w>.x.f16 (and .z.f16, the degraded view, for real), frames × SPAN dB values
// (features.js, bins −MARGIN … BINS + MARGIN), and <name>-<w>.y.f32, frames × 4: f0 (Hz, 0 unvoiced),
// weight (0: no pitch label), source index, sample rate. Output directory: $NP_DATA, default
// ~/.cache/audiojs/data/neural-pitch. Workers: $NP_WORKERS, default 4.

import { Worker, isMainThread, parentPort, workerData } from 'node:worker_threads'
import { openSync, writeSync, closeSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { analyzer, BINS, MARGIN } from '../features.js'
import { clip, degrade, rng, SOURCE_NAMES } from './data.js'
import { readWav } from './wav.js'
import resample from '@audio/resample-sinc'
import { track } from '@audio/pitch-pyin'

export const SPAN = BINS + 2 * MARGIN
const OUT = process.env.NP_DATA || path.join(os.homedir(), '.cache', 'audiojs', 'data', 'neural-pitch')
const WORKERS = +(process.env.NP_WORKERS || 4)
const KINDS = [...SOURCE_NAMES, 'none', 'real']

function writer(file) {
	let fd = openSync(file, 'w')
	return { put: a => writeSync(fd, new Uint8Array(a.buffer, a.byteOffset, a.byteLength)), close: () => closeSync(fd) }
}

// dB spectra of x at the given centre samples: frames × SPAN, fp16
function spectra(x, fs, centres) {
	let a = analyzer(fs, -MARGIN, SPAN), out = new Float16Array(centres.length * SPAN), db = new Float64Array(SPAN)
	centres.forEach((c, j) => { a.db(x, c, db); out.set(db, j * SPAN) })
	return out
}

async function synth({ name, clips, seed, w }) {
	let X = writer(path.join(OUT, `${name}-${w}.x.f16`)), Y = writer(path.join(OUT, `${name}-${w}.y.f32`)), frames = 0
	for (let i = w; i < clips; i += WORKERS) {
		let c = await clip(seed + i)
		let y = new Float32Array(c.frames.length * 4)
		for (let j = 0; j < c.frames.length; j++) y.set([c.f0[j], c.weight[j], KINDS.indexOf(c.kind), c.fs], 4 * j)
		X.put(spectra(c.x, c.fs, c.frames)); Y.put(y)
		frames += c.frames.length
		if ((i / WORKERS | 0) % 50 === 0) parentPort.postMessage({ w, i, frames })
	}
	X.close(); Y.close()
	return frames
}

// A recording as two views. Teacher labels: pYIN's track on the recording, voiced frames where its
// voiced probability is at least 0.5 and the pitch holds within 50 cents over ±2 frames; unvoiced where
// it decodes unvoiced; weight 0 in between.
async function real({ name, files, seed, w }) {
	let X = writer(path.join(OUT, `${name}-${w}.x.f16`)), Z = writer(path.join(OUT, `${name}-${w}.z.f16`)), Y = writer(path.join(OUT, `${name}-${w}.y.f32`)), frames = 0
	for (let i = w; i < files.length; i += WORKERS) {
		let r = rng(seed + i), { data, fs } = readWav(files[i])
		let t = track(data, { fs }), hop = Math.round(fs * 256 / 44100)   // pitch-pyin's default hop
		let H = Math.round(0.025 * fs), centres = []
		for (let c = r.int(0, H - 1); c < data.length; c += H) centres.push(c)
		let y = new Float32Array(centres.length * 4)
		centres.forEach((c, j) => {
			let k = Math.round(c / hop), f = t.f0[k] || 0, sure = t.prob[k] >= 0.5
			for (let d = -2; d <= 2 && f > 0; d++) { let g = t.f0[k + d] || 0; if (!(g > 0) || Math.abs(1200 * Math.log2(g / f)) > 50) sure = false }
			y.set([f, f > 0 ? +sure : 1, KINDS.indexOf('real'), fs], 4 * j)
		})
		// the degraded view, at another rate sometimes: centres map by time
		let fz = +r.weighted({ [fs]: 5, 16000: 2, 22050: 1, 48000: 1 })
		let z = fz === fs ? Float32Array.from(data) : resample(Float32Array.from(data), { from: fs, to: fz })
		z = await degrade(r, z, fz)
		X.put(spectra(data, fs, centres)); Z.put(spectra(z, fz, centres.map(c => Math.round(c * fz / fs)))); Y.put(y)
		frames += centres.length
		if ((i / WORKERS | 0) % 20 === 0) parentPort.postMessage({ w, i, frames })
	}
	X.close(); Z.close(); Y.close()
	return frames
}

if (isMainThread) {
	let [mode, name, arg, seed = '1'] = process.argv.slice(2)
	if (!['synth', 'real'].includes(mode) || !name || !arg) {
		console.error('usage: node scripts/factory.js synth <name> <clips> <seed> | real <name> <list-file> <seed>')
		process.exit(1)
	}
	mkdirSync(OUT, { recursive: true })
	let job = mode === 'synth' ? { mode, name, clips: +arg, seed: +seed } : { mode, name, files: readFileSync(arg, 'utf8').split('\n').filter(Boolean), seed: +seed }
	let t0 = Date.now()
	let counts = await Promise.all(Array.from({ length: WORKERS }, (_, w) => new Promise((ok, fail) => {
		let wk = new Worker(new URL(import.meta.url), { workerData: { ...job, w } })
		wk.on('message', m => m.done ? ok(m.frames) : console.log(JSON.stringify(m)))
		wk.on('error', fail)
		wk.on('exit', c => c && fail(new Error('worker exit ' + c)))
	})))
	let meta = { mode, name, shards: WORKERS, frames: counts, span: SPAN, margin: MARGIN, bins: BINS, kinds: KINDS, seed: +seed, seconds: (Date.now() - t0) / 1000, ...(mode === 'synth' ? { clips: +arg } : { files: job.files.length }) }
	writeFileSync(path.join(OUT, `${name}.json`), JSON.stringify(meta, null, 1))
	console.log(JSON.stringify(meta))
} else {
	let frames = await (workerData.mode === 'synth' ? synth : real)(workerData)
	parentPort.postMessage({ done: true, frames })
}
