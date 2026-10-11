#!/usr/bin/env node
// Build a DX7 catalog: every chosen voice rendered at the reference notes (scripts/recipe.js refs),
// each note embedded by the exported encoder through @audio/neural-runtime, the mean of the audible
// ones kept as the voice's vector (a note below 4 int16 steps, which some key scalings give, is
// skipped, as scripts/catalogs.py skips it). Names, cartridges and vectors only: no voice data
// leaves the sysex.
//
//   node scripts/catalog.js <voices> [--model file] [--out file] [--encoding int8-base64|int4-base64]
//
//   voices   factory (Yamaha ROM1A-4B and VRC-101-112), heldout, dev, train or all (every audible voice)
//   model    default ~/.cache/audiojs/neural/timbre/encoder.int8.onnx
//   out      default ~/.cache/audiojs/neural/timbre/catalogs/dx7-<voices>.json
//
// Four worker threads render and prepare notes; the main thread runs the encoder in batches.
import { Worker, isMainThread, parentPort, workerData } from 'node:worker_threads'
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs'
import path from 'node:path'
import os from 'node:os'
import { cqt, onset } from '../features.js'
import { refs, note } from './recipe.js'

const DATA = process.env.TIMBRE_DATA ?? path.join(os.homedir(), '.cache/audiojs/data/dx7')
const CACHE = path.join(process.env.AUDIO_NEURAL_CACHE ?? path.join(os.homedir(), '.cache/audiojs/neural'), 'timbre')
const WORKERS = 4, CHUNK = 16

const SOURCES = { factory: v => v.source === 'yamaha-factory' || v.source === 'yamaha-vrc' }
const pick = which => SOURCES[which] ?? (which === 'all' ? () => true : v => v.split === which)

// 'factory/rom1a.syx' → 'ROM1A'; other cartridges keep their path under the source folder
const cartridge = file => file.startsWith('factory/') || file.startsWith('vrc/') ? path.basename(file, path.extname(file)).toUpperCase() : file.replace(/^[^/]+\//, '')

async function main() {
	let [which, ...rest] = process.argv.slice(2), o = {}
	for (let i = 0; i < rest.length; i += 2) o[rest[i].replace(/^--/, '')] = rest[i + 1]
	let { createEncoder, catalog } = await import('../timbre.js')
	let voices = JSON.parse(readFileSync(path.join(DATA, 'voices.json'), 'utf8')).voices
	let chosen = voices.map((v, i) => [v, i]).filter(([v]) => !v.silent && pick(which)(v))
	let encoder = await createEncoder({ model: o.model ?? path.join(CACHE, 'encoder.int8.onnx') })
	let vectors = new Array(chosen.length), t0 = performance.now(), done = 0
	let todo = Array.from({ length: Math.ceil(chosen.length / CHUNK) }, (_, c) => c)
	let pending = Promise.resolve()
	await Promise.all(Array.from({ length: Math.min(WORKERS, todo.length) }, () => new Promise((resolve, reject) => {
		let w = new Worker(new URL(import.meta.url), { workerData: { data: DATA } })
		let next = () => todo.length ? (c => w.postMessage({ c, voices: chosen.slice(c * CHUNK, (c + 1) * CHUNK).map(([, i]) => i) }))(todo.shift()) : w.postMessage(null)
		w.on('message', ({ c, patches, owners }) => {
			// each voice's audible reference notes → one mean vector; runs are serialized on the session
			pending = pending.then(async () => {
				let e = patches.length ? await encoder.project(patches) : []
				e.forEach((v, j) => {
					let k = c * CHUNK + owners[j], mean = vectors[k] ??= new Float32Array(v.length)
					for (let d = 0; d < v.length; d++) mean[d] += v[d]
				})
				done += Math.min(CHUNK, chosen.length - c * CHUNK)
				if (done % 1024 < CHUNK) console.log(`${done}/${chosen.length} voices, ${(done / ((performance.now() - t0) / 1000)).toFixed(1)}/s`)
			})
			next()
		})
		w.on('error', reject); w.on('exit', resolve); next()
	})))
	await pending
	let { unit } = await import('../timbre.js')
	let kept = chosen.map((_, k) => k).filter(k => vectors[k])
	let entries = kept.map(k => chosen[k][0]).map(v => ({ name: v.name, cartridge: cartridge(v.cartridge), index: v.index + 1, source: v.source, algorithm: v.algorithm, id: v.id }))
	let json = catalog(entries, kept.map(k => unit(vectors[k])), encoder.binding, {
		encoding: o.encoding ?? 'int8-base64',
		name: `DX7 ${which}`, synth: 'Yamaha DX7 (@audio/synth-dx7, MSFA)',
		references: `MIDI notes ${refs(0).map(r => r.note).filter((n, i, a) => a.indexOf(n) === i).join(', ')} at velocities 64 and 112, held 1 s, 44.1 kHz`,
		sources: 'names, cartridges and vectors only; each source and its stated terms: the @audio/neural-timbre README, Catalogs',
	})
	let out = o.out ?? path.join(CACHE, 'catalogs', `dx7-${which}.json`)
	mkdirSync(path.dirname(out), { recursive: true })
	writeFileSync(out, JSON.stringify(json))
	encoder.free()
	console.log(`${out}: ${entries.length} voices in ${((performance.now() - t0) / 60000).toFixed(1)} min`)
}

function worker() {
	let patches = new Uint8Array(readFileSync(path.join(workerData.data, 'patches.u8')))
	parentPort.on('message', msg => {
		if (!msg) return parentPort.close()
		let out = [], owners = []
		msg.voices.forEach((v, k) => {
			for (let r of refs(v)) {
				let x = note(patches.subarray(v * 155, v * 155 + 155), r), peak = 0
				for (let s of x) peak = Math.max(peak, Math.abs(s))
				if (peak < 2 ** -13) continue
				out.push(cqt(x.subarray(onset(x)))); owners.push(k)
			}
		})
		parentPort.postMessage({ c: msg.c, patches: out, owners }, out.map(p => p.buffer))
	})
}

if (isMainThread) main().catch(e => { console.error(e); process.exit(1) })
else worker()
