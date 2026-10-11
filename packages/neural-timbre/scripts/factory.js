#!/usr/bin/env node
// Feature sets for training and the benchmark: every row of a set (scripts/recipe.js) rendered by
// synth-dx7, augmented by scripts/augment.js and prepared by the package's own frontend, stored
// as float16 [rows, bands, frames] on disk for memory mapping.
//
//   node scripts/factory.js <set> [--name NAME] [--limit N] [--views N] [--kinds logmel,cqt] [--workers N] [--data dir]
//
//   set      train (views of training voices), dev (references and clean + chain queries of
//            development voices), heldout (references and every benchmark condition), refs
//            (references of training voices, for the catalogs scripts/catalogs.py builds)
//   writes   <data>/features/<name>.json (rows and feature settings; name defaults to the set),
//            <name>.<kind>.f16 and <name>.stats.f32 (the dry note's peak and RMS a row); chunks done
//            so far are kept in <name>.done, so a stopped run resumes
//
// Needs @audio/synth-dx7 and the augmentation packages resolvable from here (npm link or a local
// node_modules); it runs at most --workers threads (default 4).
import { Worker, isMainThread, parentPort, workerData } from 'node:worker_threads'
import { readFileSync, writeFileSync, openSync, writeSync, closeSync, existsSync, mkdirSync, ftruncateSync } from 'node:fs'
import path from 'node:path'
import os from 'node:os'
import { FEATURES, cqt, onset } from '../features.js'
import { LOGMEL, logmel } from './logmel.js'
import { views, refs, queries, specOf, note, mixNote, seedOf } from './recipe.js'
import { apply, CONDITIONS } from './augment.js'

const KINDS = { cqt: [FEATURES, cqt], logmel: [LOGMEL, logmel] } // logmel: the feature pilot only
const CHUNK = 64

function args(argv) {
	let o = { set: argv[0], limit: Infinity, views: 8, kinds: ['cqt'], workers: 4, data: path.join(os.homedir(), '.cache/audiojs/data/dx7') }
	for (let i = 1; i < argv.length; i += 2) {
		let [k, v] = [argv[i].replace(/^--/, ''), argv[i + 1]]
		o[k] = k === 'kinds' ? v.split(',') : k === 'data' || k === 'name' ? v : Number(v)
	}
	return o
}

function rowsOf(set, voices, o) {
	let pick = split => voices.map((v, i) => [v, i]).filter(([v]) => v.split === split && !v.silent)
	if (set === 'train') {
		// a fixed order by id hash, so --limit takes the same voices every time
		let train = pick('train').sort((a, b) => seedOf(a[0].id) - seedOf(b[0].id)).slice(0, o.limit)
		return { rows: train.flatMap(([v, i]) => views(i, v.id, o.views)), pool: 'train' }
	}
	if (set === 'refs') return { rows: pick('train').slice(0, o.limit).flatMap(([, i]) => refs(i)), pool: 'train' }
	let split = set === 'dev' ? 'dev' : 'heldout', conds = set === 'dev' ? ['clean', 'chain'] : Object.keys(CONDITIONS)
	let list = pick(split).slice(0, o.limit)
	return { rows: list.flatMap(([v, i]) => [...refs(i), ...queries(i, v.id, conds)]), pool: split }
}

async function main() {
	let o = args(process.argv.slice(2))
	if (!['train', 'dev', 'heldout', 'refs'].includes(o.set)) throw new Error('set: train, dev, heldout or refs')
	let voices = JSON.parse(readFileSync(path.join(o.data, 'voices.json'), 'utf8')).voices
	let { rows, pool } = rowsOf(o.set, voices, o)
	let out = path.join(o.data, 'features'); mkdirSync(out, { recursive: true })
	let base = path.join(out, o.name ?? o.set), meta = base + '.json'
	let settings = Object.fromEntries(o.kinds.map(k => [k, KINDS[k][0]]))
	let keys = ['voice', 'kind', 'note', 'velocity', 'hold', 'pre', 'cond', 'seed']
	// full precision: a row re-renders exactly from its JSON
	let table = { version: 1, set: o.set, pool, keys, features: settings, rows: rows.map(r => keys.map(k => r[k])) }
	if (existsSync(meta)) {
		let old = JSON.parse(readFileSync(meta, 'utf8'))
		if (JSON.stringify(old.rows) !== JSON.stringify(table.rows) || JSON.stringify(old.features) !== JSON.stringify(settings)) throw new Error(`${meta} holds another set: remove it and its .f16/.done to rebuild`)
		table = old
	} else writeFileSync(meta, JSON.stringify(table))
	let files = [...o.kinds.map(k => [`${base}.${k}.f16`, KINDS[k][0].bands * KINDS[k][0].frames * 2]), [base + '.stats.f32', 8]]
	for (let [f, size] of files) if (!existsSync(f)) { let fd = openSync(f, 'w'); ftruncateSync(fd, rows.length * size); closeSync(fd) }
	let doneFile = base + '.done', done = existsSync(doneFile) ? new Uint8Array(readFileSync(doneFile)) : new Uint8Array(Math.ceil(rows.length / CHUNK))
	let todo = [...done.keys()].filter(c => !done[c])
	console.log(`${o.name ?? o.set}: ${rows.length} rows, ${todo.length} of ${done.length} chunks to do, ${o.workers} workers, ${o.kinds}`)
	let t0 = performance.now(), finished = 0
	await Promise.all(Array.from({ length: Math.min(o.workers, todo.length) }, () => new Promise((resolve, reject) => {
		let w = new Worker(new URL(import.meta.url), { workerData: { data: o.data, base, kinds: o.kinds, pool } })
		let next = () => {
			if (!todo.length) return w.postMessage(null)
			let c = todo.shift()
			w.postMessage({ chunk: c, rows: rows.slice(c * CHUNK, (c + 1) * CHUNK), start: c * CHUNK })
		}
		w.on('message', ({ chunk }) => {
			done[chunk] = 1; finished++
			if (finished % 20 === 0) {
				writeFileSync(doneFile, done)
				let rate = finished * CHUNK / ((performance.now() - t0) / 1000)
				console.log(`${finished}/${done.length} chunks, ${rate.toFixed(0)} rows/s, ${((todo.length * CHUNK) / rate / 60).toFixed(1)} min left`)
			}
			next()
		})
		w.on('error', reject)
		w.on('exit', resolve)
		next()
	})))
	writeFileSync(doneFile, done)
	let stats = new Float32Array(readFileSync(base + '.stats.f32').buffer), silent = 0
	for (let i = 0; i < rows.length; i++) if (stats[2 * i] < 2 ** -13) silent++
	console.log(`done in ${((performance.now() - t0) / 60000).toFixed(1)} min; ${silent} silent rows`)
}

async function worker() {
	let { data, base, kinds, pool: split } = workerData
	let patches = new Uint8Array(readFileSync(path.join(data, 'patches.u8')))
	let voices = JSON.parse(readFileSync(path.join(data, 'voices.json'), 'utf8')).voices
	let pool = voices.map((v, i) => [v, i]).filter(([v]) => v.split === split).map(([, i]) => i)
	let patch = i => patches.subarray(i * 155, i * 155 + 155)
	let fds = kinds.map(k => openSync(`${base}.${k}.f16`, 'r+')), statsFd = openSync(base + '.stats.f32', 'r+')
	parentPort.on('message', async msg => {
		if (!msg) { [...fds, statsFd].forEach(closeSync); return parentPort.close() }
		for (let [j, row] of msg.rows.entries()) {
			let dry = note(patch(row.voice), row), peak = 0, sum = 0
			for (let x of dry) { peak = Math.max(peak, Math.abs(x)); sum += x * x }
			let spec = specOf(row, pool)
			let y = await apply(dry, spec, m => mixNote(patch(m.voice), m, row.pre))
			let x = y.subarray(onset(y))
			kinds.forEach((k, ki) => {
				let [f, fn] = KINDS[k], v = new Float16Array(fn(x, f)), size = f.bands * f.frames * 2
				writeSync(fds[ki], new Uint8Array(v.buffer), 0, size, (msg.start + j) * size)
			})
			writeSync(statsFd, new Uint8Array(Float32Array.of(peak, Math.sqrt(sum / dry.length)).buffer), 0, 8, (msg.start + j) * 8)
		}
		parentPort.postMessage({ chunk: msg.chunk })
	})
}

if (isMainThread) main().catch(e => { console.error(e); process.exit(1) })
else worker()
