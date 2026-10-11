#!/usr/bin/env node
// The rebuild half of the benchmark: for held-out queries, the spectral distance (neural-synth's
// specLoss) between the query note and the best retrieved preset played at the same note, before
// and after seeded refinement with @audio/neural-synth's match, against blind match at the same
// budget. The query is the clean held-out note; retrieval ranks it against dx7-heldout.
//
//   node scripts/rebuild.js [--n 64] [--budget 1000] [--model file] [--out file] [--misses file]
//
//   misses   a previous run's output: rerun only its retrieval misses (at another budget, say)
//
// match searches the 145 sound parameters of the VCED voice (MAX bounds; the 10 name bytes, which
// MAX also bounds, cannot change the sound and stay fixed), rounding each candidate to integers.
// Each CMA-ES start is evaluated before it steps (cmaes.js), so a seeded result is never worse than
// the retrieved preset (`kept`, the better of the two, equals `seeded`). `random` is another
// held-out voice at the same note: the scale of a wrong answer.
import { Worker, isMainThread, parentPort, workerData } from 'node:worker_threads'
import { readFileSync, writeFileSync, openSync, readSync, closeSync } from 'node:fs'
import path from 'node:path'
import os from 'node:os'
import { createEncoder, readCatalog, rank } from '../timbre.js'
import { note, seedOf } from './recipe.js'

const DATA = process.env.TIMBRE_DATA ?? path.join(os.homedir(), '.cache/audiojs/data/dx7')
const CACHE = path.join(process.env.AUDIO_NEURAL_CACHE ?? path.join(os.homedir(), '.cache/audiojs/neural'), 'timbre')
const WORKERS = 4

async function main() {
	let o = { n: 64, budget: 1000 }
	for (let a = process.argv.slice(2), i = 0; i < a.length; i += 2) o[a[i].replace(/^--/, '')] = isNaN(a[i + 1]) ? a[i + 1] : Number(a[i + 1])
	let meta = JSON.parse(readFileSync(path.join(DATA, 'features/heldout.json'), 'utf8'))
	let f = meta.features.cqt, size = f.bands * f.frames
	let rows = meta.rows.map(r => Object.fromEntries(meta.keys.map((k, i) => [k, r[i]])))
	let stats = new Float32Array(readFileSync(path.join(DATA, 'features/heldout.stats.f32')).buffer)
	let voices = JSON.parse(readFileSync(path.join(DATA, 'voices.json'), 'utf8')).voices
	// a fixed sample: clean queries ordered by a hash of their voice id
	let picked = rows.map((r, i) => [r, i]).filter(([r, i]) => r.kind === 'query' && r.cond === 'clean' && stats[2 * i] >= 2 ** -13)
		.sort((a, b) => seedOf(voices[a[0].voice].id, 'rebuild') - seedOf(voices[b[0].voice].id, 'rebuild')).slice(0, o.n)
	if (o.misses) {
		let miss = new Set(JSON.parse(readFileSync(o.misses, 'utf8')).results.filter(r => !r.hit).map(r => r.voice))
		picked = picked.filter(([r]) => miss.has(voices[r.voice].id))
	}
	let fd = openSync(path.join(DATA, 'features/heldout.cqt.f16'), 'r')
	let read = i => { let b = new Uint8Array(size * 2); readSync(fd, b, 0, b.length, i * size * 2); return Float32Array.from(new Float16Array(b.buffer)) }
	let encoder = await createEncoder({ model: o.model ?? path.join(CACHE, 'encoder.int8.onnx') }) // the graph the catalogs bind
	let cat = readCatalog(JSON.parse(readFileSync(path.join(CACHE, 'catalogs/dx7-heldout.json'), 'utf8')), encoder.binding)
	let byId = new Map(voices.map((v, i) => [v.id, i]))
	let emb = await encoder.project(picked.map(([, i]) => read(i)))
	closeSync(fd); encoder.free()
	let pool = voices.map((v, i) => [v, i]).filter(([v]) => v.split === 'heldout' && !v.silent).map(([, i]) => i)
	let jobs = picked.map(([r], k) => {
		let [best] = rank(emb[k], cat, 1), other = r.voice
		for (let s = 0; other === r.voice; s++) other = pool[seedOf(voices[r.voice].id, 'random', s) % pool.length]
		return { row: r, truth: r.voice, retrieved: byId.get(cat.entries[best.index].id), random: other, score: best.score }
	})
	let results = [], t0 = performance.now()
	await Promise.all(Array.from({ length: WORKERS }, () => new Promise((resolve, reject) => {
		let w = new Worker(new URL(import.meta.url), { workerData: { data: DATA, budget: o.budget } })
		let next = () => jobs.length ? w.postMessage(jobs.shift()) : w.postMessage(null)
		w.on('message', r => { results.push(r); console.log(`${results.length}/${picked.length}: ${r.name} ${r.hit ? 'hit' : 'miss'} random ${r.random.toFixed(3)} retrieved ${r.before.toFixed(3)} seeded ${r.seeded.toFixed(3)} blind ${r.blind.toFixed(3)} (${((performance.now() - t0) / 60000).toFixed(1)} min)`); next() })
		w.on('error', reject); w.on('exit', resolve); next()
	})))
	let summary = group => {
		let s = results.filter(group), m = k => s.reduce((a, r) => a + r[k], 0) / s.length
		if (!s.length) return { n: 0 }
		let med = k => { let v = s.map(r => r[k]).sort((a, b) => a - b); return v[v.length >> 1] }
		let both = k => ({ mean: m(k), median: med(k) })
		return { n: s.length, random: both('random'), retrieved: both('before'), seeded: both('seeded'), kept: both('kept'), blind: both('blind'),
			seededBeatsBlind: s.filter(r => r.seeded < r.blind).length / s.length, seededBeatsRetrieved: s.filter(r => r.seeded < r.before).length / s.length,
			blindBeatsRetrieved: s.filter(r => r.blind < r.before).length / s.length }
	}
	let out = { budget: o.budget, n: results.length, all: summary(() => true), misses: summary(r => !r.hit), hits: summary(r => r.hit), results }
	writeFileSync(o.out ?? path.join(CACHE, 'rebuild.json'), JSON.stringify(out, null, 1))
	console.log(JSON.stringify({ all: out.all, misses: out.misses }, null, 1))
}

async function worker() {
	let { default: match, specLoss } = await import('@audio/neural-synth')
	let { MAX } = await import('@audio/synth-dx7')
	let patches = new Uint8Array(readFileSync(path.join(workerData.data, 'patches.u8')))
	let voices = JSON.parse(readFileSync(path.join(workerData.data, 'voices.json'), 'utf8')).voices
	let patch = i => patches.subarray(i * 155, i * 155 + 155)
	let bounds = Array.from(MAX.subarray(0, 145), m => [0, m])
	parentPort.on('message', job => {
		if (!job) return parentPort.close()
		let spec = { ...job.row, pre: 0 }, target = note(patch(job.truth), spec), seed = patch(job.retrieved)
		let render = p => { let q = Uint8Array.from(seed); for (let i = 0; i < 145; i++) q[i] = Math.min(MAX[i], Math.max(0, Math.round(p[i]))); return note(q, spec) }
		let before = specLoss(target, note(seed, spec)), random = specLoss(target, note(patch(job.random), spec))
		let seeded = match(render, target, { bounds, seeds: [Array.from(seed.subarray(0, 145))], budget: workerData.budget })
		let blind = match(render, target, { bounds, budget: workerData.budget })
		parentPort.postMessage({ voice: voices[job.truth].id, name: voices[job.truth].name, retrieved: voices[job.retrieved].name, hit: job.truth === job.retrieved,
			score: job.score, note: spec.note, velocity: spec.velocity, random, before, seeded: seeded.loss, kept: Math.min(seeded.loss, before), blind: blind.loss, evals: [seeded.evals, blind.evals] })
	})
}

if (isMainThread) main().catch(e => { console.error(e); process.exit(1) })
else worker()
