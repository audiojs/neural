#!/usr/bin/env node
// The frozen benchmark (scripts/freeze.js), read through the shipped path: the exported encoder
// in @audio/neural-runtime and catalogs as scripts/catalogs.py writes them.
//
//   node scripts/bench.js [--model file] [--catalogs dir] [--out file]
//
// Queries: every audible held-out voice, one note at a note and velocity no reference uses, under
// each condition of augment.js (features computed once by scripts/factory.js heldout, float16).
// Each is ranked against the held-out catalog (2,925 voices) and against every voice (dx7-all),
// with figures apart for novel voices (4 or more parameters from any training voice) and near
// copies; the algorithm head is scored on the same queries.
import { readFileSync, writeFileSync, openSync, readSync, closeSync, existsSync } from 'node:fs'
import path from 'node:path'
import os from 'node:os'
import { readCatalog, unit, sha256, featuresSha256 } from '../timbre.js'
import { load, tensor } from '@audio/neural-runtime'
import { pins } from './freeze.js'

const DATA = process.env.TIMBRE_DATA ?? path.join(os.homedir(), '.cache/audiojs/data/dx7')
const CACHE = path.join(process.env.AUDIO_NEURAL_CACHE ?? path.join(os.homedir(), '.cache/audiojs/neural'), 'timbre')

let o = {}
for (let a = process.argv.slice(2), i = 0; i < a.length; i += 2) o[a[i].replace(/^--/, '')] = a[i + 1]
let model = o.model ?? path.join(CACHE, 'encoder.int8.onnx'), dir = o.catalogs ?? path.join(CACHE, 'catalogs')

let frozen = JSON.parse(readFileSync(path.join(DATA, 'bench.json'), 'utf8')), now = pins()
for (let [k, v] of Object.entries(frozen.pins)) if (now[k] !== v) throw new Error(`bench: ${k} changed since the benchmark froze (${frozen.frozen})`)

let meta = JSON.parse(readFileSync(path.join(DATA, 'features/heldout.json'), 'utf8'))
let f = meta.features.cqt, size = f.bands * f.frames, rows = meta.rows.map(r => Object.fromEntries(meta.keys.map((k, i) => [k, r[i]])))
let stats = new Float32Array(readFileSync(path.join(DATA, 'features/heldout.stats.f32')).buffer)
let voices = JSON.parse(readFileSync(path.join(DATA, 'voices.json'), 'utf8')).voices
let fd = openSync(path.join(DATA, 'features/heldout.cqt.f16'), 'r')
let read = i => { let b = new Uint8Array(size * 2); readSync(fd, b, 0, b.length, i * size * 2); return Float32Array.from(new Float16Array(b.buffer)) }

let bytes = new Uint8Array(readFileSync(model)), binding = { encoderSha256: await sha256(bytes), featuresSha256: await featuresSha256() }
let session = await load(bytes)
// dx7-heldout is required; dx7-all joins when its references exist
let files = Object.fromEntries(['heldout', 'all'].filter(n => n === 'heldout' || existsSync(path.join(dir, `dx7-${n}.json`))).map(n => [n, JSON.parse(readFileSync(path.join(dir, `dx7-${n}.json`), 'utf8'))]))
let catalogs = Object.fromEntries(Object.entries(files).map(([n, json]) => [n, readCatalog(json, binding)]))
let position = Object.fromEntries(Object.entries(catalogs).map(([n, c]) => [n, new Map(c.entries.map((e, i) => [e.id, i]))]))

// rank of the true entry: entries scoring higher, ties broken by index as rank() does
function rankOf(e, { vectors, dimensions: d }, truth) {
	let n = vectors.length / d, t = 0
	for (let i = 0; i < d; i++) t += e[i] * vectors[truth * d + i]
	let k = 0
	for (let r = 0; r < n; r++) {
		let v = 0
		for (let i = 0; i < d; i++) v += e[i] * vectors[r * d + i]
		if (v > t || (v === t && r < truth)) k++
	}
	return k
}

let queries = rows.map((r, i) => [r, i]).filter(([r, i]) => r.kind === 'query' && stats[2 * i] >= 2 ** -13)
let results = {}, t0 = performance.now(), B = 64
for (let b = 0; b < queries.length; b += B) {
	let batch = queries.slice(b, b + B), x = new Float32Array(batch.length * size)
	batch.forEach(([, i], k) => x.set(read(i), k * size))
	let out = await session.run({ features: tensor(x, [batch.length, 1, f.bands, f.frames]) })
	let e = out.embedding.data, d = e.length / batch.length, alg = out.algorithm.data
	batch.forEach(([r], k) => {
		let v = voices[r.voice], group = v.near >= 4 ? 'novel' : 'near', emb = unit(e.subarray(k * d, (k + 1) * d))
		let a = Array.from(alg.subarray(k * 32, k * 32 + 32)), guess = a.indexOf(Math.max(...a)) + 1
		for (let [n, c] of Object.entries(catalogs)) {
			let rank = rankOf(emb, c, position[n].get(v.id))
			for (let g of [group, 'all']) {
				let s = results[`${n}|${r.cond}|${g}`] ??= { n: 0, top1: 0, top5: 0, algorithm: 0 }
				s.n++; s.top1 += rank < 1; s.top5 += rank < 5; s.algorithm += guess === v.algorithm
			}
		}
	})
	if ((b / B) % 100 === 0) console.log(`${b + batch.length}/${queries.length} queries, ${((performance.now() - t0) / 1000).toFixed(0)} s`)
}
closeSync(fd); session.free()

let table = Object.entries(results).map(([key, s]) => { let [catalog, cond, group] = key.split('|'); return { catalog, cond, group, n: s.n, top1: s.top1 / s.n, top5: s.top5 / s.n, algorithm: s.algorithm / s.n } })
let out = o.out ?? path.join(CACHE, 'bench.json')
writeFileSync(out, JSON.stringify({ model: path.basename(model), encoderSha256: binding.encoderSha256, frozen: frozen.frozen,
	catalogs: Object.fromEntries(Object.entries(files).map(([n, json]) => [n, { voices: json.entries.length, encoding: json.vectors.encoding }])),
	seconds: Math.round((performance.now() - t0) / 1000), results: table }, null, 1))
let pct = x => (100 * x).toFixed(1)
console.log(`\n| condition | held-out top-1 | top-5 | novel top-1 | novel top-5 | all voices top-1 | all top-5 | algorithm |\n|---|---:|---:|---:|---:|---:|---:|---:|`)
for (let cond of frozen.conditions) {
	let g = (c, grp) => table.find(t => t.catalog === c && t.cond === cond && t.group === grp)
	let h = g('heldout', 'all'), nv = g('heldout', 'novel'), a = g('all', 'all'), p = x => x ? pct(x) : '-'
	if (h) console.log(`| ${cond} | ${pct(h.top1)} | ${pct(h.top5)} | ${pct(nv.top1)} | ${pct(nv.top5)} | ${p(a?.top1)} | ${p(a?.top5)} | ${pct(h.algorithm)} |`)
}
console.log(`\n${out}`)
