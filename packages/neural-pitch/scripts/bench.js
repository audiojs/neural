// JavaScript methods on the benchmark audio (scripts/bench-prep.py):
//   pyin       @audio/pitch-pyin track, defaults
//   ours       the same pitch HMM on this package's candidates
//   ours-raw   the network alone (pitch(): every frame's peak, voicing probability)
//   pyin-notes, ours-notes   notes() with and without the candidates (Vocadito)
// node scripts/bench.js <method> [set ...] → <set>/out/<method>/<cond>/<clip>.json. Workers: $NP_WORKERS (4).
import { Worker, isMainThread, parentPort, workerData } from 'node:worker_threads'
import { readFileSync, writeFileSync, mkdirSync, existsSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { track, notes } from '@audio/pitch-pyin'

// the network loads only for its own methods, so the baselines run before any weights exist
let pitch, candidates
if (!isMainThread && workerData.method.startsWith('ours')) ({ default: pitch, candidates } = await import('../pitch.js'))

const OUT = process.env.NP_BENCH || path.join(os.homedir(), '.cache', 'audiojs', 'data', 'neural-pitch-bench')
const CONDS = ['clean', 'pink20', 'pink10', 'pink0', 'reverb']
const WORKERS = +(process.env.NP_WORKERS || 4)
const list = a => Array.from(a)

const METHODS = {
	pyin: (x, fs) => { let t = track(x, { fs }); return { times: list(t.times), f0: list(t.f0) } },
	ours: (x, fs) => { let t = track(x, { fs, candidates }); return { times: list(t.times), f0: list(t.f0) } },
	'ours-raw': (x, fs) => { let t = pitch(x, { fs }); return { times: list(t.times), f0: list(t.f0), voicing: list(t.voicing) } },
	'pyin-notes': (x, fs) => notes(x, { fs }).map(({ time, duration, freq }) => ({ time, duration, freq })),
	'ours-notes': (x, fs) => notes(x, { fs, candidates }).map(({ time, duration, freq }) => ({ time, duration, freq }))
}

if (isMainThread) {
	let [method, ...sets] = process.argv.slice(2)
	if (!METHODS[method]) { console.error('usage: node scripts/bench.js <' + Object.keys(METHODS).join('|') + '> [set ...]'); process.exit(1) }
	if (!sets.length) sets = method.endsWith('-notes') ? ['vocadito'] : ['vocadito', 'mdb-stem-synth', 'mir-1k']
	let jobs = []
	for (let s of sets) {
		let meta = JSON.parse(readFileSync(path.join(OUT, s, 'meta.json'), 'utf8'))
		for (let c of CONDS) {
			mkdirSync(path.join(OUT, s, 'out', method, c), { recursive: true })
			for (let k in meta) jobs.push({ s, c, k, fs: meta[k] })
		}
	}
	let t0 = performance.now()
	let secs = await Promise.all(Array.from({ length: WORKERS }, (_, w) => new Promise((ok, fail) => {
		let wk = new Worker(new URL(import.meta.url), { workerData: { method, jobs: jobs.filter((_, i) => i % WORKERS === w) } })
		wk.on('message', ok); wk.on('error', fail)
	})))
	let [audio, cpu] = secs.reduce(([a, b], [c, d]) => [a + c, b + d], [0, 0])
	console.log(JSON.stringify({ method, sets, clips: jobs.length, audio: Math.round(audio), cpu: Math.round(cpu), wall: Math.round((performance.now() - t0) / 1000), rtf: +(cpu / audio).toFixed(4) }))
} else {
	let run = METHODS[workerData.method], audio = 0, cpu = 0
	for (let { s, c, k, fs } of workerData.jobs) {
		let file = path.join(OUT, s, 'out', workerData.method, c, k + '.json')
		if (existsSync(file)) continue
		let b = readFileSync(path.join(OUT, s, c, k + '.f32')), x = new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4)
		let t = performance.now(), r = run(x, fs)
		cpu += (performance.now() - t) / 1000; audio += x.length / fs
		writeFileSync(file, JSON.stringify(r))
	}
	parentPort.postMessage([audio, cpu])
}
