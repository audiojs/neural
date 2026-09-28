// Development check of the whole path, on data kept out of training and of the benchmark:
//   synth   the development clips of scripts/factory.js (seeds 900000…), rendered again with their labels
//   vocal   held-out VocalSet singers (female9, male11), degraded as in training, against pYIN on the
//           clean recording (sure frames only)
// Scores pYIN's track with this stage 1 (κ values), and the network alone.
// node scripts/dev.js [clips=120] [kappas=0.5,1]
import { readFileSync } from 'node:fs'
import { track } from '@audio/pitch-pyin'
import pitch, { candidates } from '../pitch.js'
import { clip, degrade, rng } from './data.js'
import { readWav } from './wav.js'

let [clips = 120, ks = '0.5,1'] = process.argv.slice(2)
let kappas = ks.split(',').map(Number)
const SP = process.env.NP_VOCAL_DEV
const cents = (a, b) => 1200 * Math.log2(a / b)

function score(acc, est, ref, weight = null) {
	for (let j = 0; j < ref.length; j++) {
		let r = ref[j], e = est[j]
		if (r > 0 && (!weight || weight[j])) { acc.v++; if (e > 0) { acc.vr++; if (Math.abs(cents(e, r)) < 50) acc.rpa++ } }
		if (!(r > 0)) { acc.u++; if (e > 0) acc.vfa++ }
	}
}
const report = (name, a) => console.log(name.padEnd(22), 'RPA', (a.rpa / a.v).toFixed(4), 'VR', (a.vr / a.v).toFixed(4), 'VFA', (a.vfa / a.u).toFixed(4), 'frames', a.v, a.u)
const fresh = () => ({ rpa: 0, v: 0, vr: 0, u: 0, vfa: 0 })

// synthetic
let S = Object.fromEntries([...kappas.map(k => ['track κ=' + k, fresh()]), ['raw', fresh()], ['yin', fresh()]])
for (let i = 0; i < +clips; i++) {
	let c = await clip(900000 + i), hop = Math.round(c.fs * 256 / 44100), at = Array.from(c.frames, f => Math.round(f / hop))
	let pick = t => at.map(k => t[k] ?? 0)
	for (let k of kappas) score(S['track κ=' + k], pick(track(c.x, { fs: c.fs, candidates: (N, fs, lo, hi) => candidates(N, fs, lo, hi, { kappa: k }) }).f0), c.f0, c.weight)
	let r = pitch(c.x, { fs: c.fs })
	score(S.raw, pick(Array.from(r.f0, (f, j) => r.voicing[j] >= 0.5 ? f : 0)), c.f0, c.weight)
	score(S.yin, pick(track(c.x, { fs: c.fs }).f0), c.f0, c.weight)
}
console.log(`synthetic development clips: ${clips}`)
for (let k in S) report(k, S[k])

// held-out singers
if (SP) {
	let files = readFileSync(SP, 'utf8').split('\n').filter(Boolean)
	let V = Object.fromEntries([...kappas.map(k => ['track κ=' + k, fresh()]), ['raw', fresh()], ['yin', fresh()]])
	for (let [i, f] of files.entries()) {
		let { data, fs } = readWav(f), teacher = track(data, { fs })
		let ref = Array.from(teacher.f0), sure = ref.map((v, j) => v > 0 ? teacher.prob[j] >= 0.5 && [-2, -1, 1, 2].every(d => (teacher.f0[j + d] || 0) > 0 && Math.abs(cents(teacher.f0[j + d], v)) <= 50) : true)
		let x = await degrade(rng(700000 + i), Float32Array.from(data), fs)
		for (let k of kappas) score(V['track κ=' + k], track(x, { fs, candidates: (N, r, lo, hi) => candidates(N, r, lo, hi, { kappa: k }) }).f0, ref, sure)
		let r = pitch(x, { fs })
		score(V.raw, Array.from(r.f0, (v, j) => r.voicing[j] >= 0.5 ? v : 0), ref, sure)
		score(V.yin, track(x, { fs }).f0, ref, sure)
	}
	console.log(`held-out singers, degraded, against pYIN on the clean recording: ${files.length} files`)
	for (let k in V) report(k, V[k])
}
