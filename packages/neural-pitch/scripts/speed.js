// Real-time factor, one thread: CPU seconds per second of audio for the features, the network, and
// pYIN's track with and without this stage 1, on a minute of rendered singing (scripts/data.js).
// node scripts/speed.js
import { track } from '@audio/pitch-pyin'
import pitch, { candidates } from '../pitch.js'
import { analyzer, normalize, BINS } from '../features.js'
import { network } from '../model.js'
import { clip } from './data.js'

for (let fs of [44100, 16000]) {
	let parts = []
	for (let s = 0; s < 10; s++) parts.push((await clip(7000 + s, { fs, source: 'voice' })).x)
	let x = new Float32Array(parts.reduce((n, p) => n + p.length, 0)), o = 0
	for (let p of parts) { x.set(p, o); o += p.length }
	let sec = x.length / fs, hop = Math.round(fs * 256 / 44100), n = Math.floor(x.length / hop) + 1
	// CPU time of the process (user + system), not wall time: on a loaded machine, waiting is not work
	let time = f => { f(); let t = process.cpuUsage(); f(); let u = process.cpuUsage(t); return (u.user + u.system) / 1e6 / sec }
	let a = analyzer(fs), db = new Float64Array(BINS), input = new Float32Array(BINS), net = network()
	let feat = time(() => { for (let i = 0; i < n; i++) a.db(x, i * hop, db) })
	normalize(db, input)
	let model = time(() => { for (let i = 0; i < n; i++) net(input, -20) })
	let raw = time(() => pitch(x, { fs }))
	let ours = time(() => track(x, { fs, candidates }))
	let yin = time(() => track(x, { fs }))
	console.log(JSON.stringify({ fs, seconds: +sec.toFixed(1), frames: n, rtf: { features: +feat.toFixed(4), network: +model.toFixed(4), pitch: +raw.toFixed(4), 'track+candidates': +ours.toFixed(4), 'track (YIN)': +yin.toFixed(4) }, msPerFrame: +((feat + model) * sec / n * 1000).toFixed(3) }))
}
