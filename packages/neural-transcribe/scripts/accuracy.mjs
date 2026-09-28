// The README's accuracy table, step 1: renders labeled chords, arpeggios and vibrato lines and a
// 60 s mix with audiojs synths (the labels are the rendered notes), transcribes them, and the
// vocadito recordings when they are in the data cache, with this package and with
// @audio/mir-transcribe, and writes the estimates for scripts/accuracy.py (mir_eval) to score.
//
//   node scripts/accuracy.mjs && python scripts/accuracy.py
//
// Files: ~/.cache/audiojs/data/transcribe/<name>.wav (float32, 44.1 kHz, mono), <name>.json
// (labels), estimates.json. onnxruntime-node runs with 4 intra-op threads.
import { existsSync, mkdirSync, readFileSync, readdirSync, writeFileSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import ort from 'onnxruntime-node'
import { fetchModel } from '@audio/neural-runtime'
import poly from '@audio/synth-poly'
import pluck from '@audio/synth-pluck'
import { epiano } from '@audio/synth-fm'
import voice from '@audio/synth-voice'
import sfx from '@audio/synth-sfx'
import mirTranscribe from '@audio/mir-transcribe'
import transcribe, { MODEL } from '../transcribe.js'

const FS = 44100
const DATA = path.join(os.homedir(), '.cache', 'audiojs', 'data')
const DIR = path.join(DATA, 'transcribe')

// ------------------------------------------------------------------ render

let seed = 7
const rnd = () => (seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296
const pick = a => a[Math.floor(rnd() * a.length)]

// 5.5 Hz vibrato of ±30 cents (sfx multiplies f by 1 + d·sin); triangle, since a naive sawtooth
// aliases and its folded partials sweep against the vibrato (the 'vibrato-saw' file)
const VIB = { rate: 5.5, depth: 2 ** (30 / 1200) - 1, attack: 0.01 }
const vibrato = shape => (freq, { duration, fs }) => sfx({ freq, shape, attack: VIB.attack, sustain: Math.max(0, duration - VIB.attack), release: 0.06, vibrato: VIB.rate, vibratoDepth: VIB.depth, amp: 0.6, ...(shape === 'saw' && { lowpass: 6000 }) }, { fs })

const CHORDS = { maj: [0, 4, 7], min: [0, 3, 7], dom7: [0, 4, 7, 10], maj7: [0, 4, 7, 11], min7: [0, 3, 7, 10] }
const ROOTS = [0, 2, 4, 5, 7, 9]

function chord(root, type, lo, hi) {
	let base = lo + ((root - lo) % 12 + 12) % 12
	let notes = CHORDS[type].map(i => base + i)
	if (rnd() < 0.5 && notes[notes.length - 1] + 12 - notes[0] < hi - lo) notes = [...notes.slice(1), notes[0] + 12] // first inversion
	return notes.filter(m => m <= hi)
}

function wav(data) {
	let b = Buffer.alloc(44 + data.length * 4)
	b.write('RIFF', 0); b.writeUInt32LE(36 + data.length * 4, 4); b.write('WAVE', 8); b.write('fmt ', 12)
	b.writeUInt32LE(16, 16); b.writeUInt16LE(3, 20); b.writeUInt16LE(1, 22); b.writeUInt32LE(FS, 24)
	b.writeUInt32LE(FS * 4, 28); b.writeUInt16LE(4, 32); b.writeUInt16LE(32, 34); b.write('data', 36); b.writeUInt32LE(data.length * 4, 40)
	Buffer.from(data.buffer, data.byteOffset, data.byteLength).copy(b, 44)
	return b
}

function save(name, parts, duration) {
	let out = new Float32Array(Math.ceil(duration * FS))
	for (let { notes, voice: v, gain } of parts) {
		let y = poly(notes, { voice: v, fs: FS, duration, voices: 32 })
		for (let i = 0; i < out.length; i++) out[i] += y[i] * gain
	}
	let peak = out.reduce((m, x) => Math.max(m, Math.abs(x)), 0)
	for (let i = 0; i < out.length; i++) out[i] *= 0.9 / peak
	writeFileSync(path.join(DIR, name + '.wav'), wav(out))
	let notes = parts.flatMap(p => p.notes.map(n => ({ time: n.time, duration: n.duration, midi: n.midi, part: p.name }))).sort((a, b) => a.time - b.time || a.midi - b.midi)
	writeFileSync(path.join(DIR, name + '.json'), JSON.stringify({ rate: FS, vibrato: VIB, notes }))
	return out
}

function render() {
	// chords: 16 of 1 s every 1.25 s, FM e-piano and sawtooth voice alternating
	let ep = [], saw = []
	for (let i = 0; i < 16; i++) (i % 2 ? saw : ep).push(...chord(pick(ROOTS), pick(Object.keys(CHORDS)), 48, 76).map(midi => ({ time: 0.25 + i * 1.25, duration: 1, midi })))
	save('chords', [{ name: 'epiano', notes: ep, voice: epiano, gain: 1 }, { name: 'saw', notes: saw, voice: (f, o) => voice(f, { ...o, release: 0.1 }), gain: 1 }], 20.5)

	// arpeggio: plucked strings, eighth notes at 120 bpm over chord tones, two octaves up and down
	let arp = [], t = 0.25
	for (let bar = 0; bar < 5; bar++) {
		let tones = chord(pick(ROOTS), pick(['maj', 'min', 'dom7']), 55, 67), run = [...tones, ...tones.map(m => m + 12)]
		run = [...run, ...run.slice(1, -1).reverse()]
		for (let k = 0; k < 16; k++, t += 0.25) arp.push({ time: t, duration: 0.25, midi: run[k % run.length] })
	}
	save('arpeggio', [{ name: 'pluck', notes: arp, voice: pluck, gain: 1 }], t + 0.5)

	// vibrato line: notes of 0.6-1.4 s, as triangle and as naive sawtooth
	let mel = [], m = 67
	t = 0.25
	while (t < 19.5) {
		let d = pick([0.6, 0.8, 1, 1.2, 1.4])
		m = Math.min(79, Math.max(57, m + pick([-5, -3, -2, -1, 1, 2, 3, 4])))
		mel.push({ time: t, duration: d, midi: m })
		t += d + 0.1
	}
	save('vibrato', [{ name: 'vibrato', notes: mel, voice: vibrato('triangle'), gain: 1 }], t + 0.5)
	save('vibrato-saw', [{ name: 'vibrato', notes: mel, voice: vibrato('saw'), gain: 1 }], t + 0.5)

	// mix, 60 s: e-piano chords (2 s), plucked eighths above, the vibrato melody; no two sounding
	// notes share a pitch (a doubled pitch has no onset of its own to find)
	let chords = [], arp2 = [], mel2 = [], all = []
	let busy = (midi, t0, t1) => all.some(n => n.midi === midi && n.time < t1 && n.time + n.duration > t0)
	let add = (list, n) => { list.push(n); all.push(n) }
	for (let bar = 0; bar < 29; bar++) {
		let t0 = 0.25 + bar * 2, root = pick(ROOTS), type = pick(['maj', 'min', 'dom7', 'min7'])
		for (let midi of chord(root, type, 45, 64)) add(chords, { time: t0, duration: 1.9, midi })
		let tones = chord(root, type, 64, 76)
		for (let k = 0; k < 8; k++) {
			let midi = tones[k % tones.length] + (k >= tones.length ? 12 : 0), t1 = t0 + k * 0.25
			while (busy(midi, t1, t1 + 0.25)) midi += 12
			if (midi <= 88) add(arp2, { time: t1, duration: 0.25, midi })
		}
	}
	t = 0.25; m = 70
	while (t < 58.5) {
		let d = pick([0.6, 1, 1.4, 1.9])
		m = Math.min(84, Math.max(62, m + pick([-4, -2, -1, 1, 2, 3])))
		let midi = m
		while (busy(midi, t, t + d)) midi++
		add(mel2, { time: t, duration: d, midi })
		t += d + 0.1
	}
	save('mix', [{ name: 'epiano', notes: chords, voice: epiano, gain: 0.5 }, { name: 'pluck', notes: arp2, voice: pluck, gain: 0.7 }, { name: 'vibrato', notes: mel2, voice: vibrato('triangle'), gain: 0.8 }], 60)
}

// ------------------------------------------------------------------ transcribe

// PCM wav, 16-bit or float32, mono
function readWav(file) {
	let b = readFileSync(file), i = 12
	while (b.toString('ascii', i, i + 4) !== 'data') i += 8 + b.readUInt32LE(i + 4)
	let fmt = b.readUInt16LE(20), bits = b.readUInt16LE(34), n = b.readUInt32LE(i + 4) / (bits / 8), x = new Float32Array(n)
	if (b.readUInt16LE(22) !== 1) throw new Error(`${file}: mono expected`)
	for (let k = 0; k < n; k++) x[k] = fmt === 3 ? b.readFloatLE(i + 8 + 4 * k) : b.readInt16LE(i + 8 + 2 * k) / 32768
	return { x, rate: b.readUInt32LE(24) }
}

const NAMES = ['chords', 'arpeggio', 'vibrato', 'vibrato-saw', 'mix']
mkdirSync(DIR, { recursive: true })
if (!NAMES.every(n => existsSync(path.join(DIR, n + '.wav')))) render()

let s = await ort.InferenceSession.create(await fetchModel(MODEL), { intraOpNumThreads: 4 })
let session = () => ({
	inputs: s.inputNames.map(name => ({ name })),
	async run(feeds) {
		let out = await s.run(Object.fromEntries(Object.entries(feeds).map(([k, v]) => [k, new ort.Tensor(v.type, v.data, v.dims)])))
		return Object.fromEntries(Object.entries(out).map(([k, v]) => [k, { data: v.data, dims: v.dims }]))
	},
})
let files = NAMES.map(n => [n, path.join(DIR, n + '.wav')])
let vocadito = path.join(DATA, 'vocadito', 'Audio')
if (existsSync(vocadito)) files.push(...readdirSync(vocadito).filter(f => f.endsWith('.wav')).sort().map(f => [f.slice(0, -4), path.join(vocadito, f)]))

let est = {}
for (let [name, file] of files) {
	let { x, rate } = readWav(file)
	let neural = await transcribe(x, { sampleRate: rate, session })
	let mir = mirTranscribe(x, { fs: rate })
	est[name] = { neural: neural.map(n => [n.time, n.time + n.duration, n.freq]), mir: mir.map(n => [n.time, n.time + n.duration, 440 * 2 ** ((n.midi - 69) / 12)]) }
	console.log(`${name}: ${neural.length} notes, mir-transcribe ${mir.length}`)
}
s.release()
writeFileSync(path.join(DIR, 'estimates.json'), JSON.stringify(est))
