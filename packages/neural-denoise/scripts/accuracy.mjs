// The Accuracy section's outputs: each system on the VoiceBank+DEMAND test set and on ten Spoken Wikipedia
// narrations, one float32 file per system and input (48 kHz), in ~/.cache/audiojs/data/{vbdemand,spoken}/out/.
// Existing files are kept, so an interrupted run resumes. scripts/accuracy.py scores them.
//
//   node scripts/accuracy.mjs vbdemand SYSTEMS [SHARD/N]    e.g. rnnoise,dfn3,omlsa 0/4
//   node scripts/accuracy.mjs rooms
//
// Data. VoiceBank+DEMAND (Valentini-Botinhao 2017, CC BY 4.0, doi:10.7488/ds/2117):
// https://datashare.ed.ac.uk/bitstream/handle/10283/2791/{clean,noisy}_testset_wav.zip, MD5 34eb1c0b… and
// fb1b86ca… as DataShare lists them, unzipped into ~/.cache/audiojs/data/vbdemand/. Spoken Wikipedia: the
// ROOMS files below from Wikimedia Commons, their first 60 s decoded to 48 kHz mono float32 into
// ~/.cache/audiojs/data/spoken/<name>.f32 (ffmpeg -t 60 -i FILE -ac 1 -ar 48000 -f f32le NAME.f32).
// The classical systems run through `audio` (npm install audio), as its REPL runs them.

import { readFileSync, writeFileSync, mkdirSync, readdirSync, existsSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import resample from '@audio/resample-sinc'
import lena from 'audio-lena/raw'
import denoise, { load } from '../denoise.js'

const DATA = path.join(os.homedir(), '.cache', 'audiojs', 'data')

// the first file, by title, of the English Spoken Wikipedia category (Wikimedia Commons) in each of ten years
export const ROOMS = {
	'2007-A_Series_of_Unfortunate_Events': 'A Series of Unfortunate Events.ogg',
	'2009-2005_Atlantic_hurricane_season': '2005 Atlantic hurricane season.ogg',
	'2011-1943_The_Battle_of_Midway': '1943 The Battle of Midway.ogg',
	'2014-Abbot_Augustus_Low': 'Abbot Augustus Low.ogg',
	'2016-Adelaide_article': 'Adelaide article.oga',
	'2019-2015_East_Village_gas_explosion': '2015 East Village gas explosion.ogg',
	'2021-EN_Adolf_Hitler_article': 'EN-Adolf Hitler-article.ogg',
	'2023-1964_Textile_250_spoken_article': '1964 Textile 250 spoken article.ogg',
	'2024-_Absent_minded_professor__from_Wikipedia': "'Absent-minded professor' from Wikipedia.ogg",
	'2026-AIM_174B_EN_recording': 'AIM-174B EN recording.ogg',
}

function wav(file) {
	let b = readFileSync(file), dv = new DataView(b.buffer, b.byteOffset, b.byteLength), o = 12
	while (o < b.length) {
		let id = b.toString('ascii', o, o + 4), len = dv.getUint32(o + 4, true)
		if (id === 'data') { let x = new Float32Array(len / 2); for (let i = 0; i < x.length; i++) x[i] = dv.getInt16(o + 8 + 2 * i, true) / 32768; return x }
		o += 8 + len + (len & 1)
	}
	throw new Error(`no data chunk in ${file}`)
}

let audio, handles = {}
const CLASSICAL = ['omlsa', 'wiener', 'specsub', 'enhance-denoise', 'enhance']
const run = fx => async x => (await fx(audio.from([x], { sampleRate: 48000 })).read())[0].subarray(0, x.length)
const SYSTEMS = {
	'rnnoise': x => denoise(x, { sampleRate: 48000, model: handles.rnnoise, limit: 0 }),
	'rnnoise-little': x => denoise(x, { sampleRate: 48000, model: handles.little, limit: 0 }),
	'rnnoise-limit20': x => denoise(x, { sampleRate: 48000, model: handles.rnnoise, limit: 20 }),
	'dfn3': x => denoise(x, { sampleRate: 48000, model: handles.dfn3, limit: 0 }),
	'dfn3-chunk': x => denoise(x, { sampleRate: 48000, model: handles.dfn3, limit: 0, chunk: 100, warmup: 100, fade: 20 }),
	'dfn3-limit12': x => denoise(x, { sampleRate: 48000, model: handles.dfn3, limit: 12 }),
	'raw': x => x,
	'omlsa': run(a => a.omlsa()),
	'wiener': run(a => a.wiener()),
	'specsub': run(a => a.specsub()),
	'enhance-denoise': run(a => a.highpass(80).dehum().omlsa()),
	'enhance': run(a => a.highpass(80).dehum().omlsa().deesser().compressor({ threshold: -24, ratio: 3 }).eq(3000, 2, 1).normalize('podcast')),
}

async function prepare(list) {
	if (list.some(s => s.startsWith('rnnoise')) && !handles.rnnoise) {
		handles.rnnoise = await load('rnnoise')
		let little = path.join(process.env.AUDIO_NEURAL_CACHE || path.join(os.homedir(), '.cache', 'audiojs', 'neural'), 'rnnoise', 'rnnoise-little.bin')
		if (existsSync(little)) handles.little = await load('rnnoise', { weights: readFileSync(little) })
	}
	if (list.some(s => s.startsWith('dfn3')) && !handles.dfn3) handles.dfn3 = await load('deepfilternet3', { sessionOptions: { intraOpNumThreads: 4, interOpNumThreads: 1 } })
	if (list.some(s => CLASSICAL.includes(s)) && !audio)
		audio = (await import('audio').catch(() => { throw new Error('the classical systems need `audio`: npm install audio') })).default
}

async function each(inputs, list, out) {
	await prepare(list)
	for (let s of list) {
		if (!SYSTEMS[s]) throw new Error(`unknown system ${s}`)
		mkdirSync(path.join(out, s), { recursive: true })
		let c0 = process.cpuUsage(), dur = 0
		for (let [name, get] of inputs) {
			let file = path.join(out, s, name + '.f32')
			if (existsSync(file)) continue
			let x = get(), y = await SYSTEMS[s](x)
			if (y.length !== x.length) throw new Error(`${s} ${name}: ${y.length} samples for ${x.length}`)
			writeFileSync(file, new Uint8Array(y.buffer, y.byteOffset, y.length * 4))
			dur += x.length / 48000
		}
		let c = process.cpuUsage(c0)
		console.log(`${s}: ${dur.toFixed(0)} s of audio, ${((c.user + c.system) / 1e6).toFixed(1)} s of CPU`)
	}
	handles.dfn3?.free()
}

let [what, systems, shard = '0/1'] = process.argv.slice(2)
if (what === 'vbdemand') {
	let dir = path.join(DATA, 'vbdemand'), [k, n] = shard.split('/').map(Number)
	let names = readdirSync(path.join(dir, 'noisy_testset_wav')).filter(f => f.endsWith('.wav')).sort().filter((_, i) => i % n === k)
	await each(names.map(f => [f.slice(0, -4), () => wav(path.join(dir, 'noisy_testset_wav', f))]), systems.split(','), path.join(dir, 'out'))
} else if (what === 'rooms') {
	let dir = path.join(DATA, 'spoken'), f32 = p => { let b = readFileSync(p); return new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4) }
	let inputs = Object.keys(ROOMS).map(name => [name, () => f32(path.join(dir, name + '.f32'))])
	inputs.push(['lena', () => resample(new Float32Array(lena), { from: 44100, to: 48000 })])
	await each(inputs, ['raw', 'omlsa', 'enhance-denoise', 'rnnoise', 'dfn3', 'dfn3-limit12'], path.join(dir, 'out'))
} else console.log('usage: node scripts/accuracy.mjs vbdemand SYSTEMS [SHARD/N] | rooms')
