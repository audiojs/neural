// The Accuracy section's outputs: each system on VoiceBank+DEMAND (the test set, and a training subset the defaults
// were chosen on), band-limited copies of both, clean speech, music and ten Spoken Wikipedia narrations, one float32
// file per system and input, at the input's rate. Existing files are kept, so an interrupted run resumes.
// scripts/accuracy.py prepares the band-limited and music inputs and scores the outputs.
//
//   node scripts/accuracy.mjs SET SYSTEMS [SHARD/N]    e.g. vbtrain@16000 dfn3,dfn3-0.2 0/4
//   node scripts/accuracy.mjs rooms
//
// SET, outputs in ~/.cache/audiojs/data/<dir>/<out>/<system>/:
//   vbdemand, vbtrain           noisy speech at 48 kHz (out/)
//   vbclean, vbtrain-clean      the clean utterances as input: what speech with nothing to remove loses (out-clean/)
//   vbdemand@RATE, vbtrain@RATE the noisy utterances at RATE (out@RATE/): resample_poly from 48 kHz, as a 16 kHz
//                               file or a 44.1 kHz one arrives; the models run at 48 kHz on what is under RATE/2
//   vbdemand-lp16k, vbtrain-lp16k  the noisy utterances low-passed at 16 kHz, kept at 48 kHz (out-lp16k/): a codec's
//                               cut, as MP3 encoders make it
//   music                       repair/ pieces, VocalSet chords, MUSDB18 test previews at 44.1 kHz (repair/out-neural/)
//
// Data. VoiceBank+DEMAND (Valentini-Botinhao 2017, CC BY 4.0, doi:10.7488/ds/2117):
// https://datashare.ed.ac.uk/bitstream/handle/10283/2791/{clean,noisy}_testset_wav.zip, MD5 34eb1c0b… and
// fb1b86ca… as DataShare lists them, unzipped into ~/.cache/audiojs/data/vbdemand/. vbtrain: 18 utterances of each of
// the 28 speakers of its training set (504, speaker-disjoint from the test set), as @audio/denoise's
// `python scripts/speech.py fetch` puts them into ~/.cache/audiojs/data/vbdemand-train/{clean,noisy}/; the defaults
// were chosen on these only. Spoken Wikipedia: the ROOMS files below from Wikimedia Commons, their first 60 s decoded
// to 48 kHz mono float32 into ~/.cache/audiojs/data/spoken/<name>.f32 (ffmpeg -t 60 -i FILE -ac 1 -ar 48000 -f f32le
// NAME.f32). Music: the first 60 s of the four ~/.cache/audiojs/data/repair/<name>.f32 pieces (44.1 kHz mono float32;
// vibeace: Kevin MacLeod, "Vibe Ace", CC BY 3.0; brahms: Hungarian Dance No. 5, US Army Strings, public domain;
// nutcracker: Kevin MacLeod, "Dance of the Sugar Plum Fairy", CC BY 3.0; trumpet: Mihai Sorohan, solo trumpet 06,
// Freesound 77711, CC BY 3.0); five chords of three VocalSet singers (Wilkins 2018, CC BY 4.0: straight long tones of
// female i, female i + 4 and male i summed at equal RMS) from ~/.cache/audiojs/data/vocalset/FULL/; the 50 MUSDB18
// test previews (Rafii 2017, 7 s each, the mixture stream), decoded to mono by `python scripts/accuracy.py prepare`
// into ~/.cache/audiojs/data/musdb/test-mono/. The classical systems run through `audio` (npm install audio), as its
// REPL runs them.

import { readFileSync, writeFileSync, mkdirSync, readdirSync, existsSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import resample from '@audio/resample-sinc'
import lena from 'audio-lena/raw'
import denoise, { load } from '../denoise.js'
import { enhance, level, LEVEL } from '../deepfilter.js'

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
	let b = readFileSync(file), dv = new DataView(b.buffer, b.byteOffset, b.byteLength), o = 12, ch = 1
	while (o < b.length) {
		let id = b.toString('ascii', o, o + 4), len = dv.getUint32(o + 4, true)
		if (id === 'fmt ') ch = dv.getUint16(o + 10, true)
		if (id === 'data') { let x = new Float32Array(len / 2 / ch); for (let i = 0; i < x.length; i++) x[i] = dv.getInt16(o + 8 + 2 * i * ch, true) / 32768; return x }
		o += 8 + len + (len & 1)
	}
	throw new Error(`no data chunk in ${file}`)
}
const f32 = p => { let b = readFileSync(p); return new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength)) }
const ls = (dir, ext) => readdirSync(dir).filter(f => f.endsWith(ext)).sort()

// denoise() 0.2: DeepFilterNet3 heard at -20 dBFS with the voice guard, its features hearing the empty band above the
// input's edge as it is
async function before(x, rate, model, limit) {
	let x48 = rate === 48000 ? x : resample(x, { from: rate, to: 48000 })
	let y = await enhance(x48, model.net, { limit, gain: 10 ** ((LEVEL - level(x48)) / 20), voice: true })
	return rate === 48000 ? y : resample(y, { from: 48000, to: rate }).subarray(0, x.length)
}

let audio, handles = {}
const CLASSICAL = ['omlsa', 'wiener', 'specsub', 'enhance-denoise', 'enhance']
const run = fx => async (x, rate) => (await fx(audio.from([x], { sampleRate: rate })).read())[0].subarray(0, x.length)
const SYSTEMS = {
	'rnnoise': (x, rate) => denoise(x, { sampleRate: rate, model: handles.rnnoise, limit: 0 }),
	'rnnoise-little': (x, rate) => denoise(x, { sampleRate: rate, model: handles.little, limit: 0 }),
	'rnnoise-limit16': (x, rate) => denoise(x, { sampleRate: rate, model: handles.rnnoise }),
	'dfn3-upstream': (x, rate) => rate === 48000 ? enhance(x, handles.dfn3.net, { limit: 0 }) : null,
	'dfn3': (x, rate) => denoise(x, { sampleRate: rate, model: handles.dfn3, limit: 0 }),
	'dfn3-limit18': (x, rate) => denoise(x, { sampleRate: rate, model: handles.dfn3 }),
	'dfn3-limit12': (x, rate) => denoise(x, { sampleRate: rate, model: handles.dfn3, limit: 12 }),
	'dfn3-chunk': (x, rate) => denoise(x, { sampleRate: rate, model: handles.dfn3, limit: 0, chunk: 100, warmup: 100, fade: 20 }),
	'dfn3-0.2': (x, rate) => before(x, rate, handles.dfn3, 0),
	'dfn3-0.2-limit18': (x, rate) => before(x, rate, handles.dfn3, 18),
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

// inputs: [name, rate, () => Float32Array]
async function each(inputs, list, out) {
	await prepare(list)
	for (let s of list) {
		if (!SYSTEMS[s]) throw new Error(`unknown system ${s}`)
		mkdirSync(path.join(out, s), { recursive: true })
		let c0 = process.cpuUsage(), dur = 0
		for (let [name, rate, get] of inputs) {
			let file = path.join(out, s, name + '.f32')
			if (existsSync(file)) continue
			let x = get(), y = await SYSTEMS[s](x, rate)
			if (!y) continue
			if (y.length !== x.length) throw new Error(`${s} ${name}: ${y.length} samples for ${x.length}`)
			writeFileSync(file, new Uint8Array(y.buffer, y.byteOffset, y.length * 4))
			dur += x.length / rate
		}
		let c = process.cpuUsage(c0)
		console.log(`${s}: ${dur.toFixed(0)} s of audio, ${((c.user + c.system) / 1e6).toFixed(1)} s of CPU`)
	}
	handles.dfn3?.free()
}

// VoiceBank+DEMAND sets: [inputs, output dir]
function voicebank(set) {
	let m = set.match(/^(vbdemand|vbclean|vbtrain)(-clean)?(?:@(\d+)|-(lp16k))?$/)
	if (!m) return null
	let test = m[1] !== 'vbtrain', dir = path.join(DATA, test ? 'vbdemand' : 'vbdemand-train'), clean = m[1] === 'vbclean' || !!m[2]
	let src = path.join(dir, test ? (clean ? 'clean_testset_wav' : 'noisy_testset_wav') : (clean ? 'clean' : 'noisy'))
	let names = ls(src, '.wav').map(f => f.slice(0, -4))
	if (m[3]) return [names.map(n => [n, +m[3], () => f32(path.join(dir, `rate${m[3]}`, n + '.f32'))]), path.join(dir, `out@${m[3]}`)]
	if (m[4]) return [names.map(n => [n, 48000, () => f32(path.join(dir, m[4], n + '.f32'))]), path.join(dir, `out-${m[4]}`)]
	return [names.map(n => [n, 48000, () => wav(path.join(src, n + '.wav'))]), path.join(dir, clean ? 'out-clean' : 'out')]
}

// three VocalSet singers' straight long tones, summed at equal RMS
function chords() {
	let dir = path.join(DATA, 'vocalset', 'FULL'), tones = []
	for (let s of readdirSync(dir).sort()) {
		let d = path.join(dir, s, 'long_tones', 'straight')
		if (existsSync(d)) for (let f of ls(d, '.wav')) tones.push(path.join(d, f))
	}
	let fem = tones.filter(p => p.includes(`${path.sep}female`)), male = tones.filter(p => p.includes(`${path.sep}male`))
	return [0, 1, 2, 3, 4].map(i => [`chord${i}`, 44100, () => {
		let xs = [fem[i], fem[(i + 4) % fem.length], male[i]].map(wav), n = Math.min(...xs.map(x => x.length)), y = new Float32Array(n)
		for (let x of xs) { let r = Math.sqrt(x.subarray(0, n).reduce((s, v) => s + v * v, 0) / n); for (let j = 0; j < n; j++) y[j] += .05 * x[j] / r }
		return y
	}])
}

function music() {
	let inputs = ['vibeace', 'brahms', 'nutcracker', 'trumpet'].map(n => [n, 44100, () => f32(path.join(DATA, 'repair', n + '.f32')).slice(0, 44100 * 60)])
	inputs.push(...chords())
	let mus = path.join(DATA, 'musdb', 'test-mono')
	if (existsSync(mus)) inputs.push(...ls(mus, '.f32').map(f => ['musdb-' + f.slice(0, -4), 44100, () => f32(path.join(mus, f))]))
	return inputs
}

let [what, systems, shard = '0/1'] = process.argv.slice(2), vb = what && voicebank(what), [k, n] = shard.split('/').map(Number)
if (vb) await each(vb[0].filter((_, i) => i % n === k), systems.split(','), vb[1])
else if (what === 'music') await each(music().filter((_, i) => i % n === k), systems.split(','), path.join(DATA, 'repair', 'out-neural'))
else if (what === 'rooms') {
	let dir = path.join(DATA, 'spoken')
	let inputs = Object.keys(ROOMS).map(name => [name, 48000, () => f32(path.join(dir, name + '.f32'))])
	inputs.push(['lena', 48000, () => resample(new Float32Array(lena), { from: 44100, to: 48000 })])
	await each(inputs, ['raw', 'omlsa', 'enhance-denoise', 'rnnoise', 'rnnoise-limit16', 'dfn3-upstream', 'dfn3', 'dfn3-limit18', 'dfn3-0.2-limit18', 'dfn3-limit12'].filter(s => systems ? systems.split(',').includes(s) : true), path.join(dir, 'out'))
} else console.log('usage: node scripts/accuracy.mjs vbdemand|vbclean|vbtrain|vbtrain-clean|SET@RATE|SET-lp16k|music SYSTEMS [SHARD/N] | rooms [SYSTEMS]')
