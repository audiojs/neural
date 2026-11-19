// DeepFilterNet3's mix (denoise.js mixback()), the default and `everywhere`, measured: what each gives on real noise,
// for the README's figures. Every command reads VoiceBank+DEMAND and DEMAND from the data cache (~/.cache/audiojs/data)
// and the models from the neural cache, writes nothing there, and writes only under OUT. From the repository's root (or
// anywhere):
//
//   node packages/neural-denoise/scripts/floor.mjs set OUT
//       the 66 takes, 48 kHz float32, into OUT/set (clean and noisy) and OUT/set.json: A, 42 VoiceBank+DEMAND test takes
//       as published (every 20th of the 824 sorted names: real noise at 2.5 to 17.5 dB SNR, short pauses); B, 24 made
//       of three VoiceBank test utterances each (their own leading and trailing silence trimmed), 0.5 s lead and tail,
//       a short gap (0.12, 0.2 or 0.25 s) and a long one (0.6, 1 or 2 s), under a DEMAND recording the test set does
//       not use (DWASHING, NFIELD, NPARK, NRIVER, OHALLWAY, ch01) at 0, 5, 10 or 15 dB SNR, its offset seeded
//   node .../floor.mjs vb OUT [EVERY]
//       VoiceBank+DEMAND's test files, every EVERY-th of the 824 sorted names (8, the default: 103; 1: all), the same way
//   node .../floor.mjs render OUT [LABEL=denoise.js ...]
//       over OUT/set: the model's unlimited output (OUT/y: the model runs once), the default mix of it (OUT/default) and
//       its mix with `everywhere: true` (OUT/everywhere); and the default `denoise()` of each install named (OUT/LABEL),
//       e.g. 0.7.0=<a folder holding npm's @audio/neural-denoise@0.7.0 and its two dependencies>/…/denoise.js. Printed:
//       each install's largest difference from the default and from everywhere, and the CPU time of the model
//   node .../floor.mjs score OUT
//       per take and system (OUT's folders, and the noisy input): SI-SDR (dB, at 48 kHz, zero-mean: Le Roux 2019);
//       voiced frames (20 ms; the clean take within 20 dB of its level, the mean power of the louder half of its 50 ms
//       frames) losing 10 dB or more, by the frame's power (the output's level over them matched first) and by its
//       voice alone (its projection on the clean frame, the take's SI-SDR scale removed); the bed, the output's power
//       over voiced frames against its power over pause frames (A: the clean take's frames 40 dB or more under its
//       level, runs under 0.3 s short; B: the gaps as made, 20 ms in from the voice, far: a long gap's frames 0.3 s or
//       more from it); DNSMOS P.835 SIG, BAK, OVRL (@audio/neural-quality, at the output's own level): OUT/scores.json
//   python3 -I packages/neural-denoise/scripts/floor.py OUT [stoi] [full] [systems=a,b]
//       wide-band PESQ (and STOI; with `full` SI-SDR and DNSMOS as scripts/accuracy.py scores them) at 16 kHz:
//       OUT/pesq.json
//   node .../floor.mjs table OUT REF [REF ...]
//       per system its means, over all takes and (the 66) over A, B and B at each SNR; each system paired against each
//       REF: mean ± 95 % (1.96 standard errors), and on how many takes it is ahead
//   node .../floor.mjs noise OUT [LABEL=denoise.js ...]
//       10 s of each DEMAND noise of the made takes alone, at −30 dBFS: each mix's level against the input's (OUT unused)
//   node .../floor.mjs fixture denoise.js
//       the SHA-256 of mixback() on test.js's fixture take, from any install: 0.8.0's (npm pack
//       @audio/neural-denoise@0.8.0) gives test.js's DEFAULT and DEFAULT_6_30, 0.7.0's, whose mix floors everywhere,
//       its EVERYWHERE and EVERYWHERE_6_30

import { readFileSync, writeFileSync, mkdirSync, existsSync, readdirSync } from 'node:fs'
import { createHash } from 'node:crypto'
import { fileURLToPath, pathToFileURL } from 'node:url'
import os from 'node:os'
import path from 'node:path'

const HERE = path.dirname(fileURLToPath(import.meta.url)), DATA = path.join(os.homedir(), '.cache', 'audiojs', 'data')
const VB = path.join(DATA, 'vbdemand'), SR = 48000, F20 = 960
const [what, OUT, ...rest] = process.argv.slice(2)
const tree = () => import(pathToFileURL(path.join(HERE, '..', 'denoise.js')).href)

const f32 = p => { let b = readFileSync(p); return new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength)) }
const put = (p, x) => { mkdirSync(path.dirname(p), { recursive: true }); writeFileSync(p, Buffer.from(x.buffer, x.byteOffset, x.byteLength)) }
const json = p => JSON.parse(readFileSync(p, 'utf8'))
const cpu = () => { let u = process.cpuUsage(); return (u.user + u.system) / 1e6 }
const db = v => 10 * Math.log10(v)
const power = (x, a = 0, b = x.length) => { let s = 0; for (let i = a; i < b; i++) s += x[i] * x[i]; return s / Math.max(1, b - a) }
// a take's level: the mean power of the louder half of its 50 ms frames
const active = x => { let F = 2400, p = []; for (let i = 0; i + F <= x.length; i += F) p.push(power(x, i, i + F)); p.sort((a, b) => b - a); let k = Math.max(1, p.length >> 1); return p.slice(0, k).reduce((s, v) => s + v, 0) / k }
// 16-bit PCM WAV at 48 kHz, its first channel → Float32Array, int16 / 32768
function wav(file) {
	let b = readFileSync(file), dv = new DataView(b.buffer, b.byteOffset, b.byteLength), o = 12, ch = 1, bits = 16
	while (o < b.length) {
		let id = b.toString('ascii', o, o + 4), len = dv.getUint32(o + 4, true)
		if (id === 'fmt ') { ch = dv.getUint16(o + 10, true); bits = dv.getUint16(o + 22, true); if (dv.getUint32(o + 12, true) !== SR) throw new Error(`${file}: not 48 kHz`) }
		if (id === 'data') { if (bits !== 16) throw new Error(`${file}: ${bits}-bit`); let x = new Float32Array(len / 2 / ch); for (let i = 0; i < x.length; i++) x[i] = dv.getInt16(o + 8 + 2 * i * ch, true) / 32768; return x }
		o += 8 + len + (len & 1)
	}
	throw new Error(`no data chunk in ${file}`)
}
const names = () => readdirSync(path.join(VB, 'noisy_testset_wav')).filter(f => f.endsWith('.wav')).sort()
const sets = () => json(path.join(OUT, 'set.json'))
const take = (t, k) => f32(path.join(OUT, 'set', `${t.id}.${k}.f32`))
const systems = () => readdirSync(OUT, { withFileTypes: true }).filter(d => d.isDirectory() && !['set', 'y'].includes(d.name)).map(d => d.name).sort()
const installs = list => Promise.all(list.map(async s => { let [label, p] = s.split('='); return [label, await import(pathToFileURL(path.resolve(p)).href)] }))

// ------------------------------------------------ the takes

const NOISES = ['DWASHING', 'NFIELD', 'NPARK', 'NRIVER', 'OHALLWAY']
const noise = k => wav(path.join(DATA, 'demand', k, 'ch01.wav'))

function makeSet() {
	let set = [], all = names()
	for (let i = 0; i < all.length; i += 20) {
		let id = 'A-' + all[i].replace('.wav', ''), clean = wav(path.join(VB, 'clean_testset_wav', all[i])), noisy = wav(path.join(VB, 'noisy_testset_wav', all[i])), n = Math.min(clean.length, noisy.length)
		put(path.join(OUT, 'set', `${id}.clean.f32`), clean.slice(0, n)); put(path.join(OUT, 'set', `${id}.noisy.f32`), noisy.slice(0, n))
		set.push({ id, kind: 'A', file: all[i], n })
	}
	// the noise's offset: mulberry32 from a fixed seed
	let seed = 0x9e3779b9
	const rnd = () => { seed |= 0; seed = seed + 0x6D2B79F5 | 0; let t = Math.imul(seed ^ seed >>> 15, 1 | seed); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296 }
	// an utterance from 20 ms before its first 20 ms frame within 40 dB of its level to 20 ms after its last
	const trim = x => { let L = active(x), on = []; for (let i = 0; i + F20 <= x.length; i += F20) if (power(x, i, i + F20) > L * 1e-4) on.push(i); return x.slice(Math.max(0, on[0] - F20), Math.min(x.length, on.at(-1) + 2 * F20)) }
	let beds = Object.fromEntries(NOISES.map(k => [k, noise(k)])), SHORT = [0.12, 0.2, 0.25], LONG = [0.6, 1.0, 2.0], SNRS = [0, 5, 10, 15]
	for (let k = 0; k < 24; k++) {
		let picks = [0, 1, 2].map(j => all[(k * 31 + j * 277 + 5) % all.length]), us = picks.map(p => trim(wav(path.join(VB, 'clean_testset_wav', p))))
		let gs = SHORT[k % 3], gl = LONG[Math.floor(k / 3) % 3], lead = 0.5, tail = 0.5, gaps = k % 2 ? [gl, gs] : [gs, gl]
		let n = Math.round(lead * SR) + us.reduce((s, u) => s + u.length, 0) + gaps.reduce((s, g) => s + Math.round(g * SR), 0) + Math.round(tail * SR)
		let clean = new Float32Array(n), at = Math.round(lead * SR), voice = []
		us.forEach((u, j) => { clean.set(u, at); voice.push([at, at + u.length]); at += u.length + (j < 2 ? Math.round(gaps[j] * SR) : 0) })
		let bed = beds[NOISES[k % 5]], off = Math.floor(rnd() * (bed.length - n)), snr = SNRS[k % 4]
		// the speech's power over its own stretches against the noise's mean power
		let ps = voice.reduce((s, [a, b]) => s + power(clean, a, b) * (b - a), 0) / voice.reduce((s, [a, b]) => s + b - a, 0), g = Math.sqrt(ps / power(bed, off, off + n) / 10 ** (snr / 10))
		let id = `B-${String(k).padStart(2, '0')}`
		put(path.join(OUT, 'set', `${id}.clean.f32`), clean); put(path.join(OUT, 'set', `${id}.noisy.f32`), clean.map((v, i) => v + g * bed[off + i]))
		set.push({ id, kind: 'B', files: picks, noise: NOISES[k % 5], offset: off, snr, gaps, lead, tail, voice, n })
	}
	writeFileSync(path.join(OUT, 'set.json'), JSON.stringify(set, null, 1))
	console.log(`${set.length} takes: ${set.filter(s => s.kind === 'A').length} A, ${set.filter(s => s.kind === 'B').length} B; ${(set.reduce((s, t) => s + t.n, 0) / SR).toFixed(1)} s`)
}

function makeVb(every) {
	let set = names().filter((_, i) => i % every === 0).map(f => {
		let id = f.replace('.wav', ''), clean = wav(path.join(VB, 'clean_testset_wav', f)), noisy = wav(path.join(VB, 'noisy_testset_wav', f)), n = Math.min(clean.length, noisy.length)
		put(path.join(OUT, 'set', `${id}.clean.f32`), clean.slice(0, n)); put(path.join(OUT, 'set', `${id}.noisy.f32`), noisy.slice(0, n))
		return { id, kind: 'V', file: f, n }
	})
	writeFileSync(path.join(OUT, 'set.json'), JSON.stringify(set, null, 1))
	console.log(`${set.length} VoiceBank+DEMAND test files, every ${every}th`)
}

// ------------------------------------------------ renders

async function render(list) {
	let nd = await tree(), set = sets(), h = await nd.load('deepfilternet3'), t = 0
	for (let s of set) {
		let x = take(s, 'noisy'), yp = path.join(OUT, 'y', `${s.id}.f32`), y
		if (existsSync(yp)) y = f32(yp)
		else { let c = cpu(); y = await nd.default(x, { sampleRate: SR, model: h, limit: Infinity }); t += cpu() - c; put(yp, y) }
		put(path.join(OUT, 'default', `${s.id}.f32`), nd.mixback(x, y)); put(path.join(OUT, 'everywhere', `${s.id}.f32`), nd.mixback(x, y, { everywhere: true }))
	}
	h.free()
	console.log(`the model: ${set.length} takes, its CPU time ${t.toFixed(1)} s`)
	for (let [label, m] of await installs(list)) {
		let hh = await m.load('deepfilternet3'), dR = 0, dE = 0, c = cpu()
		for (let s of set) {
			let y = await m.default(take(s, 'noisy'), { sampleRate: SR, model: hh }), r = f32(path.join(OUT, 'default', `${s.id}.f32`)), e = f32(path.join(OUT, 'everywhere', `${s.id}.f32`))
			for (let i = 0; i < y.length; i++) { dR = Math.max(dR, Math.abs(y[i] - r[i])); dE = Math.max(dE, Math.abs(y[i] - e[i])) }
			put(path.join(OUT, label, `${s.id}.f32`), y)
		}
		hh.free()
		console.log(`${label}: CPU time ${(cpu() - c).toFixed(1)} s; its largest difference from the default ${dR}, from everywhere ${dE}`)
	}
}

// ------------------------------------------------ scores

function sisdr(s, y) {
	let n = s.length, ms = 0, my = 0
	for (let i = 0; i < n; i++) { ms += s[i]; my += y[i] }
	ms /= n; my /= n
	let sy = 0, ss = 0
	for (let i = 0; i < n; i++) { sy += (s[i] - ms) * (y[i] - my); ss += (s[i] - ms) ** 2 }
	let al = sy / ss, e = 0, t = 0
	for (let i = 0; i < n; i++) { let a = al * (s[i] - ms), b = y[i] - my; t += a * a; e += (b - a) ** 2 }
	return db(t / e)
}
// a take's 20 ms frames: voiced, and its pauses by kind
function labels(t, clean) {
	let K = Math.floor(clean.length / F20), L = active(clean), pc = Array.from({ length: K }, (_, k) => power(clean, k * F20, (k + 1) * F20))
	let voiced = [], short = [], long = [], far = []
	for (let k = 0; k < K; k++) if (pc[k] >= L * .01) voiced.push(k)
	if (t.kind !== 'B') {
		for (let k = 0; k < K;) {
			if (!(pc[k] < L * 1e-4)) { k++; continue }
			let e = k; while (e < K && pc[e] < L * 1e-4) e++
			for (let j = k; j < e; j++) ((e - k) * F20 / SR < .3 ? short : long).push(j)
			k = e
		}
	} else {
		let v = t.voice, gaps = [[0, v[0][0]], [v[0][1], v[1][0]], [v[1][1], v[2][0]], [v[2][1], clean.length]]
		for (let [a, b] of gaps) {
			let len = (b - a) / SR, dv = k => Math.min(a === 0 ? Infinity : k * F20 - a, b === clean.length ? Infinity : b - (k + 1) * F20) / SR
			for (let k = Math.ceil((a + F20) / F20); (k + 1) * F20 <= b - F20; k++) { (len < .3 ? short : long).push(k); if (len >= .3 && dv(k) >= .3) far.push(k) }
		}
	}
	return { pc, voiced, short, long, far }
}
function measures(lab, clean, y) {
	let py = k => power(y, k * F20, (k + 1) * F20), mean = ks => ks.reduce((s, k) => s + py(k), 0) / ks.length, pv = mean(lab.voiced)
	let bed = ks => ks.length ? db(pv / mean(ks)) : null
	let G = lab.voiced.reduce((s, k) => s + py(k), 0) / lab.voiced.reduce((s, k) => s + lab.pc[k], 0), n = clean.length, al = 0, ss = 0
	for (let i = 0; i < n; i++) { al += y[i] * clean[i]; ss += clean[i] * clean[i] }
	al /= ss
	let voice = k => { let p = 0, q = 0; for (let i = k * F20; i < (k + 1) * F20; i++) { p += y[i] * clean[i]; q += clean[i] * clean[i] } return p / q / al }
	return {
		sisdr: sisdr(clean, y), lost: lab.voiced.filter(k => py(k) / (lab.pc[k] * G) <= .1).length / lab.voiced.length,
		lostVoice: lab.voiced.filter(k => voice(k) <= 10 ** (-10 / 20)).length / lab.voiced.length, short: bed(lab.short), long: bed(lab.long), far: bed(lab.far),
	}
}
async function score() {
	let { default: dnsmos, load } = await import(pathToFileURL(path.join(HERE, '..', '..', 'neural-quality', 'quality.js')).href)
	let model = await load(), set = sets(), out = {}, sys = systems()
	for (let t of set) {
		let clean = take(t, 'clean'), lab = labels(t, clean), n = clean.length
		out[t.id] = { kind: t.kind, snr: t.snr ?? null }
		for (let s of ['input', ...sys]) {
			let y = s === 'input' ? take(t, 'noisy') : f32(path.join(OUT, s, `${t.id}.f32`)), z = new Float32Array(n)
			z.set(y.subarray(0, n))
			let q = await dnsmos(z, { sampleRate: SR, model })
			out[t.id][s] = { ...measures(lab, clean, z), sig: q.sig, bak: q.bak, ovrl: q.ovrl }
		}
	}
	model.free()
	writeFileSync(path.join(OUT, 'scores.json'), JSON.stringify(out))
	console.log(`${set.length} takes, ${sys.length + 1} systems scored`)
}

function table(refs) {
	let S = existsSync(path.join(OUT, 'scores.json')) ? json(path.join(OUT, 'scores.json')) : {}, P = existsSync(path.join(OUT, 'pesq.json')) ? json(path.join(OUT, 'pesq.json')) : {}
	let set = sets()
	for (let t of set) { S[t.id] ??= { kind: t.kind, snr: t.snr ?? null }; for (let [s, r] of Object.entries(P)) if (r[t.id]) Object.assign(S[t.id][s] ??= {}, r[t.id]) }
	let sys = [...new Set(Object.values(S).flatMap(t => Object.keys(t).filter(k => typeof t[k] === 'object' && t[k])))]
	let KEYS = ['sisdr', 'pesq', 'stoi', 'lost', 'lostVoice', 'short', 'long', 'far', 'sig', 'bak', 'ovrl'].filter(k => Object.values(S).some(t => sys.some(s => t[s]?.[k] != null)))
	let pct = k => k.startsWith('lost'), order = ['input', ...refs, ...sys.filter(s => !['input', ...refs].includes(s))].filter(s => sys.includes(s))
	let mean = v => { v = v.filter(x => x != null && Number.isFinite(x)); return v.length ? v.reduce((s, x) => s + x, 0) / v.length : null }
	let g = (v, k) => pct(k) ? (100 * v).toFixed(2) : v.toFixed(k === 'pesq' ? 3 : k === 'stoi' ? 4 : 2)
	let groups = [['all', () => true]]
	if (set.some(t => t.kind === 'B')) groups.push(['A, VoiceBank+DEMAND as published', t => t.kind === 'A'], ['B, made, long pauses', t => t.kind === 'B'], ...[0, 5, 10, 15].map(s => [`B at ${s} dB SNR`, t => t.kind === 'B' && t.snr === s]))
	for (let [label, pick] of groups) {
		let ts = Object.values(S).filter(pick)
		console.log(`\n## ${label}: ${ts.length} takes\n${['system', ...KEYS].map(k => k.padEnd(17)).join('')}`)
		for (let s of order) console.log([s, ...KEYS.map(k => { let m = mean(ts.map(t => t[s]?.[k])); return m == null ? '–' : g(m, k) + (pct(k) ? ' %' : '') })].map(v => v.padEnd(17)).join(''))
		for (let r of refs) for (let s of order.filter(s => s !== r && s !== 'input')) {
			let row = [`${s} − ${r}`]
			for (let k of KEYS) {
				let d = ts.map(t => [t[s]?.[k], t[r]?.[k]]).filter(([a, b]) => Number.isFinite(a) && Number.isFinite(b)).map(([a, b]) => a - b)
				if (!d.length) { row.push('–'); continue }
				let m = mean(d), ci = 1.96 * Math.sqrt(d.reduce((a, x) => a + (x - m) ** 2, 0) / Math.max(1, d.length - 1) / d.length), up = d.filter(x => pct(k) ? x < 0 : x > 0).length
				row.push(`${m >= 0 ? '+' : ''}${g(m, k)}±${g(ci, k)} ${up}/${d.length}`)
			}
			console.log(row.map(v => v.padEnd(17)).join(''))
		}
	}
	console.log('\nlost, lostVoice: voiced frames losing 10 dB or more, by the frame\'s power and by its voice alone, % (paired: points; ahead: fewer)')
}

// ------------------------------------------------ noise alone

async function noiseAlone(list) {
	let nd = await tree(), h = await nd.load('deepfilternet3'), others = await installs(list), hs = await Promise.all(others.map(([, m]) => m.load('deepfilternet3')))
	for (let k of NOISES) {
		let x = noise(k).slice(SR, 11 * SR), g = 10 ** (-30 / 20) / Math.sqrt(power(x)); x = x.map(v => v * g)
		let y = await nd.default(x, { sampleRate: SR, model: h, limit: Infinity }), r = { y, default: nd.mixback(x, y), everywhere: nd.mixback(x, y, { everywhere: true }) }
		for (let [i, [label, m]] of others.entries()) r[label] = await m.default(x, { sampleRate: SR, model: hs[i] })
		console.log(`${k.padEnd(9)} ${Object.entries(r).map(([s, z]) => `${s} ${(db(power(z)) - db(power(x))).toFixed(1)}`).join(', ')} dB against the input`)
	}
	h.free(); hs.forEach(m => m.free())
}

// ------------------------------------------------ test.js's everywhere fixture

async function fixture(entry) {
	let { mixback } = await import(pathToFileURL(path.resolve(entry)).href)
	// test.js's white(): a 32-bit LCG in ±a
	const white = (n, a, seed = 1) => { let s = seed; return Float32Array.from({ length: n }, () => (s = (Math.imul(s, 1664525) + 1013904223) >>> 0, a * (s / 2 ** 32 - .5))) }
	const sha = b => createHash('sha256').update(new Uint8Array(b.buffer, b.byteOffset, b.byteLength)).digest('hex')
	let n = 8 * 48000, t = 4.5 * 48000, s = Float32Array.from({ length: n }, (_, i) => i < t ? .1 * Math.sin(2 * Math.PI * 220 * i / 48000) : 0), w = white(n, 1, 3)
	let x = s.map((v, i) => v + .1 / Math.SQRT2 * 10 ** (-(i < 6.5 * 48000 ? 10 : 40) / 20) * Math.sqrt(12) * w[i]), y = s.map((v, i) => i >= 48000 && i < 57600 ? 0 : i >= 240000 && i < 360000 && (i - 240000) % 19200 < 480 ? .1 / Math.SQRT2 * 10 ** (-27 / 20) * Math.sqrt(12) * w[i] : v)
	console.log(`${sha(mixback(x, y, { limit: 18, floor: 40 }))}  limit 18, floor 40\n${sha(mixback(x, y, { limit: 6, floor: 30 }))}  limit 6, floor 30`)
}

if (what === 'fixture') await fixture(OUT)
else if (!OUT) console.log('usage: the comment at the top of scripts/floor.mjs')
else if (what === 'set') makeSet()
else if (what === 'vb') makeVb(+(rest[0] ?? 8))
else if (what === 'render') await render(rest)
else if (what === 'score') await score()
else if (what === 'table') table(rest)
else if (what === 'noise') await noiseAlone(rest)
else throw new Error(`unknown command '${what}'`)
