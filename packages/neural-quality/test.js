// Against the reference: fixtures/reference.json holds, for each waveform scripts/inputs.mjs makes (their SHA-256
// checked first), what microsoft/DNS-Challenge's dnsmos_local.py at 82f1b17e77 gives (scripts/reference.py: its
// ComputeScore called as its main() calls it, librosa 0.9.1, resampy 0.2.2, onnxruntime 1.30.0): SIG, BAK, OVRL,
// num_hops, every window's raw outputs, the first sample of each window of the ramps, get_polyfit_val on some values,
// and 0.5 s of librosa's 16 kHz signal for lena at 44.1 and 8 kHz. The model's tests read sig_bak_ovr.onnx from the
// neural cache (bench/speech.py's copy too) or fetch it from GitHub at the pinned commit (skipped offline).
import test, { ok, is, throws, rejects } from 'tst'
import { existsSync, readFileSync, mkdtempSync, rmSync } from 'node:fs'
import { createHash } from 'node:crypto'
import os from 'node:os'
import path from 'node:path'
import dnsmos, { load, polyfit, windows, prepare, WINDOW, SAMPLE_RATE, MODEL, SHA256 } from './quality.js'
import { dnsmos as stat } from './audio.js'
import { inputs, lena16, ramp } from './scripts/inputs.mjs'

const REF = JSON.parse(readFileSync(new URL('./fixtures/reference.json', import.meta.url))), X = inputs()
const f32 = name => { let b = readFileSync(new URL(`./fixtures/${name}`, import.meta.url)); return new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength)) }
const sha = x => createHash('sha256').update(new Uint8Array(x.buffer, x.byteOffset, x.byteLength)).digest('hex')
const CACHE = process.env.AUDIO_NEURAL_CACHE || path.join(os.homedir(), '.cache', 'audiojs', 'neural')
const cachedModel = path.join(CACHE, 'dnsmos', 'sig_bak_ovr.onnx')
const cached = existsSync(cachedModel) && createHash('sha256').update(readFileSync(cachedModel)).digest('hex') === SHA256
let online
const reachable = async () => cached || (online ??= await fetch(MODEL, { method: 'HEAD', signal: AbortSignal.timeout(8000) }).then(r => r.ok, () => false))
const RUN = { timeout: 600_000 }
let handle
// the model's tests: the neural cache's copy, else the pinned file fetched (and cached); skipped, saying so, without either
const model = (name, fn) => test(name, RUN, async () => {
	if (!await reachable()) return console.log(`  ${cachedModel} missing and ${MODEL.host} unreachable: skipped`)
	handle ??= await load()
	await fn(handle)
})
const K = ['sig', 'bak', 'ovrl']

test('inputs: the waveforms the fixtures were made from', () => {
	for (let [name, r] of Object.entries(REF.inputs)) {
		let [x, sr] = X[name]
		is([x.length, sr], [r.samples, r.rate], `${name}: ${r.samples} samples at ${r.rate} Hz`)
		is(sha(x), r.sha256, `${name}: SHA-256 as reference.py read it (else rerun scripts/reference.py)`)
	}
	is(REF.commit, '82f1b17e7776a43eee395d0f45bae8abb700ad00')
	is(REF.model, SHA256, 'the fixtures were made with the model this package pins')
})

test('windows: where ComputeScore cuts them, on ramps that name their samples', () => {
	// each ramp's sample i is i · 2^-21; reference.py read every window's first value as the sample it starts on, the
	// doubled signal's sample a being the input's a mod n
	for (let [name, r] of Object.entries(REF.inputs)) {
		if (!name.startsWith('ramp-')) continue
		let n = r.samples, w = windows(n), all = windows(n, { windows: 'all' })
		is(w.starts.map(a => a % n), r.windows, `${name}: ${r.windows.length} windows at ${r.windows.length > 12 ? r.windows.slice(0, 12) + '…' : r.windows}`)
		is(all.starts.length, r.num_hops, `${name}: num_hops ${r.num_hops}, ${r.num_hops - r.windows.length} of them dropped`)
		is(r.lengths, [WINDOW], 'every window the network got: 144160 samples')
		ok(all.starts.every(a => a + WINDOW <= w.length), `'all': every window inside the doubled signal (${w.length})`)
	}
	// the float product int((i + 9.01) · 16000) falls a sample short for i = 7 to 23: a 30 s clip is scored on its first 15.01 s
	is(windows(16000 * 30).starts, [0, 1, 2, 3, 4, 5, 6].map(i => i * SAMPLE_RATE))
	is(windows(16000 * 30, { windows: 'all' }).starts.length, 21)
	is(windows(1600).length, 1600 * 128, '0.1 s doubled 7 times')
	is(windows(WINDOW), { length: WINDOW, starts: [0] })
	throws(() => windows(0), /no windows/)
	throws(() => windows(100, { windows: 'some' }), /upstream.*all/)
})

test('polyfit: get_polyfit_val, non-personalized', () => {
	for (let { raw: [sig, bak, ovrl], mos } of REF.polyfit) {
		let p = polyfit({ sig, bak, ovrl }), d = Math.max(...K.map((k, i) => Math.abs(p[k] - mos[i])))
		ok(d <= 1e-15, `raw ${[sig, bak, ovrl].map(v => v.toFixed(4))}: ${K.map(k => p[k].toFixed(4))}, |diff| ${d.toExponential(1)}`)
	}
	is(polyfit({ sig: 0, bak: 0, ovrl: 0 }), { sig: 0.0052439, bak: -0.39604546, ovrl: 0.04602535 }, 'the constant terms')
})

test('prepare: librosa 0.9.1\'s resample (resampy 0.2.2\'s kaiser_best), sample for sample', () => {
	let [a, b] = REF.slice
	for (let name of ['lena@44.1k', 'lena@8k']) {
		let [x, sr] = X[name], r = REF.inputs[name], y = prepare(x, { sampleRate: sr }), want = f32(`${name}.librosa16.f32`), d = 0, n = 0
		is(y.length, r.length16, `${name}: ceil(n · 16000 / ${sr}) = ${r.length16} samples, librosa's length`)
		for (let i = a; i < b; i++) { let e = Math.abs(y[i] - want[i - a]); if (e) n++; d = Math.max(d, e) }
		// measured: no sample differs, on these and on 41 files at 8 to 48 kHz; a float32 step allowed for another libm's sine
		ok(d <= 2 ** -23, `${name}, samples ${a} to ${b}: max |diff| ${d.toExponential(1)}, ${n} differ`)
	}
	// lena's two lengths have ceil = round; 43,520 samples at 24 kHz do not (29,013.3)
	is(prepare(new Float32Array(43520), { sampleRate: 24000 }).length, 29014, 'ceil(43520 · 16000 / 24000), librosa 0.9.1 core/audio.py line 583, not its rounding')
	is(prepare(X['lena@44.1k'][0], { sampleRate: 44100 }), lena16(), 'lena at 44.1 kHz: inputs.mjs\'s 16 kHz lena')
	let x = X['lena@16k'][0]
	ok(prepare(x, { sampleRate: 16000 }) !== x, 'at 16 kHz a copy')
	is(prepare([x, x], { sampleRate: 16000 }), x, 'two equal channels: the one')
	is(prepare({ channelData: [x], sampleRate: 16000 }), x, '{ channelData, sampleRate }')
	throws(() => prepare(x), /sampleRate/)
	throws(() => prepare(x, { sampleRate: -1 }), /sampleRate/)
	throws(() => prepare(new Float32Array(0), { sampleRate: 16000 }), /no samples/)
	throws(() => prepare([x, x.subarray(1)], { sampleRate: 16000 }), /lengths/)
	throws(() => prepare(Float32Array.of(0, NaN, 0), { sampleRate: 16000 }), /sample 1 is NaN/)
	throws(() => prepare([Float32Array.of(0, 1), Float32Array.of(0, -Infinity)], { sampleRate: 16000 }), /sample 1 is -Infinity/)
	throws(() => prepare('lena', { sampleRate: 16000 }), TypeError)
})

model('dnsmos: the reference\'s SIG, BAK, OVRL and every window\'s raw outputs', async m => {
	// measured: raw outputs equal to the last bit at every rate, the means within 1.8e-15 (numpy sums pairwise)
	for (let name of ['lena@16k', 'lena-white5@16k', 'lena@44.1k', 'lena@8k', 'lena-0.1s@16k', 'silence@16k']) {
		let [x, sr] = X[name], r = REF.inputs[name], s = await dnsmos(x, { sampleRate: sr, model: m })
		is(s.segments.length, r.raw.length, `${name}: ${r.raw.length} windows`)
		let raw = Math.max(...s.segments.flatMap((g, i) => g.raw.map((v, k) => Math.abs(v - r.raw[i][k]))))
		ok(raw <= 1e-5, `${name}: raw outputs within ${raw.toExponential(1)}`)
		let d = Math.max(...K.map(k => Math.abs(s[k] - r[k.toUpperCase()])))
		ok(d <= 1e-4, `${name}: SIG ${s.sig.toFixed(4)} BAK ${s.bak.toFixed(4)} OVRL ${s.ovrl.toFixed(4)}, within ${d.toExponential(1)} of the reference`)
		ok(s.segments.every(g => g.time === Math.round(g.time) && K.every(k => g[k] === polyfit({ sig: g.raw[0], bak: g.raw[1], ovrl: g.raw[2] })[k])), 'each window: its polyfit, at a whole second')
	}
})

model('dnsmos: noise scores lower, BAK the most; speech over silence', async m => {
	let clean = await dnsmos(X['lena@16k'][0], { sampleRate: 16000, model: m }), noisy = await dnsmos(X['lena-white5@16k'][0], { sampleRate: 16000, model: m })
	ok(K.every(k => noisy[k] < clean[k]), `white noise 5 dB down: SIG ${clean.sig.toFixed(2)}→${noisy.sig.toFixed(2)}, BAK ${clean.bak.toFixed(2)}→${noisy.bak.toFixed(2)}, OVRL ${clean.ovrl.toFixed(2)}→${noisy.ovrl.toFixed(2)}`)
	ok(clean.bak - noisy.bak > clean.sig - noisy.sig, 'the background falls more than the voice')
	let silence = await dnsmos(X['silence@16k'][0], { sampleRate: 16000, model: m })
	ok(silence.ovrl < clean.ovrl, `digital silence: OVRL ${silence.ovrl.toFixed(2)}, not a clean recording`)
})

model('dnsmos: channels mixed to their mean; windows \'all\' hears past 15.01 s', async m => {
	let x = X['lena@16k'][0], y = X['lena-white5@16k'][0], mean = x.map((v, i) => (v + y[i]) / 2)
	is(await dnsmos([x, y], { sampleRate: 16000, model: m }), await dnsmos(mean, { sampleRate: 16000, model: m }), 'two channels: their mean')
	is(await dnsmos([x, x], { sampleRate: 16000, model: m }), await dnsmos(x, { sampleRate: 16000, model: m }), 'two equal channels: the one')
	// 30 s: lena, then white noise over 16 to 30 s at its level. The reference's windows end at 15.01 s and miss it
	let n = 16000 * 30, a = Float32Array.from({ length: n }, (_, i) => x[i % x.length]), b = a.slice(), rnd = 5
	for (let i = 16000 * 16; i < n; i++) b[i] += 0.05 * ((rnd = (rnd * 1664525 + 1013904223) % 4294967296) / 4294967296 - 0.5)
	let [ua, ub] = [await dnsmos(a, { sampleRate: 16000, model: m }), await dnsmos(b, { sampleRate: 16000, model: m })]
	is([ua.segments.length, ub.sig, ub.bak, ub.ovrl], [7, ua.sig, ua.bak, ua.ovrl], 'upstream: 7 windows, the noise unheard')
	let [wa, wb] = [await dnsmos(a, { sampleRate: 16000, model: m, windows: 'all' }), await dnsmos(b, { sampleRate: 16000, model: m, windows: 'all' })]
	is(wa.segments.map(s => s.time), Array.from({ length: 21 }, (_, i) => i), "'all': 21 windows, one a second")
	ok(wb.bak < wa.bak - 0.1, `'all': the noise heard, BAK ${wa.bak.toFixed(2)}→${wb.bak.toFixed(2)}`)
})

model('load: a handle held across calls; without one, loaded and freed per call', async m => {
	let x = X['lena-0.1s@16k'][0], r = REF.inputs['lena-0.1s@16k'], own = await dnsmos(x, { sampleRate: 16000 })
	is(own, await dnsmos(x, { sampleRate: 16000, model: m }), 'the same scores either way')
	ok(Math.abs(own.ovrl - r.OVRL) <= 1e-4, `0.1 s, doubled to 12.8 s: OVRL ${own.ovrl.toFixed(4)} (reference ${r.OVRL.toFixed(4)})`)
	let h = await load(), s = await dnsmos(x, { sampleRate: 16000, model: h })
	h.free()
	await rejects(() => dnsmos(x, { sampleRate: 16000, model: h }), /free/, 'a freed handle refuses')
	is(s, own)
})

model('audio.js: the stat manifest', async m => {
	is(stat.stat, 'dnsmos')
	let x = X['lena@16k'][0]
	is(await stat.compute([x], { sampleRate: 16000, model: m }), await dnsmos([x], { sampleRate: 16000, model: m, windows: 'all' }), "compute(channels, { sampleRate }) is dnsmos() over every window")
	is(await stat.compute([x], { sampleRate: 16000, model: m, windows: 'upstream' }), await dnsmos([x], { sampleRate: 16000, model: m }), "and with windows: 'upstream', the reference script's")
})

test('weights: refused unless of SHA256; an unreachable copy names opts.weights', async () => {
	let dir = mkdtempSync(path.join(os.tmpdir(), 'neural-quality-')), was = process.env.AUDIO_NEURAL_CACHE, x = ramp(1600)
	process.env.AUDIO_NEURAL_CACHE = dir
	try {
		let bad = async () => new Response(new Uint8Array(1000))
		await rejects(() => dnsmos(x, { sampleRate: 16000, fetch: bad }), /SHA-256 [0-9a-f]{64}, expected 269fbebd/, 'other bytes: refused')
		ok(!existsSync(path.join(dir, 'dnsmos', 'sig_bak_ovr.onnx')), 'and not cached')
		let down = async () => { throw new Error('offline') }
		await rejects(() => dnsmos(x, { sampleRate: 16000, fetch: down }), /not read \(.*offline.*\).*opts\.weights/)
	} finally {
		was == null ? delete process.env.AUDIO_NEURAL_CACHE : process.env.AUDIO_NEURAL_CACHE = was
		rmSync(dir, { recursive: true })
	}
	await rejects(() => dnsmos(x, { sampleRate: 0 }), /sampleRate/, 'the input checked before the model is read')
})
