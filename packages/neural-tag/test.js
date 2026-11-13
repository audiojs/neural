// Against upstream: fixtures/ hold, for each waveform scripts/inputs.mjs makes (their SHA-256 checked first), the Keras
// model's scores (tensorflow/models at export.py's COMMIT, yamnet.h5) and the log-mel of some of its patches as
// features.py makes them (scripts/export.py --reference). The model's tests read yamnet.fp16.onnx from the neural cache
// (export.py writes it there) or from the hosting directory prepared for its upload, else the hosted copy once REVISION
// is set; without any they are skipped.
import test, { ok, is, throws, rejects } from 'tst'
import { existsSync, readFileSync, mkdtempSync, rmSync } from 'node:fs'
import { createHash } from 'node:crypto'
import os from 'node:os'
import path from 'node:path'
import resample from '@audio/resample-sinc'
import { tag, scores, features, CLASSES, MODEL, REVISION, SHA256, FRAMES, BANDS } from './tag.js'
import { inputs, lena44, sweep, beep, LENA_RATE } from './scripts/inputs.mjs'

const f32 = name => { let b = readFileSync(new URL(`./fixtures/${name}`, import.meta.url)); return new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength)) }
const REF = JSON.parse(readFileSync(new URL('./fixtures/reference.json', import.meta.url))), X = inputs()
const sha = x => createHash('sha256').update(new Uint8Array(x.buffer, x.byteOffset, x.byteLength)).digest('hex')
const CACHE = process.env.AUDIO_NEURAL_CACHE || path.join(os.homedir(), '.cache', 'audiojs', 'neural')
const FP16 = [path.join(CACHE, 'yamnet', 'yamnet.fp16.onnx'), path.join(CACHE, 'hosting', 'yamnet', 'yamnet.fp16.onnx')].find(existsSync)
const FP32 = path.join(CACHE, 'yamnet', 'yamnet.onnx')
const W = FP16 ? { weights: FP16 } : REVISION ? {} : null
const model = W ? test : test.skip
if (!W) console.log(`  no yamnet.fp16.onnx in ${CACHE}/yamnet or ${CACHE}/hosting/yamnet, and REVISION unset: the model's tests are skipped (scripts/export.py writes it)`)
const RUN = { timeout: 300_000 }
const maxDiff = (rows, want) => rows.reduce((m, r, i) => r.reduce((m, v, k) => Math.max(m, Math.abs(v - want[i * CLASSES.length + k])), m), 0)

test('inputs: the waveforms the fixtures were made from', () => {
	for (let [key, r] of Object.entries(REF.inputs)) {
		let x = X[key.split('@')[0]]
		is(x.length, r.samples, `${key}: ${r.samples} samples`)
		is(sha(x), r.sha256, `${key}: SHA-256 as export.py read it (else rerun scripts/export.py --reference)`)
	}
})

test('features: upstream\'s log-mel patches, patch for patch', () => {
	// upstream computes in float32 and its own float64 run differs from it by up to 3.3e-3 where a synthetic tone leaves
	// a band near the log's floor, log(0.001); these features are within 2.4e-7 of that float64 run
	let tol = { lena: 1e-4, noise: 1e-4, pluck: 2e-3, sweep: 5e-3, beep: 2e-3 }
	for (let [key, r] of Object.entries(REF.inputs)) {
		let name = key.split('@')[0], f = features(X[name], { hop: r.hop }), N = FRAMES * BANDS
		is(f.patches, r.patches, `${key}: ${r.patches} patches, as pad_waveform pads`)
		is(f.data.length, f.patches * N)
		if (!r.features.length) continue
		let want = f32(`${key}.patches.f32`), d = 0
		r.features.forEach((p, j) => { for (let i = 0; i < N; i++) d = Math.max(d, Math.abs(f.data[p * N + i] - want[j * N + i])) })
		ok(d < tol[name], `${key}, patches ${r.features}: max |diff| ${d.toExponential(2)}`)
	}
	let floor = Math.fround(Math.log(0.001))
	for (let name of ['silence', 'empty']) ok(features(X[name]).data.every(v => v === floor), `${name}: every cell log(0.001)`)
	is(features(new Float32Array(15600)).patches, 1, '0.975 s: one patch')
	is(features(new Float32Array(15601)).patches, 2, 'a sample more: a second, its tail zeros')
	is(features(new Float32Array(15600 + 7680)).patches, 2, 'and a hop more: still two')
	is(features(new Float32Array(16000 * 10), { hop: 0.96 }).patches, 1 + Math.ceil((160000 - 15600) / 15360), 'hop 0.96 s')
	throws(() => features(new Float32Array(100), { hop: 0.004 }), /hop/)
})

model('scores: these features through the float16 model, against the Keras model on the same waveform', RUN, async () => {
	// the float16 weights' rounding: 3.6e-3 at most over the 49 patches (scripts/export.py --verify), the float32 export's 1.6e-6
	let fp32 = existsSync(FP32)
	for (let [key, r] of Object.entries(REF.inputs)) {
		let x = X[key.split('@')[0]], want = f32(`${key}.scores.f32`), s = await scores(x, 16000, { ...W, hop: r.hop })
		is(s.scores.length, r.patches, `${key}: ${r.patches} patches`)
		is(s.hop, r.hop)
		// Keras's top 3 in its order, those of them over 0.01 (silence's others score 1e-7 or so, their order noise)
		let d = maxDiff(s.scores, want), kept = r.top.filter(t => t[2] >= 0.01)
		let top = (await tag(x, 16000, { ...W, hop: r.hop, top: kept.length })).map(t => t.index)
		ok(d < 5e-3, `${key}: max |score − Keras| ${d.toExponential(2)}`)
		is(top, kept.map(t => t[0]), `${key}: Keras's top ${kept.length}, ${kept.map(t => `${t[1]} ${t[2]}`).join(', ')}`)
		if (fp32) { let d32 = maxDiff((await scores(x, 16000, { weights: FP32, hop: r.hop })).scores, want); ok(d32 < 1e-4, `${key}, float32 export: ${d32.toExponential(2)}`) }
	}
})

model('tag: lena is speech; 44.1 kHz in, resampled as the fixture was', RUN, async () => {
	let t = await tag(lena44(), LENA_RATE, W)
	is(t.length, 5, 'top 5 by default')
	is(t[0].name, 'Speech')
	is(t[0], { name: 'Speech', id: '/m/09x0r', index: 0, score: t[0].score })
	ok(t[0].score > 0.5 && t[0].score < 1, `Speech ${t[0].score.toFixed(3)}`)
	ok(t.every((x, i) => !i || t[i - 1].score >= x.score), 'best first')
	let s = await scores(lena44(), LENA_RATE, W), want = f32('lena.scores.f32')
	ok(maxDiff(s.scores, want) < 5e-3, 'the same scores as at 16 kHz')
	is(s.times.length, s.scores.length)
	is(s.times[1], 0.48)
})

model('tag: 48 kHz agrees with 16 kHz, 44.1 kHz too', RUN, async () => {
	// lena at 48 kHz (resampled from its 44.1) against lena at 16 kHz: 1.4e-3 at most measured, the top 5 the same.
	// A synthetic tone moves more: the empty bands between its harmonics sit at the log's floor, where the resampler's
	// in-band ripple (83 dB SNR) shows; its patches up to 2.5e-2, its mean scores 2.6e-3
	let l16 = resample(lena44(), { from: LENA_RATE, to: 16000 }), l48 = resample(lena44(), { from: LENA_RATE, to: 48000 })
	let a = await scores(l48, 48000, W), b = await scores(l16, 16000, W), d = maxDiff(a.scores, b.scores.flatMap(r => [...r]))
	ok(d < 5e-3, `lena 48 kHz: max |diff| per patch ${d.toExponential(2)}`)
	is((await tag(l48, 48000, W)).map(t => t.index), (await tag(l16, 16000, W)).map(t => t.index), 'lena 48 kHz: the same top 5')
	for (let sr of [44100, 48000]) for (let [name, make] of [['sweep', sweep], ['beep', beep]]) {
		let p = await tag(make(sr), sr, W), q = await tag(make(16000), 16000, W), m = Math.max(...p.map((t, k) => Math.abs(t.score - q[k].score)))
		is(p.map(t => t.index), q.map(t => t.index), `${name} at ${sr}: the same top 5 (${p[0].name}, …)`)
		ok(m < 5e-3, `${name} at ${sr}: their mean scores within ${m.toExponential(2)}`)
	}
})

model('tag: channels are mixed to their mean; shorter than a patch, silence and nothing as upstream', RUN, async () => {
	let x = X.pluck, y = sweep(16000), mean = x.map((v, i) => (v + y[i]) / 2)
	is(await tag([x, y], 16000, W), await tag(mean, 16000, W), 'two channels: their mean, as inference.py')
	is(await tag([x, x], 16000, W), await tag(x, 16000, W), 'two equal channels: the one')
	await rejects(() => tag([x, y.subarray(1)], 16000, W), /lengths/)
	let b = await scores(X.beep, 16000, W)
	is([b.scores.length, b.times], [1, [0]], 'a 0.5 s beep: one patch, zeros after it')
	is((await tag(X.beep, 16000, W))[0].name, 'Beep, bleep')
	for (let name of ['silence', 'empty']) {
		let t = await tag(X[name], 16000, { ...W, top: 1 })
		is(t[0].name, 'Silence', `${name}: Silence, as the Keras model says`)
		ok(t[0].score > 0.99, `${t[0].score}`)
	}
	is((await scores(X.empty, 16000, W)).scores.length, 1, 'nothing: one patch of zeros, as pad_waveform pads it')
})

model('tag: top, and the hop between patches', RUN, async () => {
	is(await tag(X.noise, 16000, { ...W, top: 0 }), [])
	let all = await tag(X.noise, 16000, { ...W, top: Infinity })
	is(all.length, 521, 'Infinity: all 521')
	is(new Set(all.map(t => t.index)).size, 521)
	ok(all.every((t, i) => !i || all[i - 1].score >= t.score), 'ranked')
	is(all[0].name, 'White noise')
	is((await tag(X.noise, 16000, { ...W, top: 3 })).map(t => t.name), all.slice(0, 3).map(t => t.name))
	await rejects(() => tag(X.noise, 16000, { ...W, top: -1 }), /top/)
	let s = await scores(X.lena, 16000, { ...W, hop: 2 })
	is([s.hop, s.times.length, s.times[3]], [2, REF.inputs['lena@2'].patches, 6], 'a patch every 2 s')
})

test('tag: without weights and without a hosted revision, an error that says what to pass', async () => {
	if (REVISION) return console.log('  REVISION is set: the hosted file is read without weights (tested below)')
	is(MODEL, null)
	let dir = mkdtempSync(path.join(os.tmpdir(), 'neural-tag-')), was = process.env.AUDIO_NEURAL_CACHE
	process.env.AUDIO_NEURAL_CACHE = dir
	try { await rejects(() => tag(X.beep, 16000), /not hosted yet.*opts\.weights/) }
	finally { was == null ? delete process.env.AUDIO_NEURAL_CACHE : process.env.AUDIO_NEURAL_CACHE = was; rmSync(dir, { recursive: true }) }
	await rejects(() => tag(X.beep, 0), /sampleRate/)
})

test('CLASSES: yamnet_class_map.csv, in output order', () => {
	is(CLASSES.length, 521)
	is(CLASSES[0], { id: '/m/09x0r', name: 'Speech' })
	is(CLASSES[132], { id: '/m/04rlf', name: 'Music' })
	is(CLASSES[494].name, 'Silence')
	is(new Set(CLASSES.map(c => c.id)).size, 521, 'ids unique')
})

test('model: the file tested is the one SHA256 names', () => {
	if (!FP16) return console.log('  no local copy: skipped')
	is(createHash('sha256').update(readFileSync(FP16)).digest('hex'), SHA256, FP16)
})

// The hosted file (Hugging Face, the revision pinned). Network: skipped offline or until REVISION is set
let online
const offline = async () => !(online ??= await fetch('https://huggingface.co/', { method: 'HEAD', signal: AbortSignal.timeout(8000) }).then(r => r.ok, () => false)) && (console.log('  offline: skipped'), true)
test('tag: the hosted file is pinned to its revision and SHA-256', async () => {
	if (!REVISION) return console.log('  REVISION unset (not uploaded yet): skipped')
	if (await offline()) return
	let res = await fetch(MODEL, { method: 'HEAD', redirect: 'manual' })
	is(res.headers.get('x-linked-etag')?.replace(/"/g, ''), SHA256, String(MODEL))
})

test('tag: no weights and an empty cache, the hosted copy is fetched into the cache and runs', async () => {
	if (!REVISION) return console.log('  REVISION unset (not uploaded yet): skipped')
	if (await offline()) return
	let dir = mkdtempSync(path.join(os.tmpdir(), 'neural-tag-')), was = process.env.AUDIO_NEURAL_CACHE
	process.env.AUDIO_NEURAL_CACHE = dir
	try {
		is((await tag(X.beep, 16000))[0].name, 'Beep, bleep', 'run on the fetched copy')
		is(createHash('sha256').update(readFileSync(path.join(dir, 'yamnet', 'yamnet.fp16.onnx'))).digest('hex'), SHA256, 'kept whole in the cache')
	}
	finally { was == null ? delete process.env.AUDIO_NEURAL_CACHE : process.env.AUDIO_NEURAL_CACHE = was; rmSync(dir, { recursive: true }) }
}, { timeout: 300_000 })
