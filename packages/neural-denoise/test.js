// Without downloads: RNNoise bit-exact against upstream's portable C build (fixtures/rnnoise.json, from
// scripts/rnnoise-reference.mjs), the frame API, offline alignment, the worklet's FIFO, and DeepFilterNet's
// pipeline (STFT, alignment, deep-filter taps, chunking) through an identity model. The little RNNoise
// model runs when the reference script has put it in the neural cache. With the DeepFilterNet3 export in
// the cache (the first denoise(…, { model: 'deepfilternet3' }) fetches it) the ONNX path runs; with
// scripts/deepfilter-reference.py's output there too, it is compared against Python DeepFilterNet. The music guard
// runs against its float32 reference (fixtures/guard.json, from scripts/guard.py); with VoiceBank+DEMAND's training
// utterances in ~/.cache/audiojs/data/vbdemand-train, noisy speech is checked to pass none of its frames.
import test, { ok, is, rejects, throws } from 'tst'
import { existsSync, readFileSync, readdirSync } from 'node:fs'
import { createHash } from 'node:crypto'
import os from 'node:os'
import path from 'node:path'
import { execFileSync } from 'node:child_process'
import denoise, { load, weights, rnnoise, mixback, guard, segments, KINDS, FRAME, MODEL } from './denoise.js'
import { model } from './rnnoise.js'
import { loadDeepFilter, enhance, erbWidths, voicing, bandEdge, LEVEL, FILL, FLOOR } from './deepfilter.js'
import { rfft, irfft, WINDOW, N, HOP, BINS } from './fft.js'
import { guardNet, net as guardModel, analyzer, states, patchOf, offline, online, gains, RAMP } from './guard.js'
import { inputs, sha } from './scripts/rnnoise-reference.mjs'

const CACHE = process.env.AUDIO_NEURAL_CACHE || path.join(os.homedir(), '.cache', 'audiojs', 'neural')
const FIX = JSON.parse(readFileSync(new URL('./fixtures/rnnoise.json', import.meta.url)))
const LITTLE = path.join(CACHE, 'rnnoise', 'rnnoise-little.bin')
const HAS_DFN = existsSync(path.join(CACHE, createHash('sha256').update(MODEL).digest('hex')))
const DFN_REF = path.join(CACHE, 'deepfilternet3', 'reference')
const IN = inputs()
const f32 = file => { let b = readFileSync(file); return new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4) }
const snr = (ref, x) => { let s = 0, e = 0; for (let i = 0; i < ref.length; i++) { s += ref[i] ** 2; e += (ref[i] - x[i]) ** 2 } return 10 * Math.log10(s / e) }
const maxDiff = (a, b) => { let m = 0; for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i])); return m }
const MODEL_RUN = { timeout: 300000 }
// mixback() on the fixture take below, as the published 0.8.0 (the default) and 0.7.0 (`everywhere`) give it: their
// SHA-256, printed by `node scripts/floor.mjs fixture <that install's denoise.js>`
const DEFAULT = 'c05f775d9e550f82e1095f653c5dd45e16aaaef6374b40051c608bfdad47ee13', DEFAULT_6_30 = 'd7fbc9e4947bccbccbfec0cbba4709170c95d833be486d232f6874a0aa6192f8'
const EVERYWHERE = 'a83a99ede12e517987b9b966e43ca9267d0d974a2d89d3d4e56c6c555f373f23', EVERYWHERE_6_30 = '81ff49c7863c27f786f2ddea0cd575d271c2ae68c65fbcba0c1c144e311c7192'

// frames through the frame API → { out, vad } at int16 scale
function frames(net, x) {
	let st = rnnoise(net), out = new Float32Array(x.length), vad = new Float32Array(x.length / FRAME)
	for (let i = 0, k = 0; i < x.length; i += FRAME, k++) vad[k] = st.process(x.subarray(i, i + FRAME), out.subarray(i, i + FRAME))
	return { out, vad }
}

// ------------------------------------------------ fft.js

test('fft: 960-point transform against a direct DFT, perfect reconstruction, power-complementary window', () => {
	let s = 7, x = Float32Array.from({ length: N }, () => (s = (s * 1664525 + 1013904223) % 4294967296, s / 4294967296 - .5))
	let Xr = new Float32Array(BINS), Xi = new Float32Array(BINS), err = 0
	rfft(x, Xr, Xi)
	for (let k = 0; k < BINS; k++) {
		let re = 0, im = 0
		for (let n = 0; n < N; n++) { re += x[n] * Math.cos(2 * Math.PI * k * n / N) / N; im -= x[n] * Math.sin(2 * Math.PI * k * n / N) / N }
		err = Math.max(err, Math.hypot(Xr[k] - re, Xi[k] - im))
	}
	ok(err < 1e-7, `max bin error ${err.toExponential(2)}`)
	let y = new Float32Array(N)
	irfft(Xr, Xi, y)
	ok(maxDiff(x, y) < 1e-6, `inverse(forward(x)) = x to ${maxDiff(x, y).toExponential(2)}`)
	let w = 0
	for (let i = 0; i < HOP; i++) w = Math.max(w, Math.abs(WINDOW[i] ** 2 + WINDOW[i + HOP] ** 2 - 1))
	ok(w < 1e-6, `w[n]² + w[n + 480]² = 1 to ${w.toExponential(2)}`)
})

// ------------------------------------------------ RNNoise

test('rnnoise: the bundled weights are the blob upstream dump_weights_blob writes for model 0a8755f8', async () => {
	let b = await weights()
	is(b.length, 3544320)
	is(sha(b), FIX.blob)
	let net = model(b)
	is([net.conv1.nIn, net.conv1.nOut, net.conv2.nOut, net.gru1.input.nOut, net.dense.nIn, net.dense.nOut], [195, 128, 384, 1152, 1536, 32])
})

test('rnnoise: bit-exact to upstream C (portable path, -ffp-contract=off), with the wasm SIMD products and without', async () => {
	let bytes = await weights()
	for (let simd of [true, false]) {
		let net = model(bytes, { simd }), path = net.gru1.input.simd ? 'wasm' : 'js'
		ok(simd ? path === 'wasm' : path === 'js', `int8 products in ${path}`)
		for (let [name, x] of Object.entries(IN)) {
			let { out, vad } = frames(net, x)
			is(x.length / FRAME, FIX.cases[name].frames, `${name}: frames`)
			is(sha(out), FIX.cases[name].regular.out, `${path} ${name}: output`)
			is(sha(vad), FIX.cases[name].regular.vad, `${path} ${name}: VAD`)
		}
	}
})

test('rnnoise: the little model, bit-exact too (skipped without the blob scripts/rnnoise-reference.mjs caches)', () => {
	if (!existsSync(LITTLE)) return console.log('  (no little model in the cache: skipped)')
	let net = model(readFileSync(LITTLE))
	is(sha(readFileSync(LITTLE)), FIX.little)
	for (let [name, x] of Object.entries(IN)) {
		let { out, vad } = frames(net, x)
		is(sha(out), FIX.cases[name].little.out, `${name}: output`)
		is(sha(vad), FIX.cases[name].little.vad, `${name}: VAD`)
	}
})

test('rnnoise: run(), the frame with no number returned (real-time loops), bit-exact to upstream C too, its voice probability in voice[0]', async () => {
	let net = model(await weights())
	for (let [name, x] of Object.entries(IN)) {
		let st = rnnoise(net), out = new Float32Array(x.length), vad = new Float32Array(x.length / FRAME)
		for (let i = 0, k = 0; i < x.length; i += FRAME, k++) { st.run(x.subarray(i, i + FRAME), out.subarray(i, i + FRAME)); vad[k] = st.voice[0] }
		is(sha(out), FIX.cases[name].regular.out, `${name}: output`)
		is(sha(vad), FIX.cases[name].regular.vad, `${name}: VAD`)
	}
})

// Upstream's rnnoise_process_frame starts each frame at vad_prob = 0, and only compute_rnn, which a silent frame (E <
// 0.04) skips, sets it (xiph/rnnoise 70f1d25, src/denoise.c:467, :474-476, :503). The fixtures' digital silence comes
// before any voice, where a probability left over from the frame before is 0 too; here it follows two seconds of lena.
// From 0.3 s into it the biquad's ring and the window are long gone and every frame is silent.
test('rnnoise: digital silence after speech: from 0.3 s into it each frame\'s voice probability is 0, from process() and in voice[0]', async () => {
	let net = model(await weights()), x = new Float32Array(3 * 48000), a = rnnoise(net), b = rnnoise(net), y = new Float32Array(FRAME), most = 0, left = []
	x.set(IN.lena.subarray(0, 2 * 48000))
	for (let i = 0; i < x.length; i += FRAME) {
		let f = x.subarray(i, i + FRAME), p = a.process(f, y)
		b.run(f, y)
		if (i < 2 * 48000) most = Math.max(most, p)
		else if (i >= 2.3 * 48000 && (p !== 0 || b.voice[0] !== 0)) left.push(i / FRAME)
	}
	ok(most > .5, `the speech before it: a voice probability up to ${most.toFixed(2)}`)
	is(left.length, 0, `silent frames with a probability left over: ${left.slice(0, 5).join(', ') || 'none'}`)
})

test('rnnoise: malformed weights throw', async () => {
	let b = await weights()
	throws(() => model(b.subarray(0, 100)), /truncated|bad weight record/)
	let bad = b.slice()
	new DataView(bad.buffer).setInt32(12 + 64 + 99840, 508, true) // conv1_bias record: 508 bytes instead of 512
	throws(() => model(bad), /conv1_bias/)
	throws(() => model(new Uint8Array(0)), /missing/)
})

test('denoise: rnnoise offline, limit Infinity = the frame API with its 960-sample delay removed; same length; a limit mixes the input back, 16 dB by default', async () => {
	let x = IN['lena+noise'].map(v => v / 32768), net = model(await weights())
	let pad = new Float32Array(Math.ceil((x.length + 960) / FRAME) * FRAME)
	pad.set(IN['lena+noise'])
	let { out } = frames(net, pad), y = await denoise(x, { sampleRate: 48000, limit: Infinity })
	is(y.length, x.length)
	is(maxDiff(y, out.subarray(960, 960 + x.length).map(v => v / 32768)), 0, 'aligned sample for sample')
	for (let [opts, db] of [[{ limit: 6 }, 6], [{}, 16]]) {
		let z = await denoise(x, { sampleRate: 48000, ...opts }), lim = 10 ** (-db / 20)
		ok(maxDiff(z, y.map((v, i) => v * (1 - lim) + x[i] * lim)) < 1e-6, `${opts.limit ? 'limit' : 'default'} ${db} dB: ${lim.toFixed(2)} of the input mixed in`)
	}
})

// limit 0 takes nothing off: its share, 10^(−0/20) = 1, mixes the input back whole and RNNoise's output times 0, so at
// 48 kHz denoise() gives the input itself; at another rate the input as resample-sinc carries it to 48 kHz and back,
// as denoise() resamples (music 'enhance', so that the guard blends nothing in). A share that is not 1 at 0 fails it.
// A limit that is no number is refused rather than read as some setting
test('denoise: limit 0 takes nothing off: RNNoise offline gives the input at 48 kHz, at 44.1 kHz the input through resample-sinc and back; a limit that is no number, a TypeError', async () => {
	let x = IN['lena+noise'].subarray(0, 2 * 48000).map(v => v / 32768), m = await load(), { default: resample } = await import('@audio/resample-sinc')
	let same = (a, b) => a.length === b.length && a.every((v, i) => Object.is(v, b[i]))
	ok(same(await denoise(x, { sampleRate: 48000, model: m, limit: 0 }), x), '48 kHz: the input, sample for sample')
	let a = x.subarray(0, 44100), back = resample(resample(Float32Array.from(a), { from: 44100, to: 48000 }), { from: 48000, to: 44100 }).subarray(0, a.length)
	ok(same(await denoise(a, { sampleRate: 44100, model: m, limit: 0, music: 'enhance' }), back), `44.1 kHz: the input through resample-sinc to 48 kHz and back, sample for sample (${snr(a, back).toFixed(1)} dB SNR from the input)`)
	for (let limit of [NaN, '16']) await rejects(() => denoise(x, { sampleRate: 48000, model: m, limit }), /^TypeError: .*limit is dB/, `limit ${typeof limit === 'string' ? `'${limit}'` : limit}: a TypeError`)
})

test('denoise: shapes (Float32Array, channels, { channelData, sampleRate }) and rates (44.1 kHz in and out)', async () => {
	let a = IN.lena.subarray(0, 44100).map(v => v / 32768), b = IN['lena+noise'].subarray(0, 44100).map(v => v / 32768)
	let m = await load()
	let one = await denoise(a, { sampleRate: 44100, model: m })
	ok(one instanceof Float32Array && one.length === 44100, 'mono')
	let two = await denoise([a, b], { sampleRate: 44100, model: m })
	ok(Array.isArray(two) && two.length === 2 && two[1].length === 44100, 'channels')
	is(maxDiff(two[0], one), 0, 'channels independent')
	let obj = await denoise({ channelData: [a], sampleRate: 44100 }, { model: m })
	ok(obj.sampleRate === 44100 && obj.channelData[0].length === 44100, 'decode() result')
})

test('denoise: input errors', async () => {
	await rejects(() => denoise(new Float32Array(10), {}), /sampleRate/)
	await rejects(() => denoise('x', { sampleRate: 48000 }), /audio must be/)
	await rejects(() => denoise(new Float32Array(10), { sampleRate: 48000, model: 'demucs' }), /unknown model/)
})

test('worklet: 128-sample quanta, output = the frame API delayed by 448 samples (1408 in all), limit on the dry path, 16 dB by default, 0 the input', async () => {
	let procs = {}
	globalThis.sampleRate = 48000
	globalThis.AudioWorkletProcessor = class {}
	globalThis.registerProcessor = (name, cls) => procs[name] = cls
	await import('./worklet.js')
	let bytes = await weights(), x = IN['lena+noise'].subarray(0, 48000)
	let { out } = frames(model(bytes), x), want = new Float32Array(x.length)
	for (let i = 448; i < x.length; i++) want[i] = out[i - 448] / 32768
	for (let limit of [Infinity, 12, undefined, 0]) {
		let node = new procs['neural-denoise']({ processorOptions: { weights: bytes, limit } }), y = new Float32Array(x.length)
		for (let i = 0; i < x.length; i += 128) {
			let q = x.subarray(i, i + 128).map(v => v / 32768)
			node.process([[q]], [[y.subarray(i, i + 128)]])
		}
		let db = limit ?? 16, lim = 10 ** (-db / 20)
		let exp = want.map((v, i) => v * (1 - lim) + (i >= 1408 ? x[i - 1408] / 32768 : 0) * lim)
		ok(maxDiff(y, exp) < 1e-7, `limit ${limit ?? 'unset, 16'}: max |d| ${maxDiff(y, exp).toExponential(1)}`)
		// 0 takes nothing off: the input mixed back whole, the model's output times 0
		if (limit === 0) is(maxDiff(y, Float32Array.from(y, (_, i) => i >= 1408 ? x[i - 1408] / 32768 : 0)), 0, 'limit 0: the input itself, 1408 samples late')
	}
	throws(() => new procs['neural-denoise']({ processorOptions: { weights: bytes, limit: NaN } }), /^TypeError: .*limit is dB/, 'a limit that is no number')
	throws(() => { globalThis.sampleRate = 44100; new procs['neural-denoise']({ processorOptions: { weights: bytes } }) }, /48 kHz/)
	globalThis.sampleRate = 48000
})

// ------------------------------------------------ audio.js manifest

// the contract atom hosted as hosts run it: blocks of `size(k)` samples, then its declared latency of silence
async function hosted(sr, chs, limit, size, music) {
	let { rnnoise: atom } = await import('./audio.js')
	let D = atom.latency({ sampleRate: sr, params: {} }), params = { limit: new Float32Array([limit]), music }
	let proc = atom({ sampleRate: sr, maxBlockSize: 4096, maxChannels: 32, params }), n = chs[0].length + D
	let inp = chs.map(x => { let v = new Float32Array(n); v.set(x); return v }), out = chs.map(() => new Float32Array(n))
	for (let p = 0, k = 0, m; p < n; p += m) { m = Math.min(n - p, size(k++)); proc([inp.map(v => v.subarray(p, p + m))], [out.map(v => v.subarray(p, p + m))], params) }
	return { D, out: out.map(v => v.subarray(D)) }
}

test('manifest: rnnoise streams denoise() sample for sample at 48 kHz, 1439 samples late (960 + a 479-sample queue), in blocks of any size', async () => {
	let x = IN['lena+noise'].subarray(0, 3 * 48000).map(v => v / 32768), s = 3
	let rnd = () => (s = (s * 1664525 + 1013904223) % 4294967296, s / 4294967296)
	for (let limit of [20, 100, 0]) {
		let want = await denoise(x, { sampleRate: 48000, limit })
		for (let [name, size] of [['1024', () => 1024], ['128', () => 128], ['1 to 3000', () => 1 + Math.floor(rnd() * 3000)], ['1', () => 1]]) {
			let { D, out: [y] } = await hosted(48000, [x], limit, size)
			is(D, 1439)
			is(maxDiff(y, want), 0, `limit ${limit}, blocks of ${name}: equal`)
		}
	}
})

test('manifest: other rates resample in and out; the output follows denoise() behind the declared delay; channels apart', async () => {
	let a = IN['lena+noise'].subarray(0, 3 * 44100).map(v => v / 32768), b = IN.lena.subarray(0, 3 * 44100).map(v => v / 32768)
	let { D, out: [y, z] } = await hosted(44100, [a, b], 20, () => 1000)
	is(D, 1354)
	let cut = a.length - 2205, want = (await denoise(a, { sampleRate: 44100, limit: 20 })).subarray(0, cut)
	// audio.js streams resample-sinc 1.2.0's arithmetic; denoise() resamples with it offline: the same samples
	// but where the offline resampler sees the end (the last 50 ms left out)
	is(maxDiff(y.subarray(0, cut), want), 0, 'equal to denoise()')
	let { out: [alone] } = await hosted(44100, [b], 20, () => 1000)
	is(maxDiff(z, alone), 0, 'each channel on its own')
	is((await import('./audio.js')).rnnoise.latency({ sampleRate: 8000 }), 271, '8 kHz: 33.9 ms')
})

// What process allocates, counted as @audio/compile's validator counts it (tools/verify.js taken(), its child's
// flags), in a child Node: what V8's young generation and the stores of typed arrays take in between two forced minor
// collections, and what goes straight into the old generation. One instance warms the code; a second is made, as a host
// holding two builds it, its first block counted (made there, two channels' state took 1.05 MB, their arrays' stores),
// then, once V8 has optimized what the new stream is first to reach, its blocks: 2^15 of 8 frames, the less of two runs.
// Each held to the validator's line, 6144 bytes a run (3/64 of a byte a block over its 2^17, kept by its readings of
// 2^15), the old generation to 64 KB.
test('manifest: process allocates nothing, from a second instance\'s first block on (V8\'s heap and the typed arrays\' stores, as @audio/compile\'s validator counts them)', async () => {
	let B = 1 << 15, LINE = 6144, OLD = 1 << 16, child = `
		import v8 from 'node:v8'
		if (!v8.GCProfiler) { console.log(JSON.stringify({ na: 'no v8.GCProfiler before Node 18.15' })); process.exit() }
		let { rnnoise } = await import(${JSON.stringify(new URL('./audio.js', import.meta.url).href)})
		let F = 8, B = ${B}, params = { limit: Float32Array.of(16), music: 'pass' }
		let make = () => rnnoise({ sampleRate: 48000, maxBlockSize: 128, maxChannels: 2, params })
		let s = 7, noise = () => Float32Array.from({ length: 64 * F }, () => (s = (Math.imul(s, 1664525) + 1013904223) >>> 0, .3 * (s / 2 ** 32 - .5)))
		let src = [noise(), noise()], ins = Array.from({ length: 64 }, (_, k) => [src.map(c => c.subarray(k * F, k * F + F))])
		let out = [[new Float32Array(F), new Float32Array(F)]], run = (p, n) => { for (let i = 0; i < n; i++) p(ins[i & 63], out, params) }
		let used = (g, sp) => g.heapSpaceStatistics.reduce((a, q) => sp.includes(q.spaceName) ? a + q.spaceUsedSize : a, 0)
		let taken = f => {
			let prof = new v8.GCProfiler(), r = { young: 0, old: 0, stores: 0 }
			prof.start(); gc({ type: 'minor' }); f(); gc({ type: 'minor' })
			let g = prof.stop().statistics
			for (let i = 1; i < g.length; i++) {
				let a = g[i - 1].afterGC, b = g[i].beforeGC
				r.young += used(b, ['new_space', 'new_large_object_space']) - used(a, ['new_space', 'new_large_object_space'])
				r.old += Math.max(0, used(b, ['old_space', 'large_object_space']) - used(a, ['old_space', 'large_object_space']))
				r.stores += b.heapStatistics.externalMemory - a.heapStatistics.externalMemory
			}
			return r
		}
		run(make(), B)
		let b = make(), first = taken(() => run(b, 1))
		run(b, B)
		console.log(JSON.stringify({ first, runs: [taken(() => run(b, B)), taken(() => run(b, B))] }))`
	let flags = ['--expose-gc', '--no-concurrent-recompilation', '--no-allocation-site-pretenuring', '--no-concurrent-array-buffer-sweeping']
	let r = JSON.parse(execFileSync(process.execPath, [...flags, '--input-type=module', '-e', child], { encoding: 'utf8' }))
	if (r.na) return console.log(`  (${r.na}: skipped)`)
	let heap = x => x.young + x.stores, run = r.runs.reduce((a, x) => heap(x) < heap(a) ? x : a)
	ok(heap(r.first) <= LINE && r.first.old <= OLD, `the first block of a second instance: ${heap(r.first)} bytes, ${r.first.old} old`)
	ok(heap(run) <= LINE && run.old <= OLD, `${heap(run)} bytes in ${B} blocks of 8 frames (${(heap(run) / B).toFixed(4)} a block), ${run.old} old`)
})

// The guard's frame counter first wraps at frame 408, 4.08 s into a stream. While the wrap was a branch, its first
// taking had no type feedback and deoptimized the function there: 1.2 to 2.4 MB at 4.0 to 4.5 s of every process's
// first stream, on Node 22 and 25, at 8 to 96 kHz. Counted as the test above counts, in a child Node: one instance's
// first stream at 48 kHz in blocks of 8 frames, six windows of half a second from 3 s on, each within the validator's
// line, 6144 bytes a run, the old generation within 64 KB.
test('manifest: no burst of allocation 3 to 6 s into the first stream, where the guard\'s frame counter first wraps (4.08 s)', { timeout: 120000 }, async () => {
	let LINE = 6144, OLD = 1 << 16, child = `
		import v8 from 'node:v8'
		if (!v8.GCProfiler) { console.log(JSON.stringify({ na: 'no v8.GCProfiler before Node 18.15' })); process.exit() }
		let { rnnoise } = await import(${JSON.stringify(new URL('./audio.js', import.meta.url).href)})
		let F = 8, params = { limit: Float32Array.of(16), music: 'pass' }, p = rnnoise({ sampleRate: 48000, maxBlockSize: 128, maxChannels: 2, params })
		let s = 7, noise = () => Float32Array.from({ length: 64 * F }, () => (s = (Math.imul(s, 1664525) + 1013904223) >>> 0, .3 * (s / 2 ** 32 - .5)))
		let src = [noise(), noise()], ins = Array.from({ length: 64 }, (_, k) => [src.map(c => c.subarray(k * F, k * F + F))])
		let out = [[new Float32Array(F), new Float32Array(F)]], k = 0, run = n => { for (let e = k + n; k < e; k++) p(ins[k & 63], out, params) }
		let used = (g, sp) => g.heapSpaceStatistics.reduce((a, q) => sp.includes(q.spaceName) ? a + q.spaceUsedSize : a, 0)
		let taken = f => {
			let prof = new v8.GCProfiler(), r = { young: 0, old: 0, stores: 0 }
			prof.start(); gc({ type: 'minor' }); f(); gc({ type: 'minor' })
			let g = prof.stop().statistics
			for (let i = 1; i < g.length; i++) {
				let a = g[i - 1].afterGC, b = g[i].beforeGC
				r.young += used(b, ['new_space', 'new_large_object_space']) - used(a, ['new_space', 'new_large_object_space'])
				r.old += Math.max(0, used(b, ['old_space', 'large_object_space']) - used(a, ['old_space', 'large_object_space']))
				r.stores += b.heapStatistics.externalMemory - a.heapStatistics.externalMemory
			}
			return r
		}
		run(3 * 6000)
		let windows = []
		for (let w = 0; w < 6; w++) windows.push(taken(() => run(3000)))
		console.log(JSON.stringify({ windows }))`
	let flags = ['--expose-gc', '--no-concurrent-recompilation', '--no-allocation-site-pretenuring', '--no-concurrent-array-buffer-sweeping']
	let r = JSON.parse(execFileSync(process.execPath, [...flags, '--input-type=module', '-e', child], { encoding: 'utf8', timeout: 120000 }))
	if (r.na) return console.log(`  (${r.na}: skipped)`)
	let heap = x => x.young + x.stores
	ok(r.windows.every(x => heap(x) <= LINE && x.old <= OLD), `bytes in each half second from 3 s: ${r.windows.map(heap).join(', ')} (old ${r.windows.map(x => x.old).join(', ')})`)
})

test('manifest: a call longer than ctx.maxBlockSize or wider than ctx.maxChannels throws a RangeError naming what it carried and the bound; a ctx without either, a TypeError naming it', async () => {
	let { rnnoise: atom } = await import('./audio.js'), params = { limit: Float32Array.of(16), music: 'pass' }
	let p = atom({ sampleRate: 48000, maxBlockSize: 128, maxChannels: 2, params }), chans = (c, n) => [Array.from({ length: c }, () => new Float32Array(n))]
	p(chans(2, 128), chans(2, 128), params)
	throws(() => p(chans(1, 129), chans(1, 129), params), /^RangeError: .*129 frames.*maxBlockSize is 128/, 'a call of 129 frames, 128 the most')
	throws(() => p(chans(3, 64), chans(3, 64), params), /^RangeError: .*3 channels.*maxChannels is 2/, 'a call of 3 channels, 2 the most')
	throws(() => atom({ sampleRate: 48000, maxChannels: 2, params }), /^TypeError: .*maxBlockSize/, 'no maxBlockSize')
	throws(() => atom({ sampleRate: 48000, maxBlockSize: 128, params }), /^TypeError: .*maxChannels/, 'no maxChannels')
})

// A host here, so that the tests below run wherever this file does. Of the duties contract 1.0 lists under What a host
// owes it keeps: (1) the instance built before its calls, with `params` (the values at frame 0, Context); (3) values
// changed only between calls: the call split at each of `moves` (a list of [frame, values], in order), each call handed
// the values at its first frame, a number as a Float32Array of one it keeps and writes in place (Parameters, Values),
// and a number that declares `smoothing` ramped from the value in force over that many seconds in calls of 32 frames at
// most, as toStream ramps it (limit and music declare none, so they jump); (5) calls of `block` frames, its
// maxBlockSize, or of cut(i) at frame i held to it, bus 0 carrying the input's channels (the inputs here are finite,
// so it makes no NaN 0); (8) the latency taken at construction and, with `tail`, the tail declared at the values in
// force at the end rendered, as toStream's end() renders it. It leaves out (2), clamping and the check of an enum's
// values: the tests give values in range; (7), ctx.transport, which rnnoise does not read; and those with nothing to
// act on here: (4) restart, (6) notes, (9) state, (10) a real-time host's, (11) a plugin's. It renders the input, the
// latency's frames of silence, the tail's, then `more` frames of silence. hosting() makes one call a step, its first
// step building the instance, so that a test can call two instances in turn; host() takes every step.
// → { y: each channel aligned by the latency, L: the latency, T: the tail's frames }
function* hosting(m, x, { params = {}, moves = [], block = 128, cut, tail = false, more = 0, sampleRate: sr = 48000 } = {}) {
	let spec = m.params, set = {}, ramp = {}
	for (let k in spec) set[k] = params[k] ?? spec[k].default
	// each value in force, the last one handed, a number as it was before its Float32Array rounded it
	let cur = { ...set }, live = valuesOf(spec, set), setting = v => ({ sampleRate: sr, params: valuesOf(spec, v), buffers: {} })
	let L = typeof m.latency === 'function' ? m.latency(setting(set)) : m.latency ?? 0
	let T = tail ? Math.round(tailOf(m, setting(Object.assign({ ...set }, ...moves.map(([, v]) => v)))) * sr) : 0
	let n = x[0].length, N = n + L + T + more, inp = x.map(c => { let v = new Float32Array(N); v.set(c); return v }), out = x.map(() => new Float32Array(N))
	let proc = m({ sampleRate: sr, maxBlockSize: block, maxChannels: x.length, params: setting(set).params, buffers: {} })
	// a ramp [from, to, its first frame, its frames]: its value at frame i
	let along = ([a, b, at, len], i) => i - at < len ? b + (a - b) * (1 - (i - at) / len) : b
	yield
	for (let i = 0, j = 0, k; i < N; i += k) {
		for (; j < moves.length && moves[j][0] <= i; j++) for (let [p, v] of Object.entries(moves[j][1])) {
			let len = spec[p].smoothing > 0 ? Math.max(1, Math.round(spec[p].smoothing * sr)) : 0
			if (len) ramp[p] = [cur[p], v, i, len]
			else if (spec[p].type === 'number') live[p][0] = cur[p] = v
			else live[p] = cur[p] = v
		}
		k = Math.min(N - i, block, cut ? cut(i) : block)
		if (j < moves.length) k = Math.min(k, moves[j][0] - i)
		for (let p in ramp) {
			let [, , at, len] = ramp[p]
			live[p][0] = cur[p] = along(ramp[p], i)
			if (i - at < len) k = Math.min(k, 32, at + len - i)
		}
		proc([inp.map(c => c.subarray(i, i + k))], [out.map(c => c.subarray(i, i + k))], live)
		yield
	}
	return { y: out.map(c => c.subarray(L)), L, T }
}
const host = (m, x, opts) => { let h = hosting(m, x, opts), r; while (!(r = h.next()).done); return r.value }
// a setting's values as a host hands them: a number as a Float32Array of one
const valuesOf = (spec, v) => Object.fromEntries(Object.keys(spec).map(k => [k, spec[k].type === 'number' ? Float32Array.of(v[k]) : v[k]]))
// audio's batch.js, the host `audio` renders with: the audio package, else the audio checkout beside @audio, as
// @audio/compile's validator finds its host. Where neither is, the tests below run on host() alone
const HOST = await import('audio/batch').catch(() => import(new URL('../../../../audio/batch.js', import.meta.url).href)).catch(() => null)
// host()'s render through toStream: calls of `block` frames, the input written in parts that start at each of `moves`,
// each part with its values, the latency's frames of nothing, end() with `tail`, then `more` frames of nothing
function streamed(m, x, { params = {}, moves = [], block = 128, tail = false, more = 0, sampleRate: sr = 48000 } = {}) {
	let s = HOST.toStream(m, { sampleRate: sr, maxBlockSize: block, params }), n = x[0].length, parts = [], T = 0
	let add = r => { if (r) parts.push(r instanceof Float32Array ? [r] : r) }
	for (let j = 0, a = 0; j <= moves.length; j++) {
		let b = j < moves.length ? Math.min(moves[j][0], n) : n
		if (b > a) add(s.write(x.map(c => c.subarray(a, b)), moves[j - 1]?.[1]))
		a = b
	}
	add(s.write(s.latency))
	if (tail) { let t = s.end(); add(t); T = t ? (t instanceof Float32Array ? t : t[0]).length : 0 }
	if (more) add(s.write(more))
	let y = x.map((_, c) => { let v = new Float32Array(parts.reduce((k, p) => k + p[c].length, 0)), o = 0; for (let p of parts) { v.set(p[c], o); o += p[c].length } return v.subarray(s.latency) })
	return { y, L: s.latency, T }
}
// a setting rendered by host(), and by audio's toStream where it is found: the same samples, the same tail
function played(m, x, opts, label) {
	let a = host(m, x, opts)
	if (HOST) {
		let b = streamed(m, x, opts), same = a.T === b.T && a.y.every((c, k) => c.length === b.y[k].length && c.every((v, i) => Object.is(v, b.y[k][i])))
		ok(same, `${label}: audio's toStream renders the same samples${a.T === b.T ? '' : `, its tail ${b.T} frames, here ${a.T}`}`)
	}
	return a
}
const rmsOf = (a, from, to) => { let s = 0; for (let i = from; i < to; i++) s += a[i] * a[i]; return Math.sqrt(s / (to - from)) }
// a manifest's tail as a host reads it: seconds, or a function of the setting
const tailOf = (m, setting) => typeof m.tail === 'function' ? m.tail(setting) : m.tail ?? 0

// The contract's tail (contract 1.0, Latency, tail): the seconds after the input stops until the output stays 60 dB
// under the input's RMS level, which a host goes on rendering. What rings is RNNoise's DC-blocking biquad (upstream's
// rnn_biquad, a = [−1.99599, 0.996]: xiph/rnnoise 70f1d25, src/denoise.c:469, :409-419), computed here alone: after a
// DC step it stays over 60 dB down for `dc` samples; after a tone of 16.3 Hz, the one the declared tail rests on
// (audio.js: 1.4183 of its peak), stopping at the phase of the 16 tried after which it rings longest, for `rings[q]`
// against the tone's RMS, what the tail at the top, 100, covers, within 2 ms. Through the processor at the default limit
// and at 100, at 48 and 44.1 kHz: that tone at that phase, quiet enough to pass ungated, a DC step at −46 dBFS and the
// validator's 312 Hz tone of 0.5. The host renders the declared tail; past it every window of 64 frames is 60 dB under
// the input's RMS (the validator's reading); the tone is over it in the 2 ms before the tail, so a tail 2 ms shorter
// fails; the step rings past half of the tail
test('manifest: its tail covers the ring of RNNoise\'s DC-blocking biquad after the input stops, the longest to within 2 ms, and a host renders it', async () => {
	let { rnnoise: atom } = await import('./audio.js'), a0 = Math.fround(-1.99599), a1 = Math.fround(.996)
	// the biquad alone on x: the samples after x's first n until |y| stays at or under `floor`
	let biquad = (x, n, floor) => {
		let m0 = 0, m1 = 0, ring = 0
		for (let i = 0; i < x.length; i++) { let y = x[i] + m0; m0 = m1 + (-2 * x[i] - a0 * y); m1 = x[i] - a1 * y; if (i >= n && Math.abs(y) > floor) ring = i - n + 1 }
		return ring
	}
	// n samples at rate sr of a 16.3 Hz tone of amplitude A ending at phase qπ/8, then 0.1 s of nothing
	let tone = (sr, n, A, q) => Float32Array.from({ length: n + sr / 10 }, (_, i) => i < n ? A * Math.sin(2 * Math.PI * 16.3 * (i - n) / sr + Math.PI * q / 8) : 0)
	let dc = biquad(Float32Array.from({ length: 96000 }, (_, i) => i < 48000 ? 1 : 0), 48000, 1e-3)
	let rings = Array.from({ length: 16 }, (_, q) => biquad(tone(48000, 28800, 1, q), 28800, 1e-3 / Math.SQRT2)), q = rings.indexOf(Math.max(...rings))
	let T0 = tailOf(atom, { sampleRate: 48000, params: { limit: Float32Array.of(100) } })
	ok(T0 * 48000 >= dc && T0 * 48000 < 2 * dc, `limit 100: ${(T0 * 1e3).toFixed(1)} ms declared; the biquad alone 60 dB under a DC step's end after ${(dc / 48).toFixed(1)} ms`)
	ok(T0 * 48000 >= rings[q] && T0 - rings[q] / 48000 < .002, `limit 100: ${(T0 * 1e3).toFixed(2)} ms declared; the biquad alone 60 dB under the RMS of a 16.3 Hz tone stopping at phase ${q}π/8 after ${(rings[q] / 48).toFixed(2)} ms, the longest of the 16`)
	for (let sr of [48000, 44100]) for (let limit of [16, 100]) for (let [what, hold, x, kind] of [
		[`a 16.3 Hz tone of 0.002 stopping at phase ${q}π/8`, .6, n => tone(sr, n, .002, q).subarray(0, n), 'longest'],
		['a DC step at −46 dBFS', .3, n => new Float32Array(n).fill(.005), 'step'],
		['a 312 Hz tone of 0.5', .25, n => Float32Array.from({ length: n }, (_, i) => .5 * Math.sin(2 * Math.PI * 312 * i / sr)), '']]) {
		let n = Math.round(hold * sr), xs = x(n), label = `${sr} Hz, limit ${limit}, ${what}`, T = tailOf(atom, { sampleRate: sr, params: { limit: Float32Array.of(limit) } })
		let { y: [y], T: frames } = played(atom, [xs], { sampleRate: sr, params: { limit }, tail: true, more: Math.round(.2 * sr) }, label)
		ok(T > 0 && frames === Math.round(T * sr), `${label}: the host renders ${frames} frames, the tail ${(T * 1e3).toFixed(2)} ms`)
		let floor = 1e-3 * rmsOf(xs, 0, n), last = n
		let loud = s => { for (let i = n + Math.ceil(s * sr); i + 64 <= y.length; i += 64) if (rmsOf(y, i, i + 64) > floor) return i; return -1 }
		for (let i = n; i + 64 <= y.length; i += 64) if (rmsOf(y, i, i + 64) > floor) last = i + 64
		is(loud(T), -1, `${label}: 60 dB under the input's RMS from the tail on; the ring ${((last - n) / sr * 1e3).toFixed(1)} ms`)
		if (kind === 'longest') ok(loud(T - .002) >= 0, `${label}: over it in the 2 ms before the tail`)
		if (kind === 'step') ok(2 * (last - n) > T * sr, `${label}: the tail within twice its ring`)
	}
})

// limit declares no smoothing, so a host hands its jump at the call where it comes, the call split there. The share of
// the input mixed back then moves in a straight line over 20 ms of samples from that call's first, and each input
// sample is mixed at the share it came with. At 48 kHz, where the mix is made sample for sample, the output is the two
// renders held at the limits either side of the jump mixed by that line: the first itself before the call, the second
// itself from the line's last sample, and between, with music 'enhance' (the output is then the mix itself), within half
// an ulp of the output's rounding to float32 and half an ulp of the held renders' (2^-23 of the larger), so never
// louder than both. At 44.1 kHz the converter back reads its taps ahead: the first render itself up to that many
// frames before the call, the second itself from as many after the line's end, and on these tones over the move never
// louder than the louder. A share that moves at the next 48 kHz frame instead of the call's first sample, a sample
// late, or over a line not 20 ms long at the host's rate, fails it
test('manifest: limit glides: a jump moves the output along a 20 ms line between the two limits held, a latency after the call, never louder than both', async () => {
	let { rnnoise: atom } = await import('./audio.js'), n = 49152, at = 22016
	for (let sr of [48000, 44100]) for (let [from, to, hz] of [[1, 100, 220], [16, 100, 1000]]) {
		let x = [Float32Array.from({ length: n }, (_, i) => .5 * Math.sin(2 * Math.PI * hz * i / sr))], label = `${sr} Hz, a ${hz} Hz tone, music 'enhance', limit ${from} → ${to}`
		let play = (limit, more) => played(atom, x, { sampleRate: sr, params: { limit, music: 'enhance' }, ...more }, more ? label : `${label}, held at ${limit}`).y[0]
		let A = play(from), B = play(to), y = play(from, { moves: [[at, { limit: to }]] })
		// at 44.1 kHz the converter back reads its Lanczos taps ahead: resample-sinc's 16 a side widened by 48/44.1 at
		// 48 kHz, 18, which are 16.5 at 44.1 kHz
		let N = Math.round(.02 * sr), ahead = sr === 48000 ? 0 : Math.ceil(Math.ceil(16 * 48000 / sr) * sr / 48000), first = -1, last = -1, off = 0, peak = 0, held = 0
		for (let i = 0; i < n; i++) {
			if (first < 0 && !Object.is(y[i], A[i])) first = i
			if (!Object.is(y[i], B[i])) last = i
			let w = Math.min(1, Math.max(0, (i - at + 1) / N)), M = Math.max(Math.abs(A[i]), Math.abs(B[i]))
			if (Math.abs(y[i] - (A[i] + w * (B[i] - A[i]))) > 2 ** -23 * M) off++
			if (i >= at - ahead && i < at + N + ahead) { peak = Math.max(peak, Math.abs(y[i])); held = Math.max(held, M) }
		}
		ok(first >= at - ahead, `${label}: the output moves from ${first - at} frames after the call, aligned by the latency (${ahead} read ahead)`)
		ok(last < at + N - 1 + ahead, `${label}: the render held at ${to} from ${last + 1 - at} frames after the call, the line ${N} long`)
		if (sr === 48000) is(off, 0, `${label}: the two held renders mixed by the line, within 2^-23 of the larger in 'enhance', at every sample`)
		ok(peak <= held, `${label}: its peak over the move ${peak.toFixed(5)}, the louder held ${held.toFixed(5)}`)
	}
})

// A jump of limit gives the same samples whatever calls carry it and those before it: the share moves by the samples
// (audio.js slide()), so a host that cuts the stream into calls of 1, 8, 31 or 33 frames, or of 1 to 128 at random,
// gives what one in calls of 128 gives, bit for bit, on and off the 32-frame grid. A glide that takes its length or
// its steps from the calls fails it
test('manifest: a jump of limit gives the same samples whatever the calls that carry it and those before it', async () => {
	let { rnnoise: atom } = await import('./audio.js'), n = 26000, s = 5, rnd = () => (s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 2 ** 32
	for (let sr of [48000, 44100]) for (let [from, to, at] of [[25, 75, 22016], [16, 100, 22077]]) {
		let x = [Float32Array.from({ length: n }, (_, i) => .5 * Math.sin(2 * Math.PI * 1000 * i / sr))], label = `${sr} Hz, limit ${from} → ${to} at frame ${at}`
		let opts = { sampleRate: sr, params: { limit: from, music: 'enhance' }, moves: [[at, { limit: to }]] }, ref = played(atom, x, { ...opts, block: 128 }, `${label}, calls of 128`).y[0]
		for (let [k, calls] of [[1, { block: 1 }], [8, { block: 8 }], [31, { block: 31 }], [33, { block: 33 }], ['1 to 128 at random', { block: 128, cut: () => 1 + Math.floor(rnd() * 128) }]]) {
			let y = played(atom, x, { ...opts, ...calls }, `${label}, calls of ${k}`).y[0], off = 0
			for (let i = 0; i < n; i++) if (!Object.is(y[i], ref[i])) off++
			is(off, 0, `${label}: calls of ${k} frames give the samples calls of 128 give`)
		}
	}
})

// A value that comes while a line moves starts a line of its own from the share then in force, the last sample's
// (audio.js slide()). At 48 kHz each output sample is the input mixed back at the share its input sample came with, so
// it is affine in that share: the render at share a is P + (a − p)/(q − p)·(Q − P), P and Q the renders held at shares
// p and q. Here limit 1 → 100 at `at`, then 16 half a line later: each sample's share is computed below from the two
// lines, the second from where the first is when 16 comes; before `at` the output is P itself, and from the second
// line's last sample on the render held at 16. On a band the guard passes, with music 'enhance' the output is that mix
// within half an ulp of its rounding to float32 and half an ulp of P's and Q's (2^-23 of the larger); with music
// switched to 'pass' before the jump, so that the lines fall in the guard's 200 ms ramp, its blend g·z + (1 − g)·x
// (z the mix, x the input) rounds once more, and g·|z| ≤ |y| + |x| bounds what z's rounding adds: within 2^-23 of
// twice the larger plus the input's sample. A second line from the first one's target, not from where the share is,
// fails it: a step when the value comes
test('manifest: a value of limit that comes while a line moves starts a line from the share in force, at 48 kHz in \'enhance\' and in the guard\'s ramp', async () => {
	let { rnnoise: atom } = await import('./audio.js'), sr = 48000, N = Math.round(.02 * sr), sw = 72000, at = sw + 1061, at2 = at + N / 2, n = at2 + 2 * N
	let x = [band(n / sr)], share = v => 10 ** (-v / 20), p = share(1), q = share(100), r = share(16)
	// the share each input sample comes with: p; from `at` a line toward q; from at2 a line from where that one is toward r
	let mid = p + (q - p) * ((at2 - at) / N), a = i => i < at ? p : i < at2 ? p + (q - p) * ((i - at + 1) / N) : i - at2 + 1 < N ? mid + (r - mid) * ((i - at2 + 1) / N) : r
	let enhanced
	for (let music of ['enhance', 'pass']) {
		let base = music === 'pass' ? [[sw, { music: 'pass' }]] : [], label = `48000 Hz, a band, music ${music === 'pass' ? `'enhance', 'pass' from frame ${sw}` : `'enhance'`}, limit 1 → 100 at frame ${at}, 16 at ${at2}`
		let play = (limit, moves) => played(atom, x, { params: { limit, music: 'enhance' }, moves: [...base, ...moves] }, moves.length ? label : `${label}, held at ${limit}`).y[0]
		let P = play(1, []), Q = play(100, []), R = play(16, []), y = play(1, [[at, { limit: 100 }], [at2, { limit: 16 }]]), first = -1, last = -1, off = 0, worst = 0, ramp = 0
		for (let i = 0; i < n; i++) {
			if (first < 0 && !Object.is(y[i], P[i])) first = i
			if (!Object.is(y[i], R[i])) last = i
			let big = Math.max(Math.abs(P[i]), Math.abs(Q[i])), e = Math.abs(y[i] - (P[i] + (a(i) - p) / (q - p) * (Q[i] - P[i]))), bound = 2 ** -23 * (music === 'pass' ? 2 * big + Math.abs(x[0][i]) : big)
			if (e > bound) off++
			if (big) worst = Math.max(worst, e / big)
			// the guard's gain neither 1 (P the 'enhance' render's) nor 0 (P the input)
			if (music === 'pass' && i >= at && i < at2 + N && P[i] !== enhanced[i] && P[i] !== x[0][i]) ramp++
		}
		if (music === 'enhance') enhanced = P
		else is(ramp, at2 + N - at, `${label}: the lines in the guard's ramp, its gain between 0 and 1 at each of their ${at2 + N - at} samples`)
		ok(first >= at, `${label}: the output moves from ${first - at} frames after the first value`)
		ok(last < at2 + N - 1, `${label}: the render held at 16 from ${last + 1 - at2} frames after the second value, the line ${N} long`)
		is(off, 0, `${label}: the held renders mixed by the two lines' shares, ${music === 'pass' ? 'within 2^-23 of twice the larger plus the input\'s sample' : 'within 2^-23 of the larger'}, at every sample (at most ${(worst / 2 ** -23).toFixed(2)}·2^-23 of the larger)`)
	}
})

// Two instances, each with its own state: a host that holds two calls them in turn, each with its own setting, calls
// and moves of limit, and each gives what it gives made alone, bit for bit. One instance's glide or channel state
// reached from another's calls fails it
test('manifest: two instances called in turn, their limits moving apart, each give what each gives alone', async () => {
	let { rnnoise: atom } = await import('./audio.js'), n = 24000, speech = k => IN[k].subarray(0, n).map(v => v / 32768)
	let one = [[Float32Array.from({ length: n }, (_, i) => .5 * Math.sin(2 * Math.PI * 1000 * i / 48000))], { params: { limit: 16, music: 'enhance' }, moves: [[12000, { limit: 100 }], [12400, { limit: 40 }]] }]
	let two = [[speech('lena+noise'), speech('lena')], { params: { limit: 1 }, moves: [[12077, { limit: 100 }]], block: 37 }]
	let hs = [one, two].map(([x, opts]) => hosting(atom, x, opts)), r = hs.map(h => h.next())
	// both built, then one call of each in turn until both are done
	while (r.some(s => !s.done)) r = r.map((s, k) => s.done ? s : hs[k].next())
	for (let [k, [x, opts]] of [one, two].entries()) {
		let alone = played(atom, x, opts, `instance ${k + 1} alone`).y, off = 0
		alone.forEach((c, ch) => { for (let i = 0; i < c.length; i++) if (!Object.is(r[k].value.y[ch][i], c[i])) off++ })
		is(off, 0, `instance ${k + 1}: called in turn with the other, the samples it gives alone`)
	}
})

// At other rates the mix is made at 48 kHz and the converter back reads its Lanczos taps either side, so a line's
// corners spread over them and a sample can pass both held renders. On a full-scale square at 3 kHz, DC of 0.5 and a
// tone of 0.5 near the band's edge (0.45 of the rate, or of 48 kHz above it), in 'enhance', over a jump across the
// whole range, from 0 (the input) to the top and back, and from 16 to the top: the peak from 200 frames before the call
// to 200 after the line's end passes the louder held render's peak there by at most 0.01 dB, at 8, 16, 22.05, 44.1 and
// 96 kHz
test('manifest: limit glides at other rates, a jump across the range at most 0.01 dB over the louder held peak', async () => {
	let { rnnoise: atom } = await import('./audio.js')
	for (let sr of [8000, 16000, 22050, 44100, 96000]) {
		let N = Math.round(.02 * sr), at = 128 * Math.round(.4 * sr / 128) + 61, n = at + N + Math.round(.1 * sr), e = .45 * Math.min(sr, 48000), worst = -Infinity, which
		for (let [what, f] of [['a full-scale square at 3 kHz', i => Math.sin(2 * Math.PI * 3000 * i / sr + .1) >= 0 ? 1 : -1], ['DC of 0.5', () => .5], [`a ${Math.round(e)} Hz tone of 0.5`, i => .5 * Math.sin(2 * Math.PI * e * i / sr)]]) {
			let x = [Float32Array.from({ length: n }, (_, i) => f(i))], held = {}, play = (limit, moves = []) => host(atom, x, { sampleRate: sr, params: { limit, music: 'enhance' }, moves }).y[0]
			for (let [from, to] of [[0, 100], [100, 0], [16, 100]]) {
				let A = held[from] ??= play(from), B = held[to] ??= play(to), y = play(from, [[at, { limit: to }]]), peak = 0, loud = 0
				for (let i = at - 200; i < at + N + 200; i++) { peak = Math.max(peak, Math.abs(y[i])); loud = Math.max(loud, Math.abs(A[i]), Math.abs(B[i])) }
				let over = 20 * Math.log10(peak / loud)
				if (over > worst) { worst = over; which = `${what}, limit ${from} → ${to}` }
			}
		}
		ok(worst <= .01, `${sr} Hz: the peak over a move at most ${worst.toFixed(4)} dB over the louder held (${which})`)
	}
})

// limit 0 takes nothing off: the share 1 mixes the input back whole and RNNoise's output times 0, and no tail is
// declared, the tail being the ring of RNNoise's part, 1 − a = 0 of the output. At 48 kHz the output is the input itself,
// the latency late, then silence, whatever music does (the guard blends the input with itself); a jump from 16 to 0
// reaches it from the 20 ms line's last sample. At 44.1 kHz the mix is made at 48 kHz and comes back through the
// converter: the input as resample-sinc carries it to 48 kHz and back offline, but in the last 50 ms (the offline
// resampler sees the end, the stream the host's silence), and from 128 frames past the input's end (the contract's grace
// for a ring) 60 dB under its RMS. A share that is not 1 at 0, or a tail declared there, fails it
test('manifest: limit 0 takes nothing off: the input itself, the latency late, and no tail; a jump to 0 reaches it over the 20 ms line; at 44.1 kHz the input through resample-sinc and back', async () => {
	let { rnnoise: atom } = await import('./audio.js'), { default: resample } = await import('@audio/resample-sinc'), n = 48000, at = 22077, N = 960
	let x = [IN['lena+noise'].subarray(0, n).map(v => v / 32768), IN.lena.subarray(0, n).map(v => v / 32768)]
	let differ = (a, b, from, to) => { let k = 0; for (let i = from; i < to; i++) if (!Object.is(a[i], b[i])) k++; return k }
	is(tailOf(atom, { sampleRate: 48000, params: { limit: Float32Array.of(0) } }), 0, 'no tail declared at 0')
	for (let music of ['pass', 'enhance']) {
		let label = `48000 Hz, music '${music}', limit 0`, { y, T } = played(atom, x, { params: { limit: 0, music }, tail: true, more: 4800 }, label)
		is(T, 0, `${label}: the host renders no tail`)
		y.forEach((c, k) => is(differ(c, x[k], 0, n) + c.subarray(n).filter(v => v !== 0).length, 0, `${label}, channel ${k + 1}: the input itself, sample for sample, then silence`))
		let z = played(atom, [x[0]], { params: { limit: 16, music }, moves: [[at, { limit: 0 }]] }, `${label}, from 16 at frame ${at}`).y[0]
		is(differ(z, x[0], at + N - 1, n), 0, `${label}, from 16 at frame ${at}: the input itself from the line's last sample, ${N - 1} frames after`)
	}
	let sr = 44100, a = x[0].subarray(0, sr), cut = sr - 2205, floor = 1e-3 * rmsOf(a, 0, sr), loud = -1
	let back = resample(resample(Float32Array.from(a), { from: sr, to: 48000 }), { from: 48000, to: sr })
	let { y: [y], T } = played(atom, [a], { sampleRate: sr, params: { limit: 0, music: 'enhance' }, tail: true, more: 4410 }, '44100 Hz, limit 0')
	is(T, 0, '44100 Hz, limit 0: the host renders no tail')
	is(differ(y, back, 0, cut), 0, `44100 Hz, limit 0: the input through resample-sinc to 48 kHz and back, sample for sample but in the last 50 ms (${snr(a.subarray(0, cut), back.subarray(0, cut)).toFixed(1)} dB SNR from the input)`)
	for (let i = sr + 128; i + 64 <= y.length; i += 64) if (rmsOf(y, i, i + 64) > floor) { loud = i - sr; break }
	is(loud, -1, '44100 Hz, limit 0: 60 dB under the input\'s RMS from 128 frames past its end on')
})

// The top, 100, mixes back 10^−5 of the input: no limit, to within that. At 48 kHz each output sample is affine in the
// share (RNNoise's part times 1 − a plus the input's times a, made in doubles and rounded once to float32), so the render
// at 100 lies 10^−5 of the way from the render with no limit (share 0: denoise()'s at Infinity, the frame API's output,
// above) to the render at 0, the input itself (above), up to the renders' rounding: with music 'enhance', where the
// output is the mix itself, half an ulp of each (2^-23 of the larger); with 'pass', where the guard blends
// g·z + (1 − g)·x and rounds once more, 2^-23 of twice the larger plus the input's sample (the test of a value mid-line
// derives it). A top that leaves more of the input, or less, fails it
test('manifest: the top of limit, 100, is no limit to within 10^-5: its render 10^-5 of the way from the unlimited one to the input', async () => {
	let { rnnoise: atom } = await import('./audio.js'), n = 2 * 48000, x = IN['lena+noise'].subarray(0, n).map(v => v / 32768)
	for (let music of ['enhance', 'pass']) {
		let label = `48000 Hz, music '${music}'`, none = await denoise(x, { sampleRate: 48000, limit: Infinity, music }), y = played(atom, [x], { params: { limit: 100, music } }, `${label}, limit 100`).y[0]
		let off = 0, most = 0, way = 0
		for (let i = 0; i < n; i++) {
			let big = Math.max(Math.abs(y[i]), Math.abs(none[i])), e = Math.abs(y[i] - (none[i] + 1e-5 * (x[i] - none[i])))
			if (e > 2 ** -23 * (music === 'pass' ? 2 * big + Math.abs(x[i]) : big)) off++
			most = Math.max(most, Math.abs(y[i] - none[i])); way = Math.max(way, Math.abs(x[i] - none[i]))
		}
		is(off, 0, `${label}: limit 100 the render with no limit plus 10^-5 of the input's difference from it, within the rounding, at every sample (at most ${most.toExponential(1)} from the unlimited render, the input at most ${way.toFixed(3)})`)
	}
})

// A host's automation curve hands each call the value at its first frame, at most every 32 frames (contract 1.0, Time).
// The share falls as the limit rises over the whole range, so a curve from one value to another moves the share between
// theirs only; at 48 kHz, where each output sample is affine in the share its input sample came with, every sample of
// the move then lies between the two renders held at its ends: at most the larger, within half an ulp of the output's
// rounding and half of the held render's (2^-23 of it), with music 'enhance'. A curve linear in dB over 200 ms from 16
// and from 100 to 0, under white noise, a 1 kHz tone and noisy speech; at 44.1 kHz, where the converter back spreads
// each line over its taps, the peak over the move at most 0.01 dB over the louder held one (the bound of the test of
// other rates above). A share that is 0 at 0 and nears 1 just above it fails it: such a curve passes the input whole
// on its way, louder than either end where the model takes noise
test('manifest: a host\'s curve of limit into 0 passes only between the renders held at its ends, at 48 and 44.1 kHz', async () => {
	let { rnnoise: atom } = await import('./audio.js'), s = 11, noise = n => Float32Array.from({ length: n }, () => .3 * ((s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 2 ** 32 - .5))
	for (let sr of [48000, 44100]) {
		let a = Math.round(.5 * sr), b = a + Math.round(.2 * sr), n = b + Math.round(.3 * sr), N = Math.round(.02 * sr), ahead = sr === 48000 ? 0 : 200
		for (let [what, x] of [['white noise', noise(n)], ['a 1 kHz tone of 0.5', Float32Array.from({ length: n }, (_, i) => .5 * Math.sin(2 * Math.PI * 1000 * i / sr))], ['noisy speech', IN['lena+noise'].subarray(0, n).map(v => v / 32768)]]) {
			let play = (limit, moves = [], label) => played(atom, [x], { sampleRate: sr, params: { limit, music: 'enhance' }, moves }, label).y[0], Q = play(0, [], `${sr} Hz, ${what}, held at 0`)
			for (let from of [16, 100]) {
				let moves = [], label = `${sr} Hz, ${what}, limit ${from} → 0 over 200 ms in steps of 32 frames`
				for (let i = a; i < b; i += 32) moves.push([i, { limit: from * (1 - (i - a) / (b - a)) }])
				moves.push([b, { limit: 0 }])
				let P = play(from, [], `${label}, held at ${from}`), y = play(from, moves, label), off = 0, peak = 0, held = 0
				for (let i = 0; i < n; i++) {
					let M = Math.max(Math.abs(P[i]), Math.abs(Q[i]))
					if (Math.abs(y[i]) > M * (1 + 2 ** -23)) off++
					if (i >= a - ahead && i < b + N + ahead) { peak = Math.max(peak, Math.abs(y[i])); held = Math.max(held, M) }
				}
				let over = 20 * Math.log10(peak / held)
				if (sr === 48000) is(off, 0, `${label}: every sample at most the larger held render's, within 2^-23 of it (the peak over the move ${over.toFixed(4)} dB against the louder held)`)
				else ok(over <= .01, `${label}: the peak over the move ${over.toFixed(4)} dB against the louder held`)
			}
		}
	}
})

// ------------------------------------------------ DeepFilterNet3

test('deepfilter: ERB bands as libDF erb_fb(48000, 960, 32, 2) gives them (df_state.erb_widths())', () => {
	is(erbWidths(48000, 960, 32, 2), [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 5, 5, 7, 7, 8, 10, 12, 13, 15, 18, 20, 24, 28, 31, 37, 42, 50, 56, 67])
})

// an upstream-shaped export (ustar: enc.onnx, erb_dec.onnx, df_dec.onnx, config.ini) with stand-in graphs
function tar(files) {
	let parts = [], te = new TextEncoder()
	for (let [name, data] of Object.entries(files)) {
		let h = new Uint8Array(512), d = typeof data === 'string' ? te.encode(data) : data
		h.set(te.encode(name)); h.set(te.encode(d.length.toString(8).padStart(11, '0') + '\0'), 124); h[156] = 48
		parts.push(h, d, new Uint8Array((512 - d.length % 512) % 512))
	}
	parts.push(new Uint8Array(1024))
	let out = new Uint8Array(parts.reduce((n, p) => n + p.length, 0)), o = 0
	for (let p of parts) { out.set(p, o); o += p.length }
	return out
}
const CONFIG = `[train]\nmodel = deepfilternet3\n\n[df]\nsr = 48000\nfft_size = 960\nhop_size = 480\nnb_erb = 32\nnb_df = 96\nnorm_tau = 1\nmin_nb_erb_freqs = 2\ndf_order = 5\ndf_lookahead = 2\n\n[deepfilternet]\nconv_lookahead = 2\n`
// stand-ins: ERB gains `gain`, one deep-filter tap of 1 at `tap` (2 = the current frame); `seen` collects the ERB
// features the encoder is fed
function standIn({ gain = 1, tap = 2, seen } = {}) {
	let t = (a, dims) => ({ data: a, dims, type: 'float32' })
	return async bytes => {
		let kind = new TextDecoder().decode(bytes)
		return {
			async run(feeds) {
				if (kind === 'enc') seen?.push(feeds.feat_erb.data.slice())
				if (kind === 'enc') { let S = feeds.feat_erb.dims[2], z = n => new Float32Array(n); return { e0: t(z(64 * S * 32), [1, 64, S, 32]), e1: t(z(64 * S * 16), [1, 64, S, 16]), e2: t(z(64 * S * 8), [1, 64, S, 8]), e3: t(z(64 * S * 8), [1, 64, S, 8]), emb: t(z(S * 512), [1, S, 512]), c0: t(z(64 * S * 96), [1, 64, S, 96]), lsnr: t(z(S), [1, S, 1]) } }
				let S = feeds.emb.dims[1]
				if (kind === 'erb_dec') return { m: t(new Float32Array(S * 32).fill(gain), [1, 1, S, 32]) }
				let c = new Float32Array(S * 96 * 10)
				if (tap != null) for (let i = 0; i < S * 96; i++) c[i * 10 + 2 * tap] = 1
				return { coefs: t(c, [1, S, 96, 10]) }
			},
			free() {},
		}
	}
}
const EXPORT = tar({ 'tmp/export/enc.onnx': 'enc', 'tmp/export/erb_dec.onnx': 'erb_dec', 'tmp/export/df_dec.onnx': 'df_dec', 'tmp/export/config.ini': CONFIG })

test('deepfilter: identity gains and taps give the input back, aligned, whole or in crossfaded chunks', async () => {
	let net = await loadDeepFilter(EXPORT, { session: standIn() })
	is([net.params.order, net.params.dfLa, net.params.convLa, net.params.nbDf, net.params.alpha], [5, 2, 2, 96, Math.fround(.99)])
	let x = IN['lena+noise'].subarray(0, 96000).map(v => v / 32768)
	let y = await enhance(x, net)
	ok(maxDiff(x, y) < 2e-6, `whole: max |y − x| ${maxDiff(x, y).toExponential(1)}`)
	let z = await enhance(x, net, { chunk: 50, warmup: 20, fade: 10 })
	is(maxDiff(y, z), 0, 'chunked = whole for a stateless model')
})

test('deepfilter: a tap one frame ahead advances the band below 4.8 kHz by 480 samples; gains 0 remove the rest', async () => {
	let net = await loadDeepFilter(EXPORT, { session: standIn({ gain: 0, tap: 3 }) })
	let n = 48000, lo = Float32Array.from({ length: n }, (_, i) => .5 * Math.sin(2 * Math.PI * 1000 * i / 48000))
	let x = lo.map((v, i) => v + .3 * Math.sin(2 * Math.PI * 9000 * i / 48000))
	let y = await enhance(x, net), err = 0
	for (let i = 2000; i < n - 3000; i++) err = Math.max(err, Math.abs(y[i] - lo[i + 480]))
	ok(err < 1e-5, `y[n] = 1 kHz part at n + 480 to ${err.toExponential(1)}`)
})

test('deepfilter: denoise() limits DeepFilterNet3 to 18 dB unless given a limit; Infinity is none', async () => {
	let m = await load('deepfilternet3', { weights: EXPORT, session: standIn({ gain: 0 }) }) // removes everything above 4.8 kHz
	let x = IN['lena+noise'].subarray(0, 48000).map(v => v / 32768), run = opts => denoise(x, { sampleRate: 48000, model: m, music: 'enhance', ...opts })
	let [unset, eighteen, none] = [await run({}), await run({ limit: 18 }), await run({ limit: Infinity })]
	is(maxDiff(unset, eighteen), 0, 'default = limit 18')
	ok(maxDiff(unset, none) > 1e-3, `limit Infinity differs by up to ${maxDiff(unset, none).toExponential(1)}`)
	is(maxDiff(unset, mixback(x, none)), 0, 'the unlimited output with the input mixed back by mixback()')
	is(maxDiff(await run({ floor: 0 }), mixback(x, none, { floor: 0 })), 0, 'floor 0: a constant 18 dB')
	is(maxDiff(await run({ everywhere: true }), mixback(x, none, { everywhere: true })), 0, 'everywhere: as mixback() gives it')
	m.free()
})

test('deepfilter: export errors', async () => {
	await rejects(() => loadDeepFilter(tar({ 'config.ini': CONFIG }), { session: standIn() }), /enc\.onnx/)
	await rejects(() => loadDeepFilter(tar({ 'enc.onnx': 'enc', 'erb_dec.onnx': 'erb_dec', 'df_dec.onnx': 'df_dec', 'config.ini': CONFIG.replace('deepfilternet3', 'deepfilternet2') }), { session: standIn() }), /deepfilternet2/)
	await rejects(() => loadDeepFilter(tar({ 'enc.onnx': 'enc', 'erb_dec.onnx': 'erb_dec', 'df_dec.onnx': 'df_dec', 'config.ini': CONFIG.replace('sr = 48000', 'sr = 16000') }), { session: standIn() }), /48 kHz/)
})

// a held /a/: harmonics of f0 (190 Hz, 5 Hz vibrato of 2%) at 1/h through three formants (700, 1220, 2600 Hz;
// Peterson & Barney 1952, men's /ɑ/), `dur` s at 48 kHz
function vowel(dur, f0 = 190) {
	let n = Math.round(dur * 48000), out = new Float32Array(n), ph = 0
	let F = [[700, 130], [1220, 70], [2600, 160]], amp = h => F.reduce((s, [c, b]) => s + 1 / (1 + ((h - c) / b) ** 2), 0) / h
	for (let i = 0; i < n; i++) {
		ph += f0 * (1 + .02 * Math.sin(2 * Math.PI * 5 * i / 48000)) / 48000
		let v = 0, fi = f0 * (1 + .02 * Math.sin(2 * Math.PI * 5 * i / 48000))
		for (let h = 1; h * fi < 8000; h++) v += amp(h * fi) * Math.sin(2 * Math.PI * h * ph)
		out[i] = v
	}
	let r = Math.sqrt(out.reduce((s, v) => s + v * v, 0) / n)
	return out.map(v => .1 * v / r)
}
// 32-bit LCG white noise in ±a
const white = (n, a, seed = 1) => { let s = seed; return Float32Array.from({ length: n }, () => (s = (Math.imul(s, 1664525) + 1013904223) >>> 0, a * (s / 2 ** 32 - .5))) }
const span = (x, a, b) => 10 * Math.log10(x.subarray(a, b).reduce((s, v) => s + v * v, 0) / (b - a) + 1e-30)

test('deepfilter: voicing() marks a held vowel standing over the noise, not the noise, not a steady buzz', () => {
	// 1 s of noise, 2 s of the vowel over it (20 dB above), 1 s of noise
	let v = vowel(2), x = white(4 * 48000, .02).map((s, i) => s + (i >= 48000 && i < 3 * 48000 ? v[i - 48000] : 0)), T = Math.floor((x.length + 960) / 480)
	let { on } = voicing(x, T, 480), share = (a, b) => on.subarray(a, b).reduce((s, v) => s + v, 0) / (b - a)
	ok(share(110, 290) > .95, `the vowel: ${(100 * share(110, 290)).toFixed(0)}% of its frames`)
	ok(share(0, 90) === 0 && share(310, 400) === 0, 'the noise around it: none')
	// a mains buzz (harmonics of 120 Hz at 1/h) under the same noise: periodic and sustained, but the background itself
	let bz = white(4 * 48000, .02, 7).map((s, i) => { let b = 0; for (let h = 1; h < 30; h++) b += Math.sin(2 * Math.PI * 120 * h * i / 48000) / h; return s + .05 * b })
	is(voicing(bz, T, 480).on.reduce((s, v) => s + v, 0), 0, 'a steady buzz: no frame')
})

test('deepfilter: the voice guard keeps a held vowel a model would remove, and nothing else', async () => {
	let net = await loadDeepFilter(EXPORT, { session: standIn({ gain: 0, tap: null }) }) // removes everything
	let v = vowel(2), x = white(4 * 48000, .02).map((s, i) => s + (i >= 48000 && i < 3 * 48000 ? v[i - 48000] : 0))
	let y = await enhance(x, net, { voice: true }), z = await enhance(x, net)
	ok(span(z, 0, x.length) < -200, 'without the guard: silence')
	let kept = span(y, 60000, 132000) - span(v, 12000, 84000)
	ok(kept > -3, `the vowel: ${kept.toFixed(1)} dB of it kept`)
	ok(span(y, 0, 40000) < -200 && span(y, 152000, x.length) < -200, 'the noise around it: removed')
})

// a perfect model (y the voice, a 220 Hz tone for 4.5 s, then 3.5 s of pause: the voice's level the tone's, the louder
// half of the frames all tone) and white noise `db` dB under the voice: how far under the voice mixback() leaves the
// noise, under the voice (0.2-4.2 s) and in its pause, from 0.8 s past it (REACH, 0.3 s, and the noise's 0.5 s clear)
test('deepfilter: mixback() drops quiet noise by the limit; loud noise in the voice\'s pauses down to the floor under it, under the voice by the limit', () => {
	let n = 8 * 48000, t = 4.5 * 48000, s = Float32Array.from({ length: n }, (_, i) => i < t ? .1 * Math.sin(2 * Math.PI * 220 * i / 48000) : 0), w = white(n, 1, 3)
	let k = db => .1 / Math.SQRT2 * 10 ** (-db / 20) * Math.sqrt(12), V = span(s, 0, t), under = (r, a, b) => V - span(r, a, b), near = (a, b, e = .2) => Math.abs(a - b) < e
	let left = (db, o) => { let r = mixback(s.map((v, i) => v + k(db) * w[i]), s, o).map((v, i) => v - s[i]); return [under(r, 9600, 201600), under(r, 254400, n)] }
	let [v40, p40] = left(40), [v25, p25] = left(25), [v10, p10] = left(10), [vo, po] = left(-5)
	ok(near(v40, 58) && near(p40, 58), `noise 40 dB under the voice: ${v40.toFixed(1)} and ${p40.toFixed(1)} dB under, the limit's 18 dB down`)
	ok(near(v25, 43) && near(p25, 43), `25 dB under: ${v25.toFixed(1)} and ${p25.toFixed(1)}, 18 down`)
	ok(near(v10, 28) && near(p10, FLOOR), `10 dB under: under the voice ${v10.toFixed(1)}, 18 down; in its pause ${p10.toFixed(1)}, at the floor (${FLOOR})`)
	ok(near(vo, 13) && near(po, FLOOR), `5 dB over: under the voice ${vo.toFixed(1)}, 18 down; in its pause ${po.toFixed(1)}, at the floor`)
	ok(near(left(10, { floor: 0 })[1], 28) && near(left(10, { floor: 30 })[1], 30), 'floor 0: 18 dB down, as before 0.5; floor 30: 30 under')
	let [v6, p6] = left(10, { limit: 6 })
	ok(near(left(40, { limit: 6 })[1], 46) && near(v6, 16) && near(p6, FLOOR), 'limit 6: quiet noise 6 dB down; loud noise 6 down under the voice, at the floor in its pause')
	is(maxDiff(mixback(s.map((v, i) => v + w[i]), s, { limit: Infinity }), s), 0, 'limit Infinity: the model\'s output')
	// limit 0, cap 1: nothing taken under the voice; in its pauses the floor still takes noise nearer the voice than it
	let [v0, p0] = left(10, { limit: 0 }), [q0, r0] = left(40, { limit: 0 }), noisy = s.map((v, i) => v + k(10) * w[i])
	ok(near(v0, 10) && near(p0, FLOOR) && near(q0, 40) && near(r0, 40), `limit 0: noise 10 dB under the voice ${v0.toFixed(1)} under it, in its pause ${p0.toFixed(1)}, at the floor; 40 under, ${q0.toFixed(1)} and ${r0.toFixed(1)}, as it came`)
	ok(mixback(noisy, s, { limit: 0, floor: 0 }).every((v, i) => Object.is(v, noisy[i])), 'limit 0 and floor 0: the input itself, sample for sample')
	for (let limit of [null, NaN, '18']) throws(() => mixback(noisy, s, { limit }), /^TypeError: .*limit is dB/, `limit ${typeof limit === 'string' ? `'${limit}'` : limit}: a TypeError`)
	// in the pause, loud noise to 6.5 s, then quiet: each as above, 0.5 s from the change
	let x = s.map((v, i) => v + k(i < 6.5 * 48000 ? 10 : 40) * w[i]), r = mixback(x, s).map((v, i) => v - s[i])
	let a = under(r, 254400, 288000), b = under(r, 336000, n)
	ok(near(a, FLOOR, .3) && near(b, 58, .3), `the noise drops 30 dB: ${a.toFixed(1)} then ${b.toFixed(1)} dB under the voice`)
	let y = s.slice(); y.set(x.subarray(0, 48000))
	is(maxDiff(mixback(x, y).subarray(0, 48000), x.subarray(0, 48000)), 0, 'where input and output agree (music passed): the input, bit for bit')
})

// a model that takes the voice in places, as DeepFilterNet3 does at low SNR: y the tone with its first 0.2 s of every
// 0.5 s gone, the input under white noise 10 dB down. Whatever the noise, the voice it took comes back as the limit
// says, 18 dB down, 6 at 6; `everywhere` takes it down with the noise, to the floor, and gives 6 and 18 the same output.
test('deepfilter: mixback() gives back the voice a model took by the limit, the noise near it or not', () => {
	let n = 4 * 48000, s = Float32Array.from({ length: n }, (_, i) => .1 * Math.sin(2 * Math.PI * 220 * i / 48000)), w = white(n, 1, 3)
	let x = s.map((v, i) => v + .1 / Math.SQRT2 * 10 ** (-10 / 20) * Math.sqrt(12) * w[i]), y = s.map((v, i) => (i / 48000) % .5 < .2 ? 0 : v)
	// the voice in the output where the model took it: its projection on the tone over the middle 0.1 s of each gap from
	// the second, dB re the tone
	let back = z => { let p = 0, q = 0; for (let g = 1; g < 8; g++) for (let i = g * 24000 + 2400; i < g * 24000 + 7200; i++) { p += z[i] * s[i]; q += s[i] * s[i] } return 20 * Math.log10(p / q) }
	let d18 = back(mixback(x, y)), d6 = back(mixback(x, y, { limit: 6 })), e18 = back(mixback(x, y, { everywhere: true })), e6 = back(mixback(x, y, { everywhere: true, limit: 6 }))
	ok(Math.abs(d18 + 18) < .5 && Math.abs(d6 + 6) < .5, `the voice taken comes back ${d18.toFixed(1)} dB at the default limit, ${d6.toFixed(1)} at 6`)
	ok(e18 < -30 && Math.abs(e18 - e6) < .5, `everywhere: ${e18.toFixed(1)} dB at the default limit, ${e6.toFixed(1)} at 6, down with the noise to the floor`)
})

// The default and `everywhere` held to the published mixes, sample for sample: the perfect model's take above (the
// tone for 4.5 s, then a pause), the noise 10 dB under the voice to 6.5 s and 40 dB under after, y without the tone's
// 1.0 to 1.2 s (a stretch the model took) and, from 5 to 7.5 s, one 10 ms frame every 0.4 s of the input's noise 27 dB
// under the voice (residue within speech()'s 30 dB)
test('deepfilter: mixback() is 0.8.0\'s mix at its defaults and 0.7.0\'s with everywhere, sample for sample', () => {
	let n = 8 * 48000, t = 4.5 * 48000, s = Float32Array.from({ length: n }, (_, i) => i < t ? .1 * Math.sin(2 * Math.PI * 220 * i / 48000) : 0), w = white(n, 1, 3)
	let x = s.map((v, i) => v + .1 / Math.SQRT2 * 10 ** (-(i < 6.5 * 48000 ? 10 : 40) / 20) * Math.sqrt(12) * w[i]), y = s.map((v, i) => i >= 48000 && i < 57600 ? 0 : i >= 240000 && i < 360000 && (i - 240000) % 19200 < 480 ? .1 / Math.SQRT2 * 10 ** (-27 / 20) * Math.sqrt(12) * w[i] : v)
	is(sha(mixback(x, y)), DEFAULT, 'the default, limit 18 and floor 40: 0.8.0\'s samples')
	is(sha(mixback(x, y, { limit: 6, floor: 30 })), DEFAULT_6_30, 'limit 6, floor 30: 0.8.0\'s samples')
	is(sha(mixback(x, y, { everywhere: true })), EVERYWHERE, 'everywhere, limit 18 and floor 40: 0.7.0\'s samples')
	is(sha(mixback(x, y, { everywhere: true, limit: 6, floor: 30 })), EVERYWHERE_6_30, 'everywhere, limit 6, floor 30: 0.7.0\'s samples')
	let f0 = mixback(x, y, { floor: 0 })
	ok(mixback(x, y, { everywhere: true, floor: 0 }).every((v, i) => Object.is(v, f0[i])), 'floor 0: a constant limit either way')
})

test('deepfilter: bandEdge() finds the band an input fills, at its own rate', () => {
	let x = white(48000, .1), lp = new Float32Array(48000)
	// a 15 kHz low-pass: 255-tap windowed sinc (Blackman), its stopband 74 dB down
	let h = Float32Array.from({ length: 255 }, (_, i) => { let k = i - 127, w = .42 - .5 * Math.cos(2 * Math.PI * i / 254) + .08 * Math.cos(4 * Math.PI * i / 254); return w * (k ? Math.sin(2 * Math.PI * 15000 / 48000 * k) / (Math.PI * k) : 2 * 15000 / 48000) })
	for (let i = 0; i < 48000; i++) { let v = 0; for (let k = 0; k < 255; k++) if (i - k >= 0) v += h[k] * x[i - k]; lp[i] = v }
	is(bandEdge(x, 48000), 24000, 'white noise at 48 kHz: the full band')
	is(bandEdge(x.subarray(0, 16000), 16000), 8000, 'at 16 kHz: its Nyquist frequency')
	let e = bandEdge(lp, 48000)
	ok(e > 15000 && e < 16500, `low-passed at 15 kHz: ${e} Hz`)
	is(bandEdge(new Float32Array(48000), 44100), 22050, 'silence: nothing measured, the rate\'s band')
	is(bandEdge(x.subarray(0, 500), 48000), 24000, 'shorter than a frame: the rate\'s band')
})

test('deepfilter: a 16 kHz input\'s empty bands reach the network as white noise FILL dB under the speech, not at libDF\'s 1e-10; a full band\'s as before; a band holding the edge as it is', async () => {
	let seen = [], m = await load('deepfilternet3', { weights: EXPORT, session: standIn({ seen }) })
	let x = white(32000, .2, 3), y = await denoise(x, { sampleRate: 16000, model: m })
	ok(y.length === x.length && y.every(Number.isFinite), 'same length, finite')
	// band 31 (20.7 to 24 kHz), all above the edge: v = 10·log10(floor) fluctuating, its mean normalization starting at
	// -90 dB (libDF), α = 0.99; the network's frame 0 is frame 2 (conv_lookahead)
	let top = seen[0].filter((_, i) => i % 32 === 31), v = 10 * Math.log10(10 ** ((LEVEL + FILL) / 10) / 1920), a = Math.fround(.99)
	let first = (v + 90) * a ** 3 / 40
	ok(Math.abs(top[0] - first) < .05, `the first frame's feature ${top[0].toFixed(3)}, the floor's ${first.toFixed(3)}; libDF's 1e-10, ${((-100 + 90) * a ** 3 / 40).toFixed(3)}`)
	let moves = 0
	for (let t = 101; t < 190; t++) moves += Math.abs(top[t] - top[t - 1]) / 89
	ok(moves > .005, `it moves as noise does: ${moves.toFixed(4)} per frame`)
	// a full band: no floor, the features as without it, bit for bit
	let z = white(48000, .2, 5), f1 = [], f2 = []
	let n1 = await loadDeepFilter(EXPORT, { session: standIn({ seen: f1 }) }), n2 = await loadDeepFilter(EXPORT, { session: standIn({ seen: f2 }) })
	await enhance(z, n1, { edge: bandEdge(z, 48000) }); await enhance(z, n2)
	is(maxDiff(f1[0], f2[0]), 0, 'a 48 kHz full-band input: the same features')
	// only the bands wholly above the edge: at 8 kHz, band 25 (8.5 kHz) on; band 24 (7.3 to 8.5 kHz) holds the input's own content
	f1.length = f2.length = 0
	await enhance(z, n1, { edge: 8000 }); await enhance(z, n2)
	let same = b => f1[0].every((v, i) => i % 32 !== b || v === f2[0][i])
	ok([...Array(25).keys()].every(same) && ![25, 26, 31].some(same), 'bands 0 to 24 as without the floor, 25 on over it')
	// the floor is heard, never applied: identity gains and taps still give the input back
	let id = await enhance(z, n1, { edge: 8000 })
	ok(maxDiff(id, z) < 2e-6, `identity model with the floor: max |y − x| ${maxDiff(id, z).toExponential(1)}`)
	m.free()
})

// ------------------------------------------------ music guard

const GUARD = JSON.parse(readFileSync(new URL('./fixtures/guard.json', import.meta.url)))
const VB = path.join(os.homedir(), '.cache', 'audiojs', 'data', 'vbdemand-train', 'noisy')

// a band, synthetic: four chords of plucked harmonic tones every half second, a bass note under each, a tick of noise
function band(sec, sr = 48000) {
	let x = new Float32Array(Math.round(sec * sr)), s = 9, rnd = () => (s = (s * 1664525 + 1013904223) % 4294967296, s / 4294967296 - .5)
	let chords = [[60, 64, 67], [57, 60, 64], [53, 57, 60], [55, 59, 62]], hz = m => 440 * 2 ** ((m - 69) / 12)
	for (let b = 0; b * .5 < sec; b++) {
		let t0 = Math.round(b * .5 * sr), c = chords[(b >> 2) % 4]
		for (let n of [...c, c[0] - 24]) for (let h = 1; h <= 8 && hz(n) * h < 8000; h++)
			for (let i = 0; i < sr && t0 + i < x.length; i++) x[t0 + i] += .05 / h * Math.exp(-i / sr * (2 + h)) * Math.sin(2 * Math.PI * hz(n) * h * i / sr)
		for (let i = 0; i < 2000 && t0 + i < x.length; i++) x[t0 + i] += .1 * rnd() * Math.exp(-i / 300)
	}
	return x
}
// 16-bit mono wav → Float32Array
function wav(file) {
	let b = readFileSync(file), dv = new DataView(b.buffer, b.byteOffset, b.byteLength), o = 12
	while (o < b.length) {
		let id = b.toString('ascii', o, o + 4), len = dv.getUint32(o + 4, true)
		if (id === 'data') return Float32Array.from({ length: len / 2 }, (_, i) => dv.getInt16(o + 8 + 2 * i, true) / 32768)
		o += 8 + len + (len & 1)
	}
}

test('guard: guard.bin is ina\'s network as scripts/guard.py writes it; on lena the port gives the float32 keras model\'s probabilities within 0.005, its wasm kernel and the JS loop the same bits', () => {
	let bytes = readFileSync(new URL('./guard.bin', import.meta.url))
	is(sha(bytes), GUARD.bin)
	let x = IN.lena.map(v => v / 32768), [P, Q] = [true, false].map(simd => analyzer(guardModel(bytes, { simd }))(x)), d = 0
	is(P.length, GUARD.lena.length, 'a patch every 100 ms')
	P.forEach((p, k) => p.forEach((v, c) => d = Math.max(d, Math.abs(v - GUARD.lena[k][c]))))
	ok(d < 5e-3, `max |p − keras| ${d.toExponential(1)} (float16 weights)`)
	ok(P.every((p, k) => p.every((v, c) => v === Q[k][c])), 'wasm = JS, bit for bit')
	throws(() => guardModel(bytes.subarray(0, 1000)), /guard weights/)
})

test('guard: music passes through untouched, bit for bit, at 48 and 44.1 kHz; music: \'enhance\' processes it as before 0.4 (0.3 removed it here)', async () => {
	let m = await load('deepfilternet3', { weights: EXPORT, session: standIn({ gain: 0, tap: null }) }) // removes everything
	for (let sr of [48000, 44100]) {
		let x = band(6, sr), y = await denoise(x, { sampleRate: sr, model: m }), z = await denoise(x, { sampleRate: sr, model: m, music: 'enhance' })
		is(maxDiff(y, x), 0, `${sr} Hz: the band comes back as it went in`)
		ok(maxDiff(z, x) > .05, `music: 'enhance': processed, by up to ${maxDiff(z, x).toFixed(2)}`)
	}
	await rejects(() => denoise(band(1), { sampleRate: 48000, model: m, music: 'keep' }), /music is 'pass' or 'enhance'/)
	m.free()
	// RNNoise decides as a stream: on what has arrived, so the band's first second is denoised, the rest passes
	let x = band(6), d = online(await guardNet()), first = -1
	for (let i = 0, k = 0; i < x.length; i += 480, k++) if (d(x.subarray(i, i + 480)) && first < 0) first = k
	ok(first > 0 && first < 120, `the stream passes the band from its input frame ${first} (${(first / 100).toFixed(2)} s)`)
	let y = await denoise(x, { sampleRate: 48000 }), from = (first - 2) * 480 + RAMP
	is(maxDiff(y.subarray(from), x.subarray(from)), 0, `rnnoise: from ${(from / 48000).toFixed(2)} s on, the band itself`)
	ok(maxDiff(y.subarray(0, 48000), x.subarray(0, 48000)) > .01, 'before it, denoised')
})

test('guard: segments switch where the material does, the gain over a 200 ms raised cosine: band, noisy speech, band (skipped without VoiceBank+DEMAND)', async () => {
	if (!existsSync(VB)) return console.log('  (no VoiceBank+DEMAND in ~/.cache/audiojs/data: skipped)')
	let nn = await guardNet(), B = band(5), files = readdirSync(VB).filter(f => f.endsWith('.wav')).sort().slice(0, 4)
	let S = files.map(f => wav(path.join(VB, f))), n = S.reduce((a, s) => a + s.length, 0), x = new Float32Array(2 * B.length + n), o = B.length
	x.set(B); for (let s of S) { x.set(s, o); o += s.length } x.set(B, o)
	let pass = offline(x, nn), cuts = [5, o / 48000]
	let edges = [...pass].flatMap((v, g) => g && v !== pass[g - 1] ? [g / 100] : [])
	ok(pass[100] && !pass[Math.round(100 * (cuts[0] + cuts[1]) / 2)] && pass.at(-100), 'the band passes, the voice is enhanced')
	ok(edges.length === 2 && edges.every((e, i) => Math.abs(e - cuts[i]) < .5), `switches at ${edges.join(' and ')} s, the cuts at ${cuts.map(c => c.toFixed(2)).join(' and ')} s`)
	let g = gains(pass, x.length, true), step = 0
	for (let i = 1; i < g.length; i++) step = Math.max(step, Math.abs(g[i] - g[i - 1]))
	ok(step <= Math.PI / 2 / RAMP * 1.001, `no jump: at most ${step.toExponential(2)} per sample, π/2 over ${RAMP}`)
	ok(Math.abs(g[Math.round(edges[0] * 48000)] - .5) < 1e-3, 'each ramp centered on its switch')
	ok(g.subarray(0, (edges[0] - .2) * 48000).every(v => v === 0) && g.subarray((edges[0] + .2) * 48000, (edges[1] - .2) * 48000).every(v => v === 1), 'exactly 0 and exactly 1 between the ramps')
})

test('guard(): the offline gain at the input\'s rate, for any enhancement: 0 over music, the 48 kHz gain itself at 48 kHz', async () => {
	let nn = await guardNet()
	for (let sr of [48000, 44100]) {
		let x = band(4, sr), g = await guard(x, sr)
		is(g.length, x.length, `${sr} Hz: a gain per sample`)
		ok(g.every(v => v === 0), `${sr} Hz: the band passes whole`)
	}
	let x = band(2), g = await guard(x, 48000)
	is(maxDiff(g, gains(offline(x, nn), x.length, true)), 0, 'at 48 kHz: gains(offline(x)) as denoise() takes it')
})

test('segments(): what an input is, stretch by stretch, as the classifier decides it: a band music, white noise noise, lena her musical opening then speech; in order, end to end; none for silence', async () => {
	for (let sr of [48000, 44100, 16000]) {
		let s = await segments(band(5, sr), sr)
		is(JSON.stringify(s), JSON.stringify([{ kind: 'music', start: 0, end: 5 }]), `${sr} Hz: the band is music, from 0 to its end`)
	}
	let seed = 7, rnd = () => (seed = (seed * 1664525 + 1013904223) % 4294967296, seed / 4294967296 - .5)
	let hiss = Float32Array.from({ length: 4 * 48000 }, () => .2 * rnd())
	is(JSON.stringify(await segments(hiss, 48000)), JSON.stringify([{ kind: 'noise', start: 0, end: 4 }]), 'white noise is noise')
	// lena: a film scene, music under its first words (README, Music): ina's own decision, a frame its nearest patch's
	let x = IN.lena.map(v => v / 32768), s = await segments(x, 48000), st = states(x, await guardNet(), 0)
	is(s.map(v => v.kind).join(), 'music,speech', `lena: ${s.map(v => `${v.kind} ${v.start}–${v.end.toFixed(2)} s`).join(', ')}`)
	ok(s[0].start === 0 && s.at(-1).end === x.length / 48000 && s.every((v, i) => !i || v.start === s[i - 1].end), 'in order, end to end, from 0 to the input\'s length')
	ok(s.every(v => [v.start, v.end - .01].every(t => KINDS[st[patchOf(Math.round(t * 100), st.length)]] === v.kind)), 'each stretch its frames\' nearest patches\' state, no prior for speech')
	ok(s[0].end > 1 && s[0].end < 4, `the opening ends at ${s[0].end} s`)
	let B = band(5)
	is((await segments(new Float32Array(48000), 48000)).length, 0, 'digital silence: none')
	is((await segments(new Float32Array(100).fill(.1), 48000)).length, 0, 'shorter than a frame: none')
	is((await segments(new Float32Array(0), 48000)).length, 0, 'empty: none')
	// silence within takes the kind around it
	let gap = new Float32Array(B.length + 48000 + B.length); gap.set(B); gap.set(B, B.length + 48000)
	is((await segments(gap, 48000)).map(v => v.kind).join(), 'music', 'a second of digital silence between two stretches of the band: music throughout')
	// a programme: the band, four noisy utterances, the band: the changes where the cuts are, to half a second
	if (!existsSync(VB)) return console.log('  (no VoiceBank+DEMAND in ~/.cache/audiojs/data: the programme skipped)')
	let S = readdirSync(VB).filter(f => f.endsWith('.wav')).sort().slice(0, 4).map(f => wav(path.join(VB, f))), n = S.reduce((a, v) => a + v.length, 0)
	let y = new Float32Array(2 * B.length + n), o = B.length
	y.set(B); for (let v of S) { y.set(v, o); o += v.length } y.set(B, o)
	let p = await segments(y, 48000), cuts = [5, o / 48000]
	is(p.map(v => v.kind).join(), 'music,speech,music', 'band, speech, band')
	ok(cuts.every((c, i) => Math.abs(p[i].end - c) < .5), `changes at ${p[0].end} and ${p[1].end} s, the cuts at ${cuts.map(c => c.toFixed(2)).join(' and ')} s`)
})

test('guard: the stream decides from what has arrived: RNNoise\'s output to frame f − 2 stays the same whatever follows frame f', async () => {
	let a = band(4), b = IN.lena.map(v => v / 32768).subarray(0, 2 * 48000), x = new Float32Array(a.length + b.length), y = new Float32Array(x.length)
	x.set(a); y.set(a); y.set(b.map(v => -v), a.length)
	let [u, v] = [await denoise(x, { sampleRate: 48000 }), await denoise(y, { sampleRate: 48000 })], f = a.length / 480
	is(maxDiff(u.subarray(0, (f - 2) * 480), v.subarray(0, (f - 2) * 480)), 0, 'the same up to the change, less RNNoise\'s 960 samples')
	ok(maxDiff(u, v) > .01, 'different after')
})

test('guard: the stream\'s decision takes its frame as a range of a longer array, as the manifest hands it (no view made), and decides as on the frame itself', async () => {
	let nn = await guardNet(), x = band(4), a = online(nn), b = online(nn), same = true, flips = 0, last = false
	for (let i = 0; i < x.length; i += 480) {
		let p = a(x.subarray(i, i + 480)), q = b(x, i, i + 480)
		same &&= p === q; flips += p !== last; last = p
	}
	ok(same, 'the same decision every frame')
	is(flips, 1, 'the band switches it to music, once')
})

test('guard: noisy speech is enhanced, no frame passed: VoiceBank+DEMAND training utterances (skipped without them)', async () => {
	if (!existsSync(VB)) return console.log('  (no VoiceBank+DEMAND in ~/.cache/audiojs/data: skipped)')
	let nn = await guardNet(), files = readdirSync(VB).filter(f => f.endsWith('.wav')).sort().filter((_, i) => i % 25 === 0)
	let passed = 0, frames = 0
	for (let f of files) {
		let x = wav(path.join(VB, f)), d = online(nn), on = []
		for (let i = 0; i < x.length + 960; i += 480) { let fr = new Float32Array(480); fr.set(x.subarray(i, Math.min(i + 480, x.length))); on.push(d(fr)) }
		let p = offline(x, nn); passed += p.reduce((s, v) => s + v, 0) + on.slice(2).reduce((s, v) => s + v, 0); frames += 2 * p.length
	}
	is(passed, 0, `${files.length} utterances, ${frames / 2} frames, offline and streaming: none passed`)
})

// a voice under a song's accompaniment as loud as itself (active levels, 10 ms frames within 35 dB of the 99th
// percentile): speaker p226's first five clean training utterances over a MUSDB18 training preview's drums, bass and
// other (scripts/accuracy.py guard-sets). 0.4 passed all 17.4 s of it as music, the voice left under the band
test('guard: a voice under a music bed as loud as itself is enhanced, offline: speech gains PRIOR a patch (skipped without the data)', async () => {
	let clean = VB.replace(/noisy$/, 'clean'), bed = path.join(VB, '..', '..', 'guard', 'train-instr', 'Bill Chudziak - Children Of No-one.f32')
	if (!existsSync(clean) || !existsSync(bed)) return console.log('  (no VoiceBank+DEMAND or guard sets in ~/.cache/audiojs/data: skipped)')
	let us = readdirSync(clean).filter(f => f.startsWith('p226')).sort().slice(0, 5).map(f => wav(path.join(clean, f)))
	let c = new Float32Array(us.reduce((n, u) => n + u.length, 0)); us.reduce((o, u) => (c.set(u, o), o + u.length), 0)
	let a = (await import('@audio/resample-sinc')).default(f32(bed), { from: 44100, to: 48000 }), b = c.map((_, i) => a[i % a.length])
	let active = x => { let p = []; for (let i = 0; i + 480 <= x.length; i += 480) p.push(x.subarray(i, i + 480).reduce((s, v) => s + v * v, 0) / 480); let q = [...p].sort((u, v) => u - v)[Math.floor(.99 * (p.length - 1))] * 10 ** -3.5, on = p.filter(v => v > q); return on.reduce((s, v) => s + v, 0) / on.length }
	let g = Math.sqrt(active(c) / active(b)), pass = offline(c.map((v, i) => v + g * b[i]), await guardNet()), share = pass.reduce((s, v) => s + v, 0) / pass.length
	ok(share < .5, `${(100 * share).toFixed(0)}% of its frames passed (0.4: 100%)`)
})

test('manifest: with music passing, the stream is denoise() sample for sample at 48 kHz, and at 44.1 kHz but the last 50 ms; music: \'enhance\' denoises it all', async () => {
	for (let sr of [48000, 44100]) {
		let a = band(3, sr), b = IN['lena+noise'].subarray(0, 2 * sr).map(v => v / 32768), x = new Float32Array(a.length + b.length + a.length)
		x.set(a); x.set(b, a.length); x.set(a, a.length + b.length)
		let { out: [y] } = await hosted(sr, [x], 16, () => 1000), cut = sr === 48000 ? x.length : x.length - sr / 20
		let want = await denoise(x, { sampleRate: sr })
		is(maxDiff(y.subarray(0, cut), want.subarray(0, cut)), 0, `${sr} Hz: equal to denoise()`)
		let tail = Math.round(2.5 * sr)
		is(maxDiff(y.subarray(tail, a.length), x.subarray(tail, a.length)), 0, `${sr} Hz: the band passes`)
		let { out: [e] } = await hosted(sr, [x], 16, () => 1000, 'enhance')
		is(maxDiff(e.subarray(0, cut), (await denoise(x, { sampleRate: sr, music: 'enhance' })).subarray(0, cut)), 0, `${sr} Hz, 'enhance': denoise() with music: 'enhance'`)
	}
})

// The input delayed by the latency, which is what passes, waits in a ring that holds the latency and a whole block: in
// blocks of maxBlockSize (4096, hosted()'s) it is as full as it gets, and a ring half that size gave back the wrong
// samples for 3156 to 4916 of them where music passed
test('manifest: with music passing in blocks of maxBlockSize (4096), the stream is denoise() sample for sample at 48 kHz, and at 44.1 kHz but the last 50 ms', async () => {
	for (let sr of [48000, 44100]) {
		let a = band(3, sr), b = IN['lena+noise'].subarray(0, 2 * sr).map(v => v / 32768), x = new Float32Array(a.length + b.length + a.length)
		x.set(a); x.set(b, a.length); x.set(a, a.length + b.length)
		let { out: [y] } = await hosted(sr, [x], 16, () => 4096), cut = sr === 48000 ? x.length : x.length - sr / 20
		is(maxDiff(y.subarray(0, cut), (await denoise(x, { sampleRate: sr })).subarray(0, cut)), 0, `${sr} Hz: equal to denoise()`)
		is(maxDiff(y.subarray(Math.round(2.5 * sr), a.length), x.subarray(Math.round(2.5 * sr), a.length)), 0, `${sr} Hz: the band passes`)
	}
})

test('manifest: its imports stop short of DeepFilterNet3 and the ONNX runtime, whose code writes `this` (the compiler to WebAssembly refused the manifest for it); weights() from rnnoise.js, the same bytes', async () => {
  // the modules the manifest reaches by static import within this package
  let seen = new Set(), bare = new Set()
  let walk = url => {
    if (seen.has(url.href)) return
    seen.add(url.href)
    for (let [, spec] of readFileSync(url, 'utf8').matchAll(/^\s*(?:import|export)\s[^'"]*?from\s*['"]([^'"]+)['"]/gm))
      spec.startsWith('.') ? walk(new URL(spec, url)) : bare.add(spec)
  }
  walk(new URL('./audio.js', import.meta.url))
  let files = [...seen].map(h => path.basename(new URL(h).pathname))
  ok(!files.includes('denoise.js') && !files.includes('deepfilter.js'), `audio.js reaches ${files.join(', ')}`)
  ok(![...bare].some(b => b.includes('neural-runtime') || b.includes('onnxruntime')), `and ${[...bare].join(', ')}`)
  let { weights: own } = await import('./rnnoise.js'), a = await own(), b = await weights()
  ok(a.length === 3544320 && a.every((v, i) => v === b[i]), 'the bundled weights, as denoise.js gives them')
})

test('denoise: empty, one sample, shorter than a frame, silence: the same length back, finite, silence kept (both models)', async () => {
	let dfn = await load('deepfilternet3', { weights: EXPORT, session: standIn() }), rnn = await load()
	for (let [name, model] of [['rnnoise', rnn], ['deepfilternet3', dfn]]) for (let rate of [48000, 16000]) {
		for (let n of [0, 1, 100]) {
			let y = await denoise(white(n, .1), { sampleRate: rate, model })
			ok(y.length === n && y.every(Number.isFinite), `${name} at ${rate}: ${n} samples in, ${y.length} out`)
		}
		let y = await denoise(new Float32Array(rate), { sampleRate: rate, model })
		ok(y.length === rate && y.every(v => v === 0), `${name} at ${rate}: 1 s of digital silence stays silent`)
	}
	dfn.free()
})

test('deepfilternet3: enhance() with upstream\'s settings against Python DeepFilterNet 0.5.6 on the same input (skipped without the model or the reference)', MODEL_RUN, async () => {
	if (!HAS_DFN) return console.log('  (no DeepFilterNet3 export in the cache: skipped)')
	let m = await load('deepfilternet3', { sessionOptions: { intraOpNumThreads: 4 } })
	try {
		// whole-file, as enhance() runs in Python (11.3 s: the default 10 s chunks would split it); gain 1, no voice guard
		let x = IN['lena+noise'].map(v => v / 32768), y = await enhance(x, m.net, { chunk: Infinity })
		is(y.length, x.length)
		let ref = path.join(DFN_REF, 'lena+noise')
		if (!existsSync(ref + '.out.f32')) return console.log('  (no scripts/deepfilter-reference.py output: compared nothing)')
		is(maxDiff(f32(ref + '.in.f32'), x), 0, 'the reference ran on this input')
		let r = f32(ref + '.out.f32'), s = snr(r, y)
		ok(s > 100, `SNR against Python ${s.toFixed(1)} dB, max |d| ${maxDiff(r, y).toExponential(1)}`)
		let z = await enhance(x, m.net, { chunk: 300, warmup: 300 })
		ok(snr(y, z) > 10, `in 3 s chunks: another GRU trajectory, ${snr(y, z).toFixed(1)} dB from the whole-file run`)
		// denoise() hears it with the speech (the louder half of 50 ms frames) at -20 dBFS, the voice guard on
		let p = []
		for (let i = 0; i + 2400 <= x.length; i += 2400) p.push(x.subarray(i, i + 2400).reduce((s, v) => s + v * v, 0) / 2400)
		p.sort((a, b) => b - a)
		let lvl = 10 * Math.log10(p.slice(0, p.length >> 1).reduce((s, v) => s + v, 0) / (p.length >> 1))
		let u = await denoise(x, { sampleRate: 48000, model: m, limit: Infinity, chunk: Infinity, music: 'enhance' }), v = await enhance(x, m.net, { chunk: Infinity, gain: 10 ** ((-20 - lvl) / 20), voice: true })
		ok(maxDiff(u, v) < 1e-6, `denoise() = enhance() heard at -20 dBFS (the speech at ${lvl.toFixed(1)}) with the voice guard`)
	} finally { m.free() }
})

// A voice through a telephone channel's band (300-3400 Hz, ITU-T G.712's passband: 4th-order Butterworth high- and
// low-pass, RBJ biquads) under white noise through it 10 dB down (re the voice's active level, its 20 ms frames within
// 40 dB of the loudest), as radio and phone takes carry it: VoiceBank+DEMAND's clean p232_003 (Valentini-Botinhao 2017,
// CC BY 4.0). At the default limit the voice loses no more than the limit: no more than 1 % of its 20 ms frames within
// 20 dB of its level (the louder half of those frames) more than 21 dB under it in the output. The model unlimited
// leaves 12 of those 188 frames under that, and so did 0.7's default, whose floor took the voice the model took down
// with the noise; now none
test('deepfilternet3: denoise() keeps a telephone-band voice under noise 10 dB down within its limit (skipped without the model or VoiceBank+DEMAND)', MODEL_RUN, async () => {
	let file = path.join(VB, '..', '..', 'vbdemand', 'clean_testset_wav', 'p232_003.wav')
	if (!HAS_DFN || !existsSync(file)) return console.log('  (no DeepFilterNet3 export or VoiceBank+DEMAND test set in the cache: skipped)')
	let bq = (x, hp, f0, Q) => { let w = 2 * Math.PI * f0 / 48000, c = Math.cos(w), al = Math.sin(w) / (2 * Q), a0 = 1 + al, b0 = (hp ? 1 + c : 1 - c) / 2 / a0, b1 = (hp ? -(1 + c) : 1 - c) / a0, a1 = -2 * c / a0, a2 = (1 - al) / a0, x1 = 0, x2 = 0, y1 = 0, y2 = 0; return x.map(v => { let y = b0 * v + b1 * x1 + b0 * x2 - a1 * y1 - a2 * y2; x2 = x1; x1 = v; y2 = y1; y1 = y; return y }) }
	let band = x => [.5412, 1.3066].reduce((y, Q) => bq(bq(y, true, 300, Q), false, 3400, Q), x)
	let s = band(wav(file)), n = band(white(s.length, 1, 5)), pw = (z, a, F) => z.subarray(a, a + F).reduce((p, v) => p + v * v, 0) / F
	let act = [], F = 960
	for (let a = 0; a + F <= s.length; a += F) act.push(pw(s, a, F))
	let top = Math.max(...act), on = act.filter(p => p >= top * 1e-4), k = Math.sqrt(on.reduce((p, v) => p + v, 0) / on.length / pw(n, 0, n.length) / 10)
	let x = s.map((v, i) => v + k * n[i]), y = await denoise(x, { sampleRate: 48000, model: 'deepfilternet3' })
	let lv = 10 * Math.log10(act.slice().sort((a, b) => b - a).slice(0, act.length >> 1).reduce((p, v) => p + v, 0) / (act.length >> 1)), frames = 0, lost = 0
	for (let a = 0; a + F <= s.length; a += F) if (10 * Math.log10(pw(s, a, F)) > lv - 20) { frames++; if (10 * Math.log10(pw(y, a, F) / pw(s, a, F)) < -21) lost++ }
	ok(lost <= frames / 100, `${lost} of ${frames} frames of the voice more than 21 dB under it`)
})
