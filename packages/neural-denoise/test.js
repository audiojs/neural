// Without downloads: RNNoise bit-exact against upstream's portable C build (fixtures/rnnoise.json, from
// scripts/rnnoise-reference.mjs), the frame API, offline alignment, the worklet's FIFO, and DeepFilterNet's
// pipeline (STFT, alignment, deep-filter taps, chunking) through an identity model. The little RNNoise
// model runs when the reference script has put it in the neural cache. With the DeepFilterNet3 export in
// the cache (the first denoise(…, { model: 'deepfilternet3' }) fetches it) the ONNX path runs; with
// scripts/deepfilter-reference.py's output there too, it is compared against Python DeepFilterNet.
import test, { ok, is, rejects, throws } from 'tst'
import { existsSync, readFileSync } from 'node:fs'
import { createHash } from 'node:crypto'
import os from 'node:os'
import path from 'node:path'
import denoise, { load, weights, rnnoise, FRAME, MODEL } from './denoise.js'
import { model } from './rnnoise.js'
import { loadDeepFilter, enhance, erbWidths } from './deepfilter.js'
import { rfft, irfft, WINDOW, N, HOP, BINS } from './fft.js'
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

test('rnnoise: malformed weights throw', async () => {
	let b = await weights()
	throws(() => model(b.subarray(0, 100)), /truncated|bad weight record/)
	let bad = b.slice()
	new DataView(bad.buffer).setInt32(12 + 64 + 99840, 508, true) // conv1_bias record: 508 bytes instead of 512
	throws(() => model(bad), /conv1_bias/)
	throws(() => model(new Uint8Array(0)), /missing/)
})

test('denoise: rnnoise offline, limit 0 = the frame API with its 960-sample delay removed; same length; a limit mixes the input back, 20 dB by default', async () => {
	let x = IN['lena+noise'].map(v => v / 32768), net = model(await weights())
	let pad = new Float32Array(Math.ceil((x.length + 960) / FRAME) * FRAME)
	pad.set(IN['lena+noise'])
	let { out } = frames(net, pad), y = await denoise(x, { sampleRate: 48000, limit: 0 })
	is(y.length, x.length)
	is(maxDiff(y, out.subarray(960, 960 + x.length).map(v => v / 32768)), 0, 'aligned sample for sample')
	for (let [opts, db] of [[{ limit: 6 }, 6], [{}, 20]]) {
		let z = await denoise(x, { sampleRate: 48000, ...opts }), lim = 10 ** (-db / 20)
		ok(maxDiff(z, y.map((v, i) => v * (1 - lim) + x[i] * lim)) < 1e-6, `${opts.limit ? 'limit' : 'default'} ${db} dB: ${lim.toFixed(2)} of the input mixed in`)
	}
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

test('worklet: 128-sample quanta, output = the frame API delayed by 448 samples (1408 in all), limit on the dry path, 20 dB by default', async () => {
	let procs = {}
	globalThis.sampleRate = 48000
	globalThis.AudioWorkletProcessor = class {}
	globalThis.registerProcessor = (name, cls) => procs[name] = cls
	await import('./worklet.js')
	let bytes = await weights(), x = IN['lena+noise'].subarray(0, 48000)
	let { out } = frames(model(bytes), x), want = new Float32Array(x.length)
	for (let i = 448; i < x.length; i++) want[i] = out[i - 448] / 32768
	for (let limit of [0, 12, undefined]) {
		let node = new procs['neural-denoise']({ processorOptions: { weights: bytes, limit } }), y = new Float32Array(x.length)
		for (let i = 0; i < x.length; i += 128) {
			let q = x.subarray(i, i + 128).map(v => v / 32768)
			node.process([[q]], [[y.subarray(i, i + 128)]])
		}
		let db = limit ?? 20, lim = db ? 10 ** (-db / 20) : 0
		let exp = want.map((v, i) => v * (1 - lim) + (i >= 1408 ? x[i - 1408] / 32768 : 0) * lim)
		ok(maxDiff(y, exp) < 1e-7, `limit ${limit ?? 'unset, 20'}: max |d| ${maxDiff(y, exp).toExponential(1)}`)
	}
	throws(() => { globalThis.sampleRate = 44100; new procs['neural-denoise']({ processorOptions: { weights: bytes } }) }, /48 kHz/)
	globalThis.sampleRate = 48000
})

// ------------------------------------------------ audio.js manifest

// the contract atom hosted as hosts run it: blocks of `size(k)` samples, then its declared latency of silence
async function hosted(sr, chs, limit, size) {
	let { rnnoise: atom } = await import('./audio.js')
	let D = atom.latency({ sampleRate: sr, params: {} }), params = { limit: new Float32Array([limit]) }
	let proc = atom({ sampleRate: sr, maxBlockSize: 4096, maxChannels: 32, params }), n = chs[0].length + D
	let inp = chs.map(x => { let v = new Float32Array(n); v.set(x); return v }), out = chs.map(() => new Float32Array(n))
	for (let p = 0, k = 0, m; p < n; p += m) { m = Math.min(n - p, size(k++)); proc([inp.map(v => v.subarray(p, p + m))], [out.map(v => v.subarray(p, p + m))], params) }
	return { D, out: out.map(v => v.subarray(D)) }
}

test('manifest: rnnoise streams denoise() sample for sample at 48 kHz, 1439 samples late (960 + a 479-sample queue), in blocks of any size', async () => {
	let x = IN['lena+noise'].subarray(0, 3 * 48000).map(v => v / 32768), s = 3
	let rnd = () => (s = (s * 1664525 + 1013904223) % 4294967296, s / 4294967296)
	for (let limit of [20, 0]) {
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
// stand-ins: ERB gains `gain`, one deep-filter tap of 1 at `tap` (2 = the current frame)
function standIn({ gain = 1, tap = 2 } = {}) {
	let t = (a, dims) => ({ data: a, dims, type: 'float32' })
	return async bytes => {
		let kind = new TextDecoder().decode(bytes)
		return {
			async run(feeds) {
				if (kind === 'enc') { let S = feeds.feat_erb.dims[2], z = n => new Float32Array(n); return { e0: t(z(64 * S * 32), [1, 64, S, 32]), e1: t(z(64 * S * 16), [1, 64, S, 16]), e2: t(z(64 * S * 8), [1, 64, S, 8]), e3: t(z(64 * S * 8), [1, 64, S, 8]), emb: t(z(S * 512), [1, S, 512]), c0: t(z(64 * S * 96), [1, 64, S, 96]), lsnr: t(z(S), [1, S, 1]) } }
				let S = feeds.emb.dims[1]
				if (kind === 'erb_dec') return { m: t(new Float32Array(S * 32).fill(gain), [1, 1, S, 32]) }
				let c = new Float32Array(S * 96 * 10)
				for (let i = 0; i < S * 96; i++) c[i * 10 + 2 * tap] = 1
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

test('deepfilter: denoise() limits DeepFilterNet3 to 12 dB unless given a limit; 0 is upstream\'s unlimited output', async () => {
	let m = await load('deepfilternet3', { weights: EXPORT, session: standIn({ gain: 0 }) }) // removes everything above 4.8 kHz
	let x = IN['lena+noise'].subarray(0, 48000).map(v => v / 32768), run = opts => denoise(x, { sampleRate: 48000, model: m, ...opts })
	let [unset, twelve, none] = [await run({}), await run({ limit: 12 }), await run({ limit: 0 })]
	is(maxDiff(unset, twelve), 0, 'default = limit 12')
	ok(maxDiff(unset, none) > 1e-3, `limit 0 differs by up to ${maxDiff(unset, none).toExponential(1)}`)
	m.free()
})

test('deepfilter: export errors', async () => {
	await rejects(() => loadDeepFilter(tar({ 'config.ini': CONFIG }), { session: standIn() }), /enc\.onnx/)
	await rejects(() => loadDeepFilter(tar({ 'enc.onnx': 'enc', 'erb_dec.onnx': 'erb_dec', 'df_dec.onnx': 'df_dec', 'config.ini': CONFIG.replace('deepfilternet3', 'deepfilternet2') }), { session: standIn() }), /deepfilternet2/)
	await rejects(() => loadDeepFilter(tar({ 'enc.onnx': 'enc', 'erb_dec.onnx': 'erb_dec', 'df_dec.onnx': 'df_dec', 'config.ini': CONFIG.replace('sr = 48000', 'sr = 16000') }), { session: standIn() }), /48 kHz/)
})

test('deepfilternet3: against Python DeepFilterNet 0.5.6 on the same input (skipped without the model or the reference)', MODEL_RUN, async () => {
	if (!HAS_DFN) return console.log('  (no DeepFilterNet3 export in the cache: skipped)')
	let m = await load('deepfilternet3', { sessionOptions: { intraOpNumThreads: 4 } })
	try {
		// whole-file, as enhance() runs in Python (11.3 s: the default 10 s chunks would split it)
		let x = IN['lena+noise'].map(v => v / 32768), y = await denoise(x, { sampleRate: 48000, model: m, limit: 0, chunk: Infinity })
		is(y.length, x.length)
		let ref = path.join(DFN_REF, 'lena+noise')
		if (!existsSync(ref + '.out.f32')) return console.log('  (no scripts/deepfilter-reference.py output: compared nothing)')
		is(maxDiff(f32(ref + '.in.f32'), x), 0, 'the reference ran on this input')
		let r = f32(ref + '.out.f32'), s = snr(r, y)
		ok(s > 100, `SNR against Python ${s.toFixed(1)} dB, max |d| ${maxDiff(r, y).toExponential(1)}`)
		let z = await denoise(x, { sampleRate: 48000, model: m, limit: 0, chunk: 300, warmup: 300 })
		ok(snr(y, z) > 10, `in 3 s chunks: another GRU trajectory, ${snr(y, z).toFixed(1)} dB from the whole-file run`)
	} finally { m.free() }
})
