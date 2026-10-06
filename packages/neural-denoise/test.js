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
import denoise, { load, weights, rnnoise, mixback, FRAME, MODEL } from './denoise.js'
import { model } from './rnnoise.js'
import { loadDeepFilter, enhance, erbWidths, voicing, bandEdge, LEVEL, FILL, FLOOR } from './deepfilter.js'
import { rfft, irfft, WINDOW, N, HOP, BINS } from './fft.js'
import { guardNet, net as guardModel, analyzer, offline, online, gains, RAMP } from './guard.js'
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

test('denoise: rnnoise offline, limit 0 = the frame API with its 960-sample delay removed; same length; a limit mixes the input back, 16 dB by default', async () => {
	let x = IN['lena+noise'].map(v => v / 32768), net = model(await weights())
	let pad = new Float32Array(Math.ceil((x.length + 960) / FRAME) * FRAME)
	pad.set(IN['lena+noise'])
	let { out } = frames(net, pad), y = await denoise(x, { sampleRate: 48000, limit: 0 })
	is(y.length, x.length)
	is(maxDiff(y, out.subarray(960, 960 + x.length).map(v => v / 32768)), 0, 'aligned sample for sample')
	for (let [opts, db] of [[{ limit: 6 }, 6], [{}, 16]]) {
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

test('worklet: 128-sample quanta, output = the frame API delayed by 448 samples (1408 in all), limit on the dry path, 16 dB by default', async () => {
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
		let db = limit ?? 16, lim = db ? 10 ** (-db / 20) : 0
		let exp = want.map((v, i) => v * (1 - lim) + (i >= 1408 ? x[i - 1408] / 32768 : 0) * lim)
		ok(maxDiff(y, exp) < 1e-7, `limit ${limit ?? 'unset, 16'}: max |d| ${maxDiff(y, exp).toExponential(1)}`)
	}
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

test('deepfilter: denoise() limits DeepFilterNet3 to 18 dB unless given a limit; 0 is none', async () => {
	let m = await load('deepfilternet3', { weights: EXPORT, session: standIn({ gain: 0 }) }) // removes everything above 4.8 kHz
	let x = IN['lena+noise'].subarray(0, 48000).map(v => v / 32768), run = opts => denoise(x, { sampleRate: 48000, model: m, music: 'enhance', ...opts })
	let [unset, eighteen, none] = [await run({}), await run({ limit: 18 }), await run({ limit: 0 })]
	is(maxDiff(unset, eighteen), 0, 'default = limit 18')
	ok(maxDiff(unset, none) > 1e-3, `limit 0 differs by up to ${maxDiff(unset, none).toExponential(1)}`)
	is(maxDiff(unset, mixback(x, none)), 0, 'the unlimited output with the input mixed back by mixback()')
	is(maxDiff(await run({ floor: 0 }), mixback(x, none, { floor: 0 })), 0, 'floor 0: a constant 18 dB')
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

// a perfect model (y the voice, a 220 Hz tone) and white noise `db` dB under the voice: how far under the voice
// mixback() leaves the noise
test('deepfilter: mixback() drops quiet noise by the limit and takes loud noise down to the floor under the voice, frame by frame', () => {
	let n = 4 * 48000, s = Float32Array.from({ length: n }, (_, i) => .1 * Math.sin(2 * Math.PI * 220 * i / 48000)), w = white(n, 1, 3)
	let k = db => .1 / Math.SQRT2 * 10 ** (-db / 20) * Math.sqrt(12), under = (r, a = 0, b = n) => span(s, a, b) - span(r, a, b)
	let left = (db, o) => under(mixback(s.map((v, i) => v + k(db) * w[i]), s, o).map((v, i) => v - s[i]))
	ok(Math.abs(left(40) - 58) < .2, `noise 40 dB under the voice: ${left(40).toFixed(1)} dB under, the limit's 18 dB down`)
	ok(Math.abs(left(25) - 43) < .2, `25 dB under: ${left(25).toFixed(1)}, 18 down`)
	ok(Math.abs(left(10) - FLOOR) < .2, `10 dB under: ${left(10).toFixed(1)}, at the floor (${FLOOR})`)
	ok(Math.abs(left(-5) - FLOOR) < .2, `5 dB over: ${left(-5).toFixed(1)}, at the floor`)
	ok(Math.abs(left(10, { floor: 0 }) - 28) < .2 && Math.abs(left(10, { floor: 30 }) - 30) < .2, 'floor 0: 18 dB down, as before 0.5; floor 30: 30 under')
	ok(Math.abs(left(40, { limit: 6 }) - 46) < .2 && Math.abs(left(10, { limit: 6 }) - FLOOR) < .2, 'limit 6: quiet noise 6 dB down, loud noise at the floor')
	is(maxDiff(mixback(s.map((v, i) => v + w[i]), s, { limit: 0 }), s), 0, 'limit 0: the model\'s output')
	// loud noise for 2 s, then quiet: each as above, 0.5 s from the change
	let x = s.map((v, i) => v + k(i < n / 2 ? 10 : 40) * w[i]), r = mixback(x, s).map((v, i) => v - s[i])
	let a = under(r, 0, 72000), b = under(r, 120000, n)
	ok(Math.abs(a - FLOOR) < .3 && Math.abs(b - 58) < .3, `the noise drops 30 dB: ${a.toFixed(1)} then ${b.toFixed(1)} dB under the voice`)
	let y = s.slice(); y.set(x.subarray(0, 48000))
	is(maxDiff(mixback(x, y).subarray(0, 48000), x.subarray(0, 48000)), 0, 'where input and output agree (music passed): the input, bit for bit')
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

test('guard: the stream decides from what has arrived: RNNoise\'s output to frame f − 2 stays the same whatever follows frame f', async () => {
	let a = band(4), b = IN.lena.map(v => v / 32768).subarray(0, 2 * 48000), x = new Float32Array(a.length + b.length), y = new Float32Array(x.length)
	x.set(a); y.set(a); y.set(b.map(v => -v), a.length)
	let [u, v] = [await denoise(x, { sampleRate: 48000 }), await denoise(y, { sampleRate: 48000 })], f = a.length / 480
	is(maxDiff(u.subarray(0, (f - 2) * 480), v.subarray(0, (f - 2) * 480)), 0, 'the same up to the change, less RNNoise\'s 960 samples')
	ok(maxDiff(u, v) > .01, 'different after')
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
		let u = await denoise(x, { sampleRate: 48000, model: m, limit: 0, chunk: Infinity, music: 'enhance' }), v = await enhance(x, m.net, { chunk: Infinity, gain: 10 ** ((-20 - lvl) / 20), voice: true })
		ok(maxDiff(u, v) < 1e-6, `denoise() = enhance() heard at -20 dBFS (the speech at ${lvl.toFixed(1)}) with the voice guard`)
	} finally { m.free() }
})
