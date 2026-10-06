// DeepFilterNet3: H. Schröter, T. Rosenkranz, A. N. Escalante-B., A. Maier, "DeepFilterNet: Perceptually
// Motivated Real-Time Speech Enhancement", Interspeech 2023 (arXiv:2305.08227). Two stages on a 10 ms
// STFT at 48 kHz: gains on 32 ERB bands for the envelope, then a complex 5-tap filter over time on each
// of the lowest 96 bins (up to 4.8 kHz) for the periodic structure. Upstream's three ONNX graphs
// (encoder, ERB decoder, deep-filter decoder) run through @audio/neural-runtime.
//
// Copyright (c) 2021 Hendrik Schröter. MIT License (upstream is MIT OR Apache-2.0); the notice is in NOTICE.
//
// Changes, 2026, audiojs: translated to JavaScript from Rikorose/DeepFilterNet at d375b2d: libDF's STFT,
// ERB filterbank and feature normalizations (libDF/src/lib.rs, transforms.rs) and df.enhance.enhance()
// with DfNet.forward (DeepFilterNet/df/enhance.py, deepfilternet3.py, modules.py, multiframe.py): the
// conv_lookahead shift, the ERB mask, the multi-frame deep filter, the attenuation limit. Model
// parameters are read from the export's config.ini. The FFT is RNNoise's (fft.js) in float32.
// Inputs longer than `chunk` frames run in chunks: upstream feeds the whole file, about 8 MB of
// activations per second of audio. Each chunk's run starts `warmup` frames early from zero GRU state
// (those outputs dropped) and continues `fade` frames past its end, where the next chunk crossfades in
// (raised cosine). The GRUs keep a long memory of their start, so chunked output is another valid
// trajectory, not the whole-file one (40 to 55 dB SNR from it on a 60 s narration; longer warm-ups don't
// converge); the STFT and the features stay exact across chunks.
//
// Added, for denoise(): the network can hear its input at another level (`gain`, its features only: they are
// not level-free, the ERB means start at fixed dB and the complex features scale with the root of the level),
// its features can hear a floor of white noise above the input's band edge (`edge`: a band-limited input leaves
// bands empty that the model, trained on full-band mixtures only, never heard empty), and a voice guard (`voice`)
// keeps the sustained voicing it takes for noise: held sung notes, chant.

import { load, tensor, fetchModel } from '@audio/neural-runtime'
import { WINDOW, rfft, irfft, N as FFT_N, HOP as FFT_HOP, BINS } from './fft.js'
import { pitch } from './rnnoise.js'

const f = Math.fround

// models/DeepFilterNet3_onnx.tar.gz at d375b2d (7,983,136 bytes, sha256 c94d91f7…): enc, erb_dec, df_dec, config.ini
export const MODEL = 'https://raw.githubusercontent.com/Rikorose/DeepFilterNet/d375b2d8309e0935d165700c91da9de862a99c31/models/DeepFilterNet3_onnx.tar.gz'

// The attenuation limit, dB, that denoise() applies to this model unless given one (0: none): the most before the
// voice itself suffers. On VoiceBank+DEMAND, DNSMOS SIG holds from 12 to 18 dB and falls past it, BAK and PESQ
// rise; unlimited, the pauses of home narrations fall to digital silence (README, Accuracy).
export const LIMIT = 18

// The floor, dB under the voice, to which denoise() takes noise the limit would leave closer to it (mixback() in
// denoise.js): the limit alone keeps room tone, and keeps noise as loud as the voice 18 dB under it. 40 dB under a voice
// at -20 dBFS is ACX's -60 dBFS floor; on audio's bench/rx/isolate.mjs tune split 35, 40 and 45 score PESQ 1.96, 1.98,
// 1.98 (0: 1.75), on VoiceBank+DEMAND's training speakers DNSMOS SIG 3.38, 3.36, 3.35 (0: 3.41) (README, Measured).
export const FLOOR = 40

// The level, dBFS, at which denoise() has the network hear the speech (`gain`): the median of the VoiceBank+DEMAND
// test set it scores its published PESQ on (level(), -20.2); its features are not level-free (README, DeepFilterNet3).
export const LEVEL = -20

// The floor the features hear in the bands above the input's band edge, dB under the speech: white noise, as the
// model always heard there. It trained on full-band 48 kHz mixtures only (config.ini: p_bandwidth_ext 0, libDF's
// BandwidthLimiterAugmentation off; noise at -5 to 40 dB SNR), so a band a 16 kHz file or a codec's low-pass left
// empty reached it as libDF's constant 1e-10 floor, without the trace of noise every mixture it knew had, and it
// cleaned the band below worse: on the VoiceBank+DEMAND training subset at 16 kHz, unlimited, PESQ 2.26 against
// 2.57 full-band. Over a floor at -40, -30, -20 and -10 dB: 2.30, 2.37, 2.42, 2.44; low-passed at 16 kHz, 2.42 and
// over -20 dB 2.51; at 8 kHz, 1.73 and 2.09. -20 dB, noise in the middle of the SNRs it trained on, takes most of
// the gain (README, Accuracy).
export const FILL = -20

const isNode = typeof process !== 'undefined' && !!process.versions?.node

// ------------------------------------------------ model archive

async function gunzip(bytes) {
	if (isNode) return new Uint8Array((await import('node:zlib')).gunzipSync(bytes))
	let stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'))
	return new Uint8Array(await new Response(stream).arrayBuffer())
}

// ustar: 512-byte headers, octal sizes, data padded to 512
function untar(bytes) {
	let files = {}, o = 0, str = (at, n) => new TextDecoder().decode(bytes.subarray(at, at + n)).replace(/\0.*$/s, '')
	while (o + 512 <= bytes.length && bytes[o]) {
		let name = str(o, 100), size = parseInt(str(o + 124, 12).trim() || '0', 8), prefix = str(o + 345, 155)
		if (bytes[o + 156] === 48 || bytes[o + 156] === 0) files[(prefix ? prefix + '/' : '') + name] = bytes.subarray(o + 512, o + 512 + size)
		o += 512 + Math.ceil(size / 512) * 512
	}
	return files
}

function ini(text) {
	let out = {}, sec = out
	for (let line of text.split(/\r?\n/)) {
		let s = line.match(/^\s*\[(.+)\]\s*$/), kv = line.match(/^\s*([^=#;]+?)\s*=\s*(.*?)\s*$/)
		if (s) sec = out[s[1]] = {}
		else if (kv) sec[kv[1]] = kv[2]
	}
	return out
}

// load(model?) → { enc, erb, df, params }: the upstream export (tar.gz: enc.onnx, erb_dec.onnx, df_dec.onnx, config.ini)
export async function loadDeepFilter(model = MODEL, opts = {}) {
	let bytes = typeof model === 'string' ? await fetchModel(model, opts) : new Uint8Array(model)
	if (bytes[0] === 0x1f && bytes[1] === 0x8b) bytes = await gunzip(bytes)
	let files = untar(bytes), pick = n => Object.entries(files).find(([k]) => k.endsWith('/' + n) || k === n)?.[1]
	let [enc, erb, df, cfg] = ['enc.onnx', 'erb_dec.onnx', 'df_dec.onnx', 'config.ini'].map(pick)
	if (!enc || !erb || !df || !cfg) throw new Error('neural-denoise: a DeepFilterNet export needs enc.onnx, erb_dec.onnx, df_dec.onnx and config.ini')
	let c = ini(new TextDecoder().decode(cfg)), d = c.df, m = c.deepfilternet
	if (c.train?.model !== 'deepfilternet3') throw new Error(`neural-denoise: model type '${c.train?.model}' is not deepfilternet3`)
	let params = {
		sr: +d.sr, n: +d.fft_size, hop: +d.hop_size, nbErb: +d.nb_erb, nbDf: +d.nb_df, order: +(d.df_order ?? m.df_order),
		dfLa: +(d.df_lookahead ?? m.df_lookahead), convLa: +m.conv_lookahead, tau: +d.norm_tau, minErb: +d.min_nb_erb_freqs,
	}
	if (params.n !== FFT_N || params.hop !== FFT_HOP || params.sr !== 48000) throw new Error('neural-denoise: only 48 kHz models with 960/480 framing are supported')
	params.widths = erbWidths(params.sr, params.n, params.nbErb, params.minErb)
	params.alpha = normAlpha(params.sr, params.hop, params.tau)
	let ld = opts.session ?? (b => load(b, { backend: opts.device, sessionOptions: opts.sessionOptions }))
	let e = await ld(enc), r = await ld(erb), g = await ld(df)
	return { enc: e, erb: r, df: g, params, free() { e.free(); r.free(); g.free() } }
}

// ------------------------------------------------ libDF features

const freq2erb = hz => f(f(9.265) * f(Math.log1p(f(hz / f(f(24.7) * f(9.265))))))
const erb2freq = e => f(f(f(24.7) * f(9.265)) * f(f(Math.exp(f(e / f(9.265)))) - 1))

// erb_fb: bins per band, equal ERB steps, at least `minNb` bins per band, all bins covered
export function erbWidths(sr, n, bands, minNb) {
	let width = f(sr / n), lo = freq2erb(0), hi = freq2erb(sr >> 1), step = f(f(hi - lo) / bands)
	let erb = new Array(bands).fill(0), prev = 0, over = 0
	for (let i = 1; i <= bands; i++) {
		let fb = Math.round(f(erb2freq(f(lo + f(i * step))) / width)), nb = fb - prev - over
		if (nb < minNb) { over = minNb - nb; nb = minNb } else over = 0
		erb[i - 1] = nb; prev = fb
	}
	erb[bands - 1] += 1
	let extra = erb.reduce((a, b) => a + b, 0) - (n / 2 + 1)
	if (extra > 0) erb[bands - 1] -= extra
	return erb
}

// df.utils.get_norm_alpha: exp(−hop/sr/τ), rounded to the fewest decimals (from 3) that keep it below 1
function normAlpha(sr, hop, tau) {
	let a = Math.exp(-hop / sr / tau), out = 1
	for (let p = 3; out >= 1; p++) out = Math.round(a * 10 ** p) / 10 ** p
	return f(out)
}

// ndarray's linspace, in f32 as libDF builds its initial normalization states
const linspace = (a, b, n) => { let step = f(f(f(b) - f(a)) / (n - 1)); return Float32Array.from({ length: n }, (_, i) => f(f(a) + f(step * i))) }

// ------------------------------------------------ input analysis

// level(x, rate) → the speech level, dBFS: mean power of the louder half of 50 ms frames (LEVEL where there is none)
export function level(x, rate = 48000) {
	let F = Math.round(rate / 20), p = []
	for (let i = 0; i + F <= x.length; i += F) { let s = 0; for (let j = i; j < i + F; j++) s += x[j] * x[j]; p.push(s / F) }
	p.sort((a, b) => b - a)
	let k = Math.max(1, p.length >> 1), s = 0
	for (let i = 0; i < k && i < p.length; i++) s += p[i]
	return s > 0 ? 10 * Math.log10(s / k) : LEVEL
}

// bandEdge(x, rate) → Hz: the input's band edge, the highest frequency whose long-term power lies within 60 dB of the
// 1 to 4 kHz median: the power of 960-point frames end to end, averaged, at the input's own rate, before any
// resampling, so a resampler's images don't count; rate/2 for a full band, and where nothing is heard from 1 to 4 kHz.
export function bandEdge(x, rate) {
	let P = new Float64Array(BINS), buf = new Float32Array(FFT_N), re = new Float32Array(BINS), im = new Float32Array(BINS)
	for (let a = 0; a + FFT_N <= x.length; a += FFT_N) {
		for (let i = 0; i < FFT_N; i++) buf[i] = x[a + i] * WINDOW[i]
		rfft(buf, re, im)
		for (let k = 0; k < BINS; k++) P[k] += re[k] * re[k] + im[k] * im[k]
	}
	let k0 = Math.ceil(1000 * FFT_N / rate), k1 = Math.min(BINS - 1, Math.floor(4000 * FFT_N / rate))
	let mid = Array.from(P.subarray(k0, k1 + 1)).sort((a, b) => a - b)[(k1 - k0 + 1) >> 1], e = BINS - 1
	if (!(mid > 0)) return rate / 2
	while (e > 0 && !(P[e] >= mid * 1e-6)) e--
	return e * rate / FFT_N
}

// ------------------------------------------------ enhance

// voicing(x, T, hop) → { period, on }: per frame, RNNoise's pitch period over the frame's 20 ms window, and `on`
// inside sustained voicing: runs of 0.3 s or more whose normalized correlation at the period is 0.45 or more
// (Praat's voicing threshold, Boersma 1993) and whose 100 ms stand 6 dB or more above the quietest 100 ms of the
// 20 s around (minimum statistics, Martin 2001: that minimum lies 4 to 11 dB under the mean of the DEMAND
// noises), digital silence (under -90 dBFS) left out. A held note stands over the background; a buzz is in it.
export function voicing(x, T, hop) {
	let L = x.length, pt = pitch(), frame = new Float32Array(hop), period = new Int16Array(T), on = new Uint8Array(T)
	// power per 100 ms of input, and the least of it over the 20 s around (blocks of digital silence left out)
	let K = Math.ceil(T / 10), e = new Float64Array(K), low = new Float64Array(K).fill(Infinity)
	for (let i = 0; i < L; i++) e[i / (10 * hop) | 0] += x[i] * x[i]
	for (let k = 0; k < K; k++) e[k] /= Math.max(1, Math.min(L, (k + 1) * 10 * hop) - k * 10 * hop)
	for (let k = 0; k < K; k++) if (e[k] > 1e-9) for (let j = Math.max(0, k - 100); j <= Math.min(K - 1, k + 100); j++) low[j] = Math.min(low[j], e[k])
	// the pitch searched every 20 ms where the frame stands over the background (every 10 ms keeps held notes no better)
	for (let t = 0; t < T; t++) {
		for (let i = 0, j = t * hop; i < hop; i++, j++) frame[i] = j < L ? x[j] * 32768 : 0
		let over = e[t / 10 | 0] >= 4 * low[t / 10 | 0]
		period[t] = pt.push(frame, over && !(t & 1))
		on[t] = over && pt.gain >= .45
	}
	for (let t = 0, r = 0; t <= T; t++) {
		if (t < T && on[t]) { r++; continue }
		if (r < 30) on.fill(0, t - r, t)
		r = 0
	}
	return { period, on }
}

// enhance(x, net, opts) → Float32Array: one channel at 48 kHz, as df.enhance.enhance(model, state, x, pad=True).
// `gain` scales the input the network hears (its features), not the spectrum its gains and filters apply to;
// `edge` (Hz, the input's band edge) has the ERB bands wholly above it hear white noise FILL dB under speech at LEVEL;
// `voice` keeps sustained voicing (voicing()) the network would remove.
export async function enhance(x, net, { limit, chunk = 1000, warmup = 300, fade = 50, gain = 1, edge = Infinity, voice = false } = {}) {
	let { sr, hop, n, nbErb, nbDf, order, dfLa, convLa, widths, alpha } = net.params
	let L = x.length, T = Math.floor((L + n) / hop), lead = order - 1 - dfLa, look = Math.max(convLa, dfLa)
	let lim = limit != null && Math.abs(limit) > 0 ? 10 ** (-Math.abs(limit) / 20) : 0, g1 = f(gain), g2 = f(gain * gain)
	let vo = voice && voicing(x, T, hop)
	// per-frame store, a ring big enough for one chunk with its context
	chunk = Math.max(1, Math.min(chunk, T)); fade = Math.min(fade, chunk)
	let cap = chunk + fade + warmup + look + lead + 2
	let Sr = new Float32Array(cap * BINS), Si = new Float32Array(cap * BINS), FE = new Float32Array(cap * nbErb), FC = new Float32Array(cap * nbDf * 2)
	let mean = linspace(-60, -90, nbErb), unit = linspace(0.001, 0.0001, nbDf), a1 = f(1 - alpha)
	let mem = new Float32Array(hop), buf = new Float32Array(n), made = 0
	let erbStart = new Int32Array(nbErb), erbK = new Float32Array(nbErb)
	for (let b = 0, s = 0; b < nbErb; s += widths[b++]) { erbStart[b] = s; erbK[b] = f(1 / widths[b]) }
	// the floor, in the bands from `empty` on, the first wholly above the edge (a band holding the edge has the input's
	// own content): white Gaussian noise of power 10^((LEVEL + FILL)/10) per sample has |X_k|² = that / 2n · E per bin,
	// E exponential of mean 1 (the window's Σw² is n/2, the FFT scaled by 1/n), drawn for every bin in every frame from
	// a fixed seed (32-bit LCG): a floor the network hears move, as noise does, not a constant its mean normalization
	// takes to zero. fill(b) → its power per bin in band b this frame
	let empty = erbStart.findIndex(s => s * sr / n > edge), floor = 10 ** ((LEVEL + FILL) / 10) / (2 * n), seed = 1
	let fill = b => {
		if (empty < 0 || b < empty) return 0
		let p = 0
		for (let j = 0; j < widths[b]; j++) p -= Math.log(((seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0) + .5) / 2 ** 32)
		return f(floor * p / widths[b])
	}

	// frame_analysis + feat_erb + feat_cplx for frames up to t (sequential: the normalizations carry state)
	function make(upto) {
		for (; made <= upto && made < T; made++) {
			let t = made, at = t * hop, slot = t % cap, o = slot * BINS
			for (let i = 0; i < hop; i++) buf[i] = f(mem[i] * WINDOW[i])
			for (let i = 0; i < hop; i++) { let v = at + i < L ? x[at + i] : 0; buf[hop + i] = f(v * WINDOW[hop + i]); mem[i] = v }
			rfft(buf, Sr.subarray(o, o + BINS), Si.subarray(o, o + BINS))
			for (let b = 0; b < nbErb; b++) {
				let s = 0, k = erbK[b]
				for (let j = erbStart[b], e = j + widths[b]; j < e; j++) s = f(s + f(f(f(Sr[o + j] * Sr[o + j]) + f(Si[o + j] * Si[o + j])) * k))
				let v = f(f(Math.log10(f(f(f(s * g2) + fill(b)) + f(1e-10)))) * 10)
				mean[b] = f(f(v * a1) + f(mean[b] * alpha))
				FE[slot * nbErb + b] = f(f(v - mean[b]) / 40)
			}
			for (let j = 0; j < nbDf; j++) {
				let re = f(Sr[o + j] * g1), im = f(Si[o + j] * g1)
				unit[j] = f(f(f(Math.hypot(re, im)) * a1) + f(unit[j] * alpha))
				let d = f(Math.sqrt(unit[j]))
				FC[slot * nbDf * 2 + j] = f(re / d); FC[slot * nbDf * 2 + nbDf + j] = f(im / d)
			}
		}
	}

	// sustained voicing: a band holding a harmonic of the pitch keeps at least c² of its input, c its normalized
	// correlation with the frame one period earlier and two (the less): the periodic share of its power, as RNNoise's
	// pitch filter restores what its band gains remove (Valin 2018). Er, Ei is the output so far.
	let P = [[new Float32Array(BINS), new Float32Array(BINS)], [new Float32Array(BINS), new Float32Array(BINS)]]
	function keep(t, o) {
		let T0 = vo.period[t], f0 = n / T0
		for (let k = 0; k < 2; k++) {
			for (let i = 0, j = (t - 1) * hop - (k + 1) * T0; i < n; i++, j++) buf[i] = f((j >= 0 && j < L ? x[j] : 0) * WINDOW[i])
			rfft(buf, P[k][0], P[k][1])
		}
		for (let b = 0; b < nbErb; b++) {
			let j0 = erbStart[b], j1 = j0 + widths[b]
			// h·f0, h ≥ 1, within the band's bins (sr/n apart)
			if (Math.floor((j1 - .5) / f0) < Math.max(1, Math.ceil((j0 - .5) / f0))) continue
			let xx = 0, ee = 0, c = 1
			for (let j = j0; j < j1; j++) { xx += Sr[o + j] ** 2 + Si[o + j] ** 2; ee += Er[j] ** 2 + Ei[j] ** 2 }
			for (let [Pr, Pi] of P) {
				let xy = 0, yy = 0
				for (let j = j0; j < j1; j++) { xy += Sr[o + j] * Pr[j] + Si[o + j] * Pi[j]; yy += Pr[j] ** 2 + Pi[j] ** 2 }
				c = Math.min(c, xy / Math.sqrt(xx * yy + 1e-30))
			}
			let p = c > 0 ? c * c : 0, g = Math.min(1, Math.sqrt(ee / (xx + 1e-30)))
			if (p <= g) continue
			// mixing a of the input in takes the band's gain from g to p
			let a = (p - g) / (1 - g)
			for (let j = j0; j < j1; j++) { Er[j] = f(f(Sr[o + j] * a) + f(Er[j] * (1 - a))); Ei[j] = f(f(Si[o + j] * a) + f(Ei[j] * (1 - a))) }
		}
	}

	let out = new Float32Array(L), syn = new Float32Array(hop), y = new Float32Array(n), d = n - hop
	let Er = new Float32Array(BINS), Ei = new Float32Array(BINS)
	// a chunk's run continues `fade` frames past its end; the next chunk crossfades from those spectra
	let tailR = new Float32Array(fade * BINS), tailI = new Float32Array(fade * BINS)
	for (let a = 0, b; a < T; a = b) {
		b = Math.min(T, a + chunk)
		let e = Math.min(T, b + fade), a0 = Math.max(0, a - warmup), S = e - a0
		make(Math.min(T - 1, e - 1 + look))
		// network input: features shifted by conv_lookahead (DfNet.pad_feat), zeros past the end
		let fe = new Float32Array(S * nbErb), fc = new Float32Array(2 * S * nbDf)
		for (let s = 0; s < S; s++) {
			let t = a0 + s + convLa
			if (t >= T) continue
			let slot = t % cap
			fe.set(FE.subarray(slot * nbErb, slot * nbErb + nbErb), s * nbErb)
			fc.set(FC.subarray(slot * nbDf * 2, slot * nbDf * 2 + nbDf), s * nbDf)
			fc.set(FC.subarray(slot * nbDf * 2 + nbDf, slot * nbDf * 2 + 2 * nbDf), S * nbDf + s * nbDf)
		}
		let enc = await net.enc.run({ feat_erb: tensor(fe, [1, 1, S, nbErb]), feat_spec: tensor(fc, [1, 2, S, nbDf]) })
		let M = (await net.erb.run({ emb: enc.emb, e3: enc.e3, e2: enc.e2, e1: enc.e1, e0: enc.e0 })).m.data
		let C = (await net.df.run({ emb: enc.emb, c0: enc.c0 }, ['coefs'])).coefs.data
		for (let t = a; t < e; t++) {
			let s = t - a0, o = (t % cap) * BINS
			// ERB gains (Mask with the unnormalized inverse filterbank: each bin takes its band's gain)
			for (let band = 0; band < nbErb; band++) {
				let g = M[s * nbErb + band]
				for (let j = erbStart[band], end = j + widths[band]; j < end; j++) { Er[j] = f(Sr[o + j] * g); Ei[j] = f(Si[o + j] * g) }
			}
			// deep filter on the unmasked spectrum: Σₖ spec[t − lead + k] · coef[t, k], complex, below nbDf
			for (let j = 0; j < nbDf; j++) { Er[j] = 0; Ei[j] = 0 }
			for (let k = 0; k < order; k++) {
				let tt = t - lead + k
				if (tt < 0 || tt >= T) continue
				let q = (tt % cap) * BINS, c = s * nbDf * order * 2
				for (let j = 0; j < nbDf; j++) {
					let cr = C[c + j * order * 2 + 2 * k], ci = C[c + j * order * 2 + 2 * k + 1], sr = Sr[q + j], si = Si[q + j]
					Er[j] = f(Er[j] + f(f(sr * cr) - f(si * ci))); Ei[j] = f(Ei[j] + f(f(sr * ci) + f(si * cr)))
				}
			}
			if (lim) for (let j = 0; j < BINS; j++) { Er[j] = f(f(Sr[o + j] * lim) + f(Er[j] * (1 - lim))); Ei[j] = f(f(Si[o + j] * lim) + f(Ei[j] * (1 - lim))) }
			if (vo && vo.on[t]) keep(t, o)
			if (t >= b) { tailR.set(Er, (t - b) * BINS); tailI.set(Ei, (t - b) * BINS); continue }
			if (a > 0 && t - a < fade) {
				let w = .5 - .5 * Math.cos(Math.PI * (t - a + .5) / fade), q = (t - a) * BINS
				for (let j = 0; j < BINS; j++) { Er[j] = f(tailR[q + j] + w * (Er[j] - tailR[q + j])); Ei[j] = f(tailI[q + j] + w * (Ei[j] - tailI[q + j])) }
			}
			// frame_synthesis: unnormalized inverse, window, overlap-add
			irfft(Er, Ei, y)
			for (let i = 0; i < hop; i++) {
				let v = f(f(y[i] * WINDOW[i]) + syn[i]), p = t * hop + i - d
				if (p >= 0 && p < L) out[p] = v
				syn[i] = f(y[hop + i] * WINDOW[hop + i])
			}
		}
	}
	return out
}
