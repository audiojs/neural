// atom manifest: RNNoise per @audio/compile CONTRACT, streaming at any sample rate.
//
// RNNoise (rnnoise.js, bit-exact to upstream's C) takes 10 ms frames, 480 samples at 48 kHz, and returns each
// 960 samples late. Blocks of any size collect into frames; the output queue starts with FRAME − 1 = 479
// samples of silence, the least that never runs dry for any block size (the worklet's 448 serves 128-sample
// quanta only), so the delay is constant: 960 + 479 = 1439 samples at 48 kHz, 30 ms. Other rates go to
// 48 kHz and back through a streaming form of @audio/resample-sinc's Lanczos kernel, the resampler denoise()
// uses offline; its lookahead adds to the delay (1354 samples at 44.1 kHz, 30.7 ms). The output is
// denoise()'s: at 48 kHz sample for sample; at other rates, with resample-sinc 1.3.0, except within 50 ms of
// the end, where the offline resampler sees the end and a stream sees the host's trailing silence.
//
// limit (dB): the input is mixed back in at 10^(−limit/20), so noise drops by at most `limit`; 0 lifts it
// (denoise()'s option, DeepFilterNet's atten_lim_db). The default is denoise()'s, 16 dB (rnnoise.js LIMIT):
// unlimited, RNNoise's current model gates clean speech and removes the voice on some noisy files; 16 is the
// most before the voice itself suffers (README, Accuracy). The limit is read per block and applies from the
// next finished frame on.
//
// music: 'pass' (default) has music pass through untouched, as denoise() does (guard.js): each 48 kHz input frame
// goes to the guard too, and the output frame two before it, the one RNNoise gives back then, takes the guard's
// decision so far; the gain between the denoised and the dry signal ramps over 200 ms, at 48 kHz, through the same
// resampler back, and mixes at the host's rate with the input delayed by the latency, so what passes is the input
// itself. Nothing is added to the delay: a stream decides on what has arrived, music is denoised for its first second
// or so, and each switch comes about a second late (README, Music). 'enhance' denoises everything, as before 0.4.
//
// Channels are denoised independently; a channel's state (70 KB, and the guard's 0.3 MB) is created on its first
// block, as the worklet does, since hosts declare up to 32 channels.

import { phases } from '@audio/resample-sinc'
import { model, create, FRAME, LIMIT } from './rnnoise.js'
import { weights } from './denoise.js'
import { guardNet, online, ramp, blend } from './guard.js'

const RATE = 48000, DELAY = 2 * FRAME, PRIME = FRAME - 1, HALF = 16
const net = model(await weights()), guard = await guardNet()

const sinc = x => x === 0 ? 1 : Math.sin(Math.PI * x) / (Math.PI * x)

/** Lanczos taps each side from rate `from` to `to`, as resample-sinc: 16, widened by the ratio on downsampling. */
function taps(from, to) {
	let rate = from / to, scale = rate > 1 ? 1 / rate : 1
	return Math.ceil(HALF / scale)
}

/** Delay at host rate sr, in samples. Output i of the converter back to sr reads 48 kHz samples up to
 *  floor(i·48000/sr) + T₂; each of those waits for the end of its frame, at most 1439 samples on; that 48 kHz
 *  sample reads host input up to floor(·sr/48000) + T₁. So output i needs input up to i + D, no further. */
const latency = sr => sr === RATE ? DELAY + PRIME : Math.floor((taps(RATE, sr) + DELAY + PRIME) * sr / RATE) + taps(sr, RATE)

/**
 * @audio/resample-sinc's resample() (Lanczos, a = 16) as a stream, in its arithmetic, so its samples: `push(x)`
 * calls `emit(v)` for every output whose taps have all arrived. Output i sits at input position i·from/to; at integer
 * rates it reads resample()'s phase table, at others it weighs by angle addition as resample() does there; taps before
 * the start are left out and the weights renormalized, as resample() does at its edges.
 */
function resampler(from, to, emit) {
	let rate = from / to, scale = rate > 1 ? 1 / rate : 1, T = Math.ceil(HALF / scale), K = HALF / (Math.PI * Math.PI)
	// input history: an output reads 2T samples, the newest of them just arrived
	let mask = (1 << Math.ceil(Math.log2(2 * T + 1))) - 1, ring = new Float32Array(mask + 1), n = 0, i = 0
	let tab = phases(from, to)
	if (tab) {
		let { L, M, W, J0, J1 } = tab, S = 2 * T, p = 0, base = 0
		return x => {
			for (let s = 0; s < x.length; s++) {
				ring[n++ & mask] = x[s]
				while (base + T < n) {
					let lo = base + 1 - T, o = p * S - lo, sum = 0, w = 0
					for (let j = Math.max(0, lo + J0[p]), e = lo + J1[p]; j < e; j++) { let k = W[o + j]; sum += ring[j & mask] * k; w += k }
					emit(Math.fround(w !== 0 ? sum / w : 0))
					if ((p += M) >= L) { let q = Math.floor(p / L); base += q; p -= q * L }
				}
			}
		}
	}
	let A = new Float64Array(2 * T), B = new Float64Array(2 * T), C = new Float64Array(2 * T), D = new Float64Array(2 * T)
	for (let j = 0, t = 1 - T; t <= T; t++, j++) {
		let a = Math.PI * scale * t
		A[j] = Math.sin(a); B[j] = Math.cos(a); C[j] = Math.sin(a / HALF); D[j] = Math.cos(a / HALF)
	}
	return x => {
		for (let s = 0; s < x.length; s++) {
			ring[n++ & mask] = x[s]
			for (;;) {
				let pos = i * rate, base = Math.floor(pos)
				if (base + T >= n) break
				let frac = pos - base
				let f = Math.PI * scale * frac, sf = Math.sin(f), cf = Math.cos(f), sg = Math.sin(f / HALF), cg = Math.cos(f / HALF)
				let sum = 0, w = 0
				for (let j = 0, t = 1 - T; t <= T; t++, j++) {
					if (base + t < 0) continue
					let y = (t - frac) * scale
					if (Math.abs(y) >= HALF) continue
					let k = Math.abs(y) < 1e-3 ? sinc(y) * sinc(y / HALF) : (A[j] * cf - B[j] * sf) * (C[j] * cg - D[j] * sg) * K / (y * y)
					sum += ring[(base + t) & mask] * k; w += k
				}
				emit(Math.fround(w !== 0 ? sum / w : 0))
				i++
			}
		}
	}
}

/** Output queue primed with `len` samples of silence (denoise-wiener's primed FIFO, filled a sample at a time). */
function queue(len) {
	let q = { buf: new Float32Array(Math.max(1 << 12, 2 * len)), len }
	q.push = v => {
		if (q.len === q.buf.length) { let nb = new Float32Array(2 * q.buf.length); nb.set(q.buf); q.buf = nb }
		q.buf[q.len++] = v
	}
	q.pull = out => {
		let n = Math.min(out.length, q.len)
		out.set(q.buf.subarray(0, n)); out.fill(0, n)
		q.buf.copyWithin(0, n, q.len); q.len -= n
	}
	return q
}

/** One channel at host rate sr: `run(x, y, limit, music)` takes host samples, gives the denoised ones `latency(sr)` later. */
function channel(sr) {
	let st = create(net), frame = new Float32Array(FRAME), den = new Float32Array(FRAME), z = new Float32Array(FRAME)
	// this frame's input and the two before it: RNNoise's output for frame f is frame f − 2's time
	let dry = new Float32Array(3 * FRAME), fill = 0, frames = 0, lim = 0, pass = false, D = latency(sr)
	let q = queue(D), gq = queue(D), dq = queue(D), decide = online(guard), next = ramp(), gz = new Float32Array(FRAME)
	let to = q => sr === RATE ? x => { for (let k = 0; k < x.length; k++) q.push(x[k]) } : resampler(RATE, sr, q.push)
	let back = to(q), gback = to(gq), g = new Float32Array(0), d = new Float32Array(0)
	let push = v => {
		frame[fill] = v * 32768; dry[(frames % 3) * FRAME + fill] = v
		if (++fill < FRAME) return
		st.process(frame, den)
		let on = decide(dry.subarray((frames % 3) * FRAME, (frames % 3 + 1) * FRAME))
		// as denoise()'s offline path: the output 960 samples on, the input mixed back at `lim`, the guard's gain beside it
		if (frames >= 2) {
			let o = ((frames - 2) % 3) * FRAME
			for (let k = 0; k < FRAME; k++) { z[k] = (den[k] / 32768) * (1 - lim) + dry[o + k] * lim; gz[k] = next(!(on && pass)) }
			back(z); gback(gz)
		}
		frames++; fill = 0
	}
	let into = sr === RATE ? x => { for (let i = 0; i < x.length; i++) push(x[i]) } : resampler(sr, RATE, push)
	return (x, y, limit, music) => {
		lim = limit ? 10 ** (-Math.abs(limit) / 20) : 0
		pass = music !== 'enhance'
		into(x)
		for (let i = 0; i < x.length; i++) dq.push(x[i])
		if (g.length < y.length) { g = new Float32Array(y.length); d = new Float32Array(y.length) }
		q.pull(y); gq.pull(g.subarray(0, y.length)); dq.pull(d.subarray(0, y.length))
		for (let i = 0; i < y.length; i++) y[i] = blend(g[i], y[i], d[i])
	}
}

export const rnnoise = (ctx) => {
	let chans = [], sr = ctx.sampleRate
	return (inputs, outputs, params) => {
		let inp = inputs[0], out = outputs[0]
		if (!inp || !inp.length) return
		let limit = params.limit[0], music = params.music
		for (let c = 0; c < inp.length; c++) (chans[c] ??= channel(sr))(inp[c], out[c], limit, music)
	}
}
rnnoise.channels = 'any'
rnnoise.latency = ctx => latency(ctx.sampleRate)
rnnoise.tail = 0
rnnoise.params = {
	limit: { type: 'number', min: 0, max: 100, default: LIMIT, unit: 'dB' },
	music: { type: 'enum', values: ['pass', 'enhance'], default: 'pass' },
}
