// atom manifest: RNNoise per @audio/compile CONTRACT, streaming at any sample rate.
//
// RNNoise (rnnoise.js, bit-exact to upstream's C) takes 10 ms frames, 480 samples at 48 kHz, and returns each
// 960 samples late. Blocks of any size collect into frames; the output queue starts with FRAME − 1 = 479
// samples of silence, the least that never runs dry for any block size (the worklet's 448 serves 128-sample
// quanta only), so the delay is constant: 960 + 479 = 1439 samples at 48 kHz, 30 ms. Other rates go to
// 48 kHz and back through a streaming form of @audio/resample-sinc's Lanczos kernel, the resampler denoise()
// uses offline; its lookahead adds to the delay (1354 samples at 44.1 kHz, 30.7 ms). The output is
// denoise()'s: at 48 kHz sample for sample; at other rates, with resample-sinc 1.2.0, except within 50 ms of
// the end, where the offline resampler sees the end and a stream sees the host's trailing silence.
//
// limit (dB): the input is mixed back in at 10^(−limit/20), so noise drops by at most `limit`; 0 lifts it
// (denoise()'s option, DeepFilterNet's atten_lim_db). The default departs from denoise()'s (none) on purpose:
// unlimited, RNNoise's current model removes the voice on 12 of the 824 VoiceBank+DEMAND test files (STOI
// down by more than 0.2); limited to 20 dB, on none, and PESQ rises from 2.11 to 2.46 (README, Accuracy).
// The limit is read per block and applies from the next finished frame on.
//
// Channels are denoised independently; a channel's state (70 KB) is created on its first block, as the
// worklet does, since hosts declare up to 32 channels.

import { model, create, FRAME, LIMIT } from './rnnoise.js'
import { weights } from './denoise.js'

const RATE = 48000, DELAY = 2 * FRAME, PRIME = FRAME - 1, HALF = 16
const net = model(await weights())

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
 * calls `emit(v)` for every output whose taps have all arrived. Output i sits at input position i·from/to;
 * taps before the start are left out and the weights renormalized, as resample() does at its edges.
 */
function resampler(from, to, emit) {
	let rate = from / to, scale = rate > 1 ? 1 / rate : 1, T = Math.ceil(HALF / scale), K = HALF / (Math.PI * Math.PI)
	let A = new Float64Array(2 * T), B = new Float64Array(2 * T), C = new Float64Array(2 * T), D = new Float64Array(2 * T)
	for (let j = 0, t = 1 - T; t <= T; t++, j++) {
		let a = Math.PI * scale * t
		A[j] = Math.sin(a); B[j] = Math.cos(a); C[j] = Math.sin(a / HALF); D[j] = Math.cos(a / HALF)
	}
	// input history: an output reads 2T samples, the newest of them just arrived
	let mask = (1 << Math.ceil(Math.log2(2 * T + 1))) - 1, ring = new Float32Array(mask + 1), n = 0, i = 0
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

/** One channel at host rate sr: `run(x, y, limit)` takes host samples, gives the denoised ones `latency(sr)` later. */
function channel(sr) {
	let st = create(net), frame = new Float32Array(FRAME), den = new Float32Array(FRAME), z = new Float32Array(FRAME)
	// this frame's input and the two before it: RNNoise's output for frame f is frame f − 2's time
	let dry = new Float32Array(3 * FRAME), fill = 0, frames = 0, lim = 0, q = queue(latency(sr))
	let back = sr === RATE ? x => { for (let k = 0; k < x.length; k++) q.push(x[k]) } : resampler(RATE, sr, q.push)
	let push = v => {
		frame[fill] = v * 32768; dry[(frames % 3) * FRAME + fill] = v
		if (++fill < FRAME) return
		st.process(frame, den)
		// as denoise()'s offline path: the output 960 samples on, the input mixed back at `lim`
		if (frames >= 2) {
			let d = ((frames - 2) % 3) * FRAME
			for (let k = 0; k < FRAME; k++) z[k] = (den[k] / 32768) * (1 - lim) + dry[d + k] * lim
			back(z)
		}
		frames++; fill = 0
	}
	let into = sr === RATE ? x => { for (let i = 0; i < x.length; i++) push(x[i]) } : resampler(sr, RATE, push)
	return (x, y, limit) => {
		lim = limit ? 10 ** (-Math.abs(limit) / 20) : 0
		into(x)
		q.pull(y)
	}
}

export const rnnoise = (ctx) => {
	let chans = [], sr = ctx.sampleRate
	return (inputs, outputs, params) => {
		let inp = inputs[0], out = outputs[0]
		if (!inp || !inp.length) return
		let limit = params.limit[0]
		for (let c = 0; c < inp.length; c++) (chans[c] ??= channel(sr))(inp[c], out[c], limit)
	}
}
rnnoise.channels = 'any'
rnnoise.latency = ctx => latency(ctx.sampleRate)
rnnoise.tail = 0
rnnoise.params = {
	limit: { type: 'number', min: 0, max: 100, default: LIMIT, unit: 'dB' },
}
