// The network of scripts/model.py in plain JavaScript: float32 weights, one frame at a time. At this
// size (a few thousand weights, under a million multiply-adds a frame) a GPU would spend more on
// transfers than on the arithmetic, so there is no WebGPU or ONNX path. Loops run without bounds
// checks over zero-padded buffers; the banded Toeplitz layer runs as one FFT convolution.

import { fft, ifft } from 'fourier-transform'
import { WEIGHTS } from './weights.js'

const SLOPE = 0.1

/** half-precision bits → float */
function half(h) {
	let e = (h >> 10) & 31, m = h & 1023, s = h & 32768 ? -1 : 1
	return e === 0 ? s * m * 2 ** -24 : e === 31 ? (m ? NaN : s * Infinity) : s * (1 + m / 1024) * 2 ** (e - 15)
}

function unpack({ config, data }) {
	let bytes = typeof atob === 'function' ? Uint8Array.from(atob(data), c => c.charCodeAt(0)) : Buffer.from(data, 'base64')
	let v = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength), at = 0
	let take = n => { let a = new Float32Array(n); for (let i = 0; i < n; i++, at += 2) a[i] = half(v.getUint16(at, true)); return a }
	let { C, T, blocks, D1, D2, H } = config, S = 2 * T + 3
	let w = {
		stem: take(C * 9), stemB: take(C),
		dw: blocks.map(() => take(C * 5)), dwB: blocks.map(() => take(C)),
		pw: blocks.map(() => take(C * C)), pwB: blocks.map(() => take(C)),
		proj: take(T * C), projB: take(T),
		comb: take(T * (D1 + D2 + 1)), bias: take(1),
		h1: take(H * S), h1B: take(H), h2: take(H), h2B: take(1)
	}
	if (at !== bytes.length) throw new Error('neural-pitch: weights do not match the configuration')
	return w
}

export const CONFIG = WEIGHTS.config
let W

/**
 * Network over one frame: `net(input, level)` → { logits (Float32Array(K), reused), voicing (0…1) }.
 * input: K model-input values (features.js normalize), level: the frame's level in dBFS.
 */
export function network() {
	W ??= unpack(WEIGHTS)
	let { C, T, blocks, D1, D2, H, K } = CONFIG, D = D1 + D2 + 1
	let P = Math.max(4, 2 * Math.max(...blocks)), R = K + 2 * P      // a channel's row: P zeros either side
	let xp = new Float32Array(K + 8), h = new Float32Array(C * R), g = new Float32Array(C * K)
	let m = new Float32Array(T * K), y = new Float32Array(K), s = new Float32Array(2 * T + 3)
	let lr = v => v > 0 ? v : v * SLOPE

	// comb: y_j = bias + Σ_c Σ_t w[c, t] m_c[j + t − D1], a circular convolution of m_c with
	// r_c[(D1 − t) mod L] = w[c, t], free of wrap-around for L ≥ K + D − 1
	let L = 2 ** Math.ceil(Math.log2(K + D - 1)), bins = (L >> 1) + 1
	let spec = Array.from({ length: T }, (_, c) => {
		let r = new Float64Array(L)
		for (let t = 0; t < D; t++) r[(D1 - t + L) % L] = W.comb[c * D + t]
		return fft(r).map(a => Float64Array.from(a))
	})
	let buf = new Float64Array(L), F = [new Float64Array(bins), new Float64Array(bins)]
	let Yr = new Float64Array(bins), Yi = new Float64Array(bins), out = new Float64Array(L)

	return (x, level) => {
		// stem: conv 1 → C, kernel 9, zero padding 4
		xp.set(x, 4)
		for (let c = 0; c < C; c++) {
			let wc = c * 9, b = W.stemB[c], o = c * R + P, st = W.stem
			let w0 = st[wc], w1 = st[wc + 1], w2 = st[wc + 2], w3 = st[wc + 3], w4 = st[wc + 4], w5 = st[wc + 5], w6 = st[wc + 6], w7 = st[wc + 7], w8 = st[wc + 8]
			for (let k = 0; k < K; k++) {
				let a = b + w0 * xp[k] + w1 * xp[k + 1] + w2 * xp[k + 2] + w3 * xp[k + 3] + w4 * xp[k + 4] + w5 * xp[k + 5] + w6 * xp[k + 6] + w7 * xp[k + 7] + w8 * xp[k + 8]
				h[o + k] = a > 0 ? a : a * SLOPE
			}
		}
		// blocks: h = lrelu(h + pointwise(depthwise(h))), depthwise kernel 5 at dilation d
		for (let b = 0; b < blocks.length; b++) {
			let d = blocks[b], dw = W.dw[b], dwB = W.dwB[b], pw = W.pw[b], pwB = W.pwB[b]
			for (let c = 0; c < C; c++) {
				let o = c * R + P, q = c * K, w0 = dw[c * 5], w1 = dw[c * 5 + 1], w2 = dw[c * 5 + 2], w3 = dw[c * 5 + 3], w4 = dw[c * 5 + 4], bc = dwB[c]
				for (let k = 0; k < K; k++) {
					let i = o + k
					g[q + k] = bc + w0 * h[i - 2 * d] + w1 * h[i - d] + w2 * h[i] + w3 * h[i + d] + w4 * h[i + 2 * d]
				}
			}
			for (let c = 0; c < C; c++) {
				let o = c * R + P, row = c * C
				y.fill(pwB[c])
				for (let e = 0; e < C; e++) {
					let w = pw[row + e], q = e * K
					for (let k = 0; k < K; k++) y[k] += w * g[q + k]
				}
				for (let k = 0; k < K; k++) h[o + k] = lr(h[o + k] + y[k])
			}
		}
		// projection C → T, each map straight into the comb's spectrum
		Yr.fill(0); Yi.fill(0)
		for (let c = 0; c < T; c++) {
			let q = c * K
			y.fill(W.projB[c])
			for (let e = 0; e < C; e++) {
				let w = W.proj[c * C + e], o = e * R + P
				for (let k = 0; k < K; k++) y[k] += w * h[o + k]
			}
			buf.fill(0)
			for (let k = 0; k < K; k++) buf[k] = m[q + k] = lr(y[k])
			fft(buf, F)
			let [fr, fi] = F, [sr, si] = spec[c]
			for (let j = 0; j < bins; j++) { Yr[j] += fr[j] * sr[j] - fi[j] * si[j]; Yi[j] += fr[j] * si[j] + fi[j] * sr[j] }
		}
		ifft(Yr, Yi, out)
		for (let j = 0; j < K; j++) y[j] = out[j] + W.bias[0]
		// voicing: per map max and mean, the logits' max and log-sum-exp − log K, level / 100
		for (let c = 0; c < T; c++) {
			let mx = -Infinity, sum = 0, o = c * K
			for (let k = 0; k < K; k++) { let v = m[o + k]; if (v > mx) mx = v; sum += v }
			s[c] = mx; s[T + c] = sum / K
		}
		let ymax = -Infinity, se = 0
		for (let j = 0; j < K; j++) if (y[j] > ymax) ymax = y[j]
		for (let j = 0; j < K; j++) se += Math.exp(y[j] - ymax)
		s[2 * T] = ymax; s[2 * T + 1] = ymax + Math.log(se) - Math.log(K); s[2 * T + 2] = level / 100
		let S = s.length, v = W.h2B[0]
		for (let i = 0; i < H; i++) {
			let a = W.h1B[i]
			for (let e = 0; e < S; e++) a += W.h1[i * S + e] * s[e]
			v += W.h2[i] * lr(a)
		}
		return { logits: y, voicing: 1 / (1 + Math.exp(-v)) }
	}
}
