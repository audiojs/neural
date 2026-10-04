// RNNoise: J.-M. Valin, "A Hybrid DSP/Deep Learning Approach to Real-Time Full-Band Speech
// Enhancement", MMSP 2018 (arXiv:1709.08243). A network of 1.8M weights predicts 32 ERB-band gains
// and a voice probability per 10 ms frame at 48 kHz; DSP does the rest: 32 bands, pitch search, a
// pitch filter restoring voiced harmonics between the bands, gain smoothing.
//
// Copyright (c) 2007-2017, 2024 Jean-Marc Valin; 2023 Amazon; 2017-2019 Mozilla; 2005-2017 Xiph.Org
// Foundation; 2008-2011 Octasic Inc.; 2007-2008 CSIRO; 2018 Gregor Richards. BSD-3-Clause: the
// conditions and the disclaimer are in LICENSE.
//
// Changes, 2026, audiojs: translated to JavaScript from xiph/rnnoise at 70f1d25 (src/denoise.c, rnn.c,
// nnet.c, nnet_arch.h, vec.h, pitch.c, celt_lpc.c, parse_lpcnet_weights.c); the portable C path
// (vec.h without SIMD) op for op in float32, so the output is bit-exact to that build compiled
// without FMA contraction (-ffp-contract=off). One file; state in typed arrays, no allocation per
// frame; weights parsed from upstream's blob (rnnoise_model_from_buffer, parse_weights, linear_init
// with the same checks). The int8 products run in a 419-byte WebAssembly SIMD kernel (gemv.wat) where
// available, else in JS: integer sums, identical either way. Not ported: x86 SU biases and unsigned
// activations (vec_avx.h), training.
//
// Float semantics: each C float op is Math.fround(a op b). For +, −, ×, ÷ and sqrt of float32
// operands the double result rounded to float32 equals the float32 operation (S. A. Figueroa,
// "When is double rounding innocuous?", SIGNUM Newsletter 30(3), 1995: 53 ≥ 2·24 + 2). Where the C
// promotes to double (double literals such as 1e-3, libm log10 and sqrt) the JS stays in double and
// rounds on the store, as the C does.

import { WINDOW, rfft, irfft } from './fft.js'

const f = Math.fround

// C float literals (the f suffix), rounded once as the compiler does
const F = { c06: f(.6), c07: f(.7), c085: f(.85), c09: f(.9), c03: f(.3), c04: f(.4), c05: f(.5), c08: f(.8), c008: f(.008),
	c10001: f(1.0001), c0001: f(.001), c1e12: f(1e-12), hpA0: f(-1.99599), hpA1: f(0.99600),
	N0: f(952.52801514), N1: f(96.39235687), N2: f(0.60863042), D0: f(952.72399902), D1: f(413.36801147), D2: f(11.88600922) }

export const FRAME = 480
// The attenuation limit, dB, that denoise(), the worklet and the audio atom apply unless given one (0:
// none, upstream's output). Unlimited, this model removes the voice on 12 of the 824 VoiceBank+DEMAND
// test files (STOI down by more than 0.2); limited to 20 dB, on none (README, Accuracy).
export const LIMIT = 20
const WIN = 960, FREQ = 481, NB = 32, NF = 65
const PMIN = 60, PMAX = 768, PFRAME = 960, PBUF = PMAX + PFRAME

// denoise.c eband20ms: ERB-spaced band edges in 50 Hz bins (0 Hz to 20 kHz)
const EBAND = [0, 2, 4, 6, 8, 10, 12, 15, 18, 21, 24, 28, 32, 36, 41, 47, 53, 60, 68, 77, 87, 98, 110, 124, 140, 157, 176, 198, 223, 251, 282, 317, 356, 400]

// ------------------------------------------------ tables (dump_rnnoise_tables.c)

const DCT = new Float32Array(NB * NB)
for (let i = 0; i < NB; i++) for (let j = 0; j < NB; j++) {
	DCT[i * NB + j] = Math.cos((i + .5) * j * Math.PI / NB)
	if (j === 0) DCT[i * NB + j] = DCT[i * NB + j] * Math.sqrt(.5)
}
const DCT_NORM = Math.sqrt(2 / 22)

// ------------------------------------------------ weights (parse_lpcnet_weights.c)

const HEAD = 64, TYPES = { 0: Float32Array, 1: Int32Array, 3: Int8Array }

// upstream's weight blob: 64-byte records { 'DNNw', version, type, size, block_size, name[44] }
function parseBlob(bytes) {
	let u8 = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes)
	let dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength), arrays = new Map(), o = 0
	while (o < u8.length) {
		if (u8.length - o < HEAD) throw new Error('rnnoise: truncated weight record')
		let type = dv.getInt32(o + 8, true), size = dv.getInt32(o + 12, true), block = dv.getInt32(o + 16, true)
		let name = '' // ASCII, NUL-terminated; no TextDecoder: AudioWorkletGlobalScope has none
		for (let i = o + 20; i < o + 64 && u8[i]; i++) name += String.fromCharCode(u8[i])
		if (block < size || block > u8.length - o - HEAD || size < 0 || u8[o + 63] !== 0) throw new Error(`rnnoise: bad weight record '${name}'`)
		let T = TYPES[type]
		if (!T) throw new Error(`rnnoise: unknown weight type ${type} for '${name}'`)
		// a copy (ArrayBuffer.slice): the bytes may sit at any offset of a shared buffer (a Node Buffer's pool)
		let data = new T(u8.buffer.slice(u8.byteOffset + o + HEAD, u8.byteOffset + o + HEAD + size))
		arrays.set(name, data)
		o += HEAD + block
	}
	return arrays
}

// linear_init: same presence and size checks; float weights win over int8 when both exist (compute_linear)
function linear(arrays, name, nIn, nOut, { int8 = false, sparse = false, diag = false } = {}) {
	let need = (key, len) => {
		let a = arrays.get(key)
		if (!a || a.length !== len) throw new Error(`rnnoise: weight '${key}' missing or not ${len} long`)
		return a
	}
	let opt = (key, len) => {
		let a = arrays.get(key)
		if (a && a.length !== len) throw new Error(`rnnoise: weight '${key}' is ${a.length} long, want ${len}`)
		return a || null
	}
	let L = { nIn, nOut, bias: need(name + '_bias', nOut), idx: null, weights: null, floats: null, scale: null, diag: null }
	let len = nIn * nOut
	if (sparse) {
		let idx = need(name + '_weights_idx', arrays.get(name + '_weights_idx')?.length ?? -1), blocks = 0, p = 0, out = nOut
		while (p < idx.length) {
			let n = idx[p++]
			if (idx.length - p < n) throw new Error(`rnnoise: '${name}_weights_idx' overruns`)
			for (let i = 0; i < n; i++) { let pos = idx[p++]; if (pos + 3 >= nIn || (pos & 3)) throw new Error(`rnnoise: '${name}_weights_idx' column ${pos} out of range`) }
			blocks += n; out -= 8
		}
		if (out !== 0) throw new Error(`rnnoise: '${name}_weights_idx' covers the wrong number of rows`)
		L.idx = idx; len = 32 * blocks
	}
	if (int8) L.weights = need(name + '_weights_int8', len)
	L.floats = opt(name + '_weights_float', len)
	if (!int8 && !L.floats) throw new Error(`rnnoise: weight '${name}_weights_float' missing`)
	if (int8) L.scale = need(name + '_scale', nOut)
	if (diag) L.diag = need(name + '_weights_diag', nOut)
	opt(name + '_subias', nOut)
	return L
}

// gemv.wat, compiled by wat2wasm (419 bytes): the int8 products of sparse_cgemv8x4 with WebAssembly SIMD
// (i16 widening, i32x4.dot_i16x8_s). The same integer sums as the JS loop, 8.5 times faster; the JS loop
// runs where WebAssembly SIMD does not (Safari before 16.4, Node before 16.4).
const GEMV = new Uint8Array([
	0,97,115,109,1,0,0,0,1,9,1,96,5,127,127,127,127,127,0,3,2,1,0,5,3,1,0,1,7,17,2,6,109,101,109,111,114,121,2,0,
	4,103,101,109,118,0,0,10,241,2,1,238,2,2,2,127,6,123,3,64,32,1,40,2,0,33,6,32,1,65,4,106,33,1,253,12,0,0,0,0,
	0,0,0,0,0,0,0,0,0,0,0,0,33,9,253,12,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,33,10,253,12,0,0,0,0,
	0,0,0,0,0,0,0,0,0,0,0,0,33,11,253,12,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,33,12,2,64,3,64,32,6,
	69,13,1,32,2,32,1,40,2,0,106,40,2,0,253,17,253,135,1,33,7,32,0,253,0,4,0,33,8,32,9,32,8,253,135,1,32,7,253,186,
	1,253,174,1,33,9,32,10,32,8,253,136,1,32,7,253,186,1,253,174,1,33,10,32,0,253,0,4,16,33,8,32,11,32,8,253,135,1,32,7,
	253,186,1,253,174,1,33,11,32,12,32,8,253,136,1,32,7,253,186,1,253,174,1,33,12,32,0,65,32,106,33,0,32,1,65,4,106,33,1,32,
	6,65,1,107,33,6,12,0,11,11,32,3,32,9,32,10,253,13,0,1,2,3,8,9,10,11,16,17,18,19,24,25,26,27,32,9,32,10,253,13,
	4,5,6,7,12,13,14,15,20,21,22,23,28,29,30,31,253,174,1,253,11,4,0,32,3,32,11,32,12,253,13,0,1,2,3,8,9,10,11,16,
	17,18,19,24,25,26,27,32,11,32,12,253,13,4,5,6,7,12,13,14,15,20,21,22,23,28,29,30,31,253,174,1,253,11,4,16,32,3,65,32,
	106,33,3,32,5,65,8,106,33,5,32,5,32,4,73,13,0,11,11,
])

// the int8 layers' weights and block indices in one wasm memory: x (int8) at 0, sums (int32) at 512, then the layers
function attachSimd(net) {
	let mod
	try { mod = new WebAssembly.Module(GEMV) } catch { return }
	let layers = [net.conv2, net.gru1.input, net.gru1.recurrent, net.gru2.input, net.gru2.recurrent, net.gru3.input, net.gru3.recurrent].filter(L => L.weights && !L.floats)
	let idxs = layers.map(L => L.idx ?? denseIdx(L.nIn, L.nOut)), size = 5120
	for (let i = 0; i < layers.length; i++) size += idxs[i].length * 4 + layers[i].weights.length + 16
	let inst = new WebAssembly.Instance(mod), mem = inst.exports.memory
	mem.grow(Math.ceil(size / 65536))
	let k = { gemv: inst.exports.gemv, x: new Int8Array(mem.buffer, 0, 512), sums: new Int32Array(mem.buffer, 512, 1152) }, o = 5120
	layers.forEach((L, i) => {
		new Int32Array(mem.buffer, o, idxs[i].length).set(idxs[i])
		let idx = o; o = (o + idxs[i].length * 4 + 15) & ~15
		new Int8Array(mem.buffer, o, L.weights.length).set(L.weights)
		L.simd = { k, idx, w: o }; o += L.weights.length
	})
}

// the block index of a dense 8×4-blocked matrix: every row block uses every column block
function denseIdx(nIn, nOut) {
	let idx = []
	for (let i = 0; i < nOut; i += 8) { idx.push(nIn >> 2); for (let j = 0; j < nIn; j += 4) idx.push(j) }
	return Int32Array.from(idx)
}

// init_rnnoise (rnnoise_data.c): conv 65·3 → 128, conv 128·3 → 384, three GRUs of 384, dense 1536 → 32 + 1
export function model(bytes, { simd = true } = {}) {
	let a = parseBlob(bytes)
	let gru = k => ({
		input: linear(a, `gru${k}_input`, 384, 1152, { int8: true, sparse: true }),
		recurrent: linear(a, `gru${k}_recurrent`, 384, 1152, { int8: true, sparse: true, diag: true }),
	})
	let net = {
		conv1: linear(a, 'conv1', 195, 128),
		conv2: linear(a, 'conv2', 384, 384, { int8: true }),
		gru1: gru(1), gru2: gru(2), gru3: gru(3),
		dense: linear(a, 'dense_out', 1536, 32),
		vad: linear(a, 'vad_dense', 1536, 1),
	}
	if (simd && typeof WebAssembly === 'object') attachSimd(net)
	return net
}

// ------------------------------------------------ network (nnet.c, nnet_arch.h, vec.h)

const XQ = new Int8Array(1536)

// compute_linear_c: out = W·in + bias (+ diag ⊙ in, GRU recurrent only)
function computeLinear(L, out, x) {
	let N = L.nOut, M = L.nIn
	if (L.floats) {
		if (L.idx) sparseSgemv(out, L.floats, L.idx, N, x)
		else sgemv(out, L.floats, N, M, x)
	} else if (L.simd) {
		let { k, idx, w } = L.simd, s = L.scale
		for (let i = 0; i < M; i++) k.x[i] = Math.floor(.5 + f(127 * x[i]))
		k.gemv(w, idx, 0, 512, N)
		for (let i = 0; i < N; i++) out[i] = f(k.sums[i] * s[i])
	} else {
		// cgemv8x4 / sparse_cgemv8x4: activations quantized to int8, floor(.5 + 127x); the dot
		// products are integers under 2^24, exact in the C's float accumulator as in doubles here
		for (let i = 0; i < M; i++) XQ[i] = Math.floor(.5 + f(127 * x[i]))
		let w = L.weights, s = L.scale, idx = L.idx, p = 0, q = 0
		for (let i = 0; i < N; i += 8) {
			let y0 = 0, y1 = 0, y2 = 0, y3 = 0, y4 = 0, y5 = 0, y6 = 0, y7 = 0
			let n = idx ? idx[q++] : M >> 2
			for (let j = 0; j < n; j++, p += 32) {
				let pos = idx ? idx[q++] : j << 2
				let x0 = XQ[pos], x1 = XQ[pos + 1], x2 = XQ[pos + 2], x3 = XQ[pos + 3]
				y0 += w[p] * x0 + w[p + 1] * x1 + w[p + 2] * x2 + w[p + 3] * x3
				y1 += w[p + 4] * x0 + w[p + 5] * x1 + w[p + 6] * x2 + w[p + 7] * x3
				y2 += w[p + 8] * x0 + w[p + 9] * x1 + w[p + 10] * x2 + w[p + 11] * x3
				y3 += w[p + 12] * x0 + w[p + 13] * x1 + w[p + 14] * x2 + w[p + 15] * x3
				y4 += w[p + 16] * x0 + w[p + 17] * x1 + w[p + 18] * x2 + w[p + 19] * x3
				y5 += w[p + 20] * x0 + w[p + 21] * x1 + w[p + 22] * x2 + w[p + 23] * x3
				y6 += w[p + 24] * x0 + w[p + 25] * x1 + w[p + 26] * x2 + w[p + 27] * x3
				y7 += w[p + 28] * x0 + w[p + 29] * x1 + w[p + 30] * x2 + w[p + 31] * x3
			}
			out[i] = f(y0 * s[i]); out[i + 1] = f(y1 * s[i + 1]); out[i + 2] = f(y2 * s[i + 2]); out[i + 3] = f(y3 * s[i + 3])
			out[i + 4] = f(y4 * s[i + 4]); out[i + 5] = f(y5 * s[i + 5]); out[i + 6] = f(y6 * s[i + 6]); out[i + 7] = f(y7 * s[i + 7])
		}
	}
	let b = L.bias
	for (let i = 0; i < N; i++) out[i] = f(out[i] + b[i])
	if (L.diag) {
		let d = L.diag
		for (let i = 0; i < M; i++) {
			out[i] = f(out[i] + f(d[i] * x[i]))
			out[i + M] = f(out[i + M] + f(d[i + M] * x[i]))
			out[i + 2 * M] = f(out[i + 2 * M] + f(d[i + 2 * M] * x[i]))
		}
	}
}

// sgemv (sgemv16x1 / sgemv8x1 / plain loop): column-major, accumulated in input order
function sgemv(out, w, N, M, x) {
	out.fill(0, 0, N)
	if (N % 8 === 0) {
		let B = N % 16 === 0 ? 16 : 8
		for (let i = 0; i < N; i += B)
			for (let j = 0; j < M; j++) {
				let xj = x[j], o = j * N + i
				for (let k = 0; k < B; k++) out[i + k] = f(out[i + k] + f(w[o + k] * xj))
			}
	}
	else for (let i = 0; i < N; i++) {
		let y = 0
		for (let j = 0; j < M; j++) y = f(y + f(w[j * N + i] * x[j]))
		out[i] = y
	}
}

// sparse_sgemv8x4: only reached with a blob carrying float copies of the sparse int8 matrices
function sparseSgemv(out, w, idx, N, x) {
	out.fill(0, 0, N)
	let p = 0, q = 0
	for (let i = 0; i < N; i += 8) {
		let n = idx[q++]
		for (let j = 0; j < n; j++) {
			let pos = idx[q++]
			for (let c = 0; c < 4; c++, p += 8) {
				let xj = x[pos + c]
				for (let k = 0; k < 8; k++) out[i + k] = f(out[i + k] + f(w[p + k] * xj))
			}
		}
	}
}

// vec.h tanh_approx: (N0 + N1x² + N2x⁴)x / (D0 + D1x² + D2x⁴), clamped to ±1
function tanh(x) {
	let x2 = f(x * x)
	let num = f(f(f(f(F.N2 * x2) + F.N1) * x2) + F.N0)
	let den = f(f(f(f(F.D2 * x2) + F.D1) * x2) + F.D0)
	num = f(f(num * x) / den)
	return num > 1 ? 1 : num < -1 ? -1 : num
}
const sigmoid = x => f(.5 + f(.5 * tanh(f(.5 * x))))

function convLayer(L, out, mem, x, nIn, act, tmp) {
	let keep = L.nIn - nIn
	tmp.set(mem.subarray(0, keep)); tmp.set(x.subarray(0, nIn), keep)
	computeLinear(L, out, tmp)
	for (let i = 0; i < L.nOut; i++) out[i] = act(out[i])
	mem.set(tmp.subarray(nIn, nIn + keep))
}

function gruLayer(G, state, x, zrh, recur) {
	let N = 384
	computeLinear(G.input, zrh, x)
	computeLinear(G.recurrent, recur, state)
	for (let i = 0; i < 2 * N; i++) zrh[i] = sigmoid(f(zrh[i] + recur[i]))
	for (let i = 0; i < N; i++) {
		let h = tanh(f(zrh[2 * N + i] + f(recur[2 * N + i] * zrh[N + i]))), z = zrh[i]
		state[i] = f(f(z * state[i]) + f(f(1 - z) * h))
	}
}

// ------------------------------------------------ DSP (denoise.c, pitch.c, celt_lpc.c)

const SUM = new Float32Array(NB + 2)
function bandEnergy(E, Xr, Xi, Pr, Pi) {
	let sum = SUM.fill(0)
	for (let i = 0; i < NB + 1; i++) {
		let n = EBAND[i + 1] - EBAND[i]
		for (let j = 0; j < n; j++) {
			let fr = f(j / n), k = EBAND[i] + j
			let t = Pr ? f(f(Xr[k] * Pr[k]) + f(Xi[k] * Pi[k])) : f(f(Xr[k] * Xr[k]) + f(Xi[k] * Xi[k]))
			sum[i] = f(sum[i] + f(f(1 - fr) * t))
			sum[i + 1] = f(sum[i + 1] + f(fr * t))
		}
	}
	sum[1] = f(f(f(sum[0] + sum[1]) * 2) / 3)
	sum[NB] = f(f(f(sum[NB] + sum[NB + 1]) * 2) / 3)
	for (let i = 0; i < NB; i++) E[i] = sum[i + 1]
}

// interp_band_gain; bins 400–480 (above 20 kHz) keep what g held: zeros in every caller
function interpGain(g, b) {
	for (let i = 1; i < NB; i++) {
		let n = EBAND[i + 1] - EBAND[i]
		for (let j = 0; j < n; j++) { let fr = f(j / n); g[EBAND[i] + j] = f(f(f(1 - fr) * b[i - 1]) + f(fr * b[i])) }
	}
	for (let j = 0; j < EBAND[1]; j++) g[j] = b[0]
	for (let j = EBAND[NB]; j < EBAND[NB + 1]; j++) g[j] = b[NB - 1]
}

function dct(out, o, x) {
	for (let i = 0; i < NB; i++) {
		let s = 0
		for (let j = 0; j < NB; j++) s = f(s + f(x[j] * DCT[j * NB + i]))
		out[o + i] = s * DCT_NORM
	}
}

function innerProd(x, xo, y, yo, n) {
	let s = 0
	for (let i = 0; i < n; i++) s = f(s + f(x[xo + i] * y[yo + i]))
	return s
}

// xcorr_kernel: four lags at once, the C's rotation of y_0…y_3 kept for the same rounding order
function xcorrKernel(x, xo, y, yo, s, len) {
	let y0 = y[yo++], y1 = y[yo++], y2 = y[yo++], y3 = 0, s0 = s[0], s1 = s[1], s2 = s[2], s3 = s[3], j = 0, t
	for (; j < len - 3; j += 4) {
		t = x[xo++]; y3 = y[yo++]
		s0 = f(s0 + f(t * y0)); s1 = f(s1 + f(t * y1)); s2 = f(s2 + f(t * y2)); s3 = f(s3 + f(t * y3))
		t = x[xo++]; y0 = y[yo++]
		s0 = f(s0 + f(t * y1)); s1 = f(s1 + f(t * y2)); s2 = f(s2 + f(t * y3)); s3 = f(s3 + f(t * y0))
		t = x[xo++]; y1 = y[yo++]
		s0 = f(s0 + f(t * y2)); s1 = f(s1 + f(t * y3)); s2 = f(s2 + f(t * y0)); s3 = f(s3 + f(t * y1))
		t = x[xo++]; y2 = y[yo++]
		s0 = f(s0 + f(t * y3)); s1 = f(s1 + f(t * y0)); s2 = f(s2 + f(t * y1)); s3 = f(s3 + f(t * y2))
	}
	if (j++ < len) { t = x[xo++]; y3 = y[yo++]; s0 = f(s0 + f(t * y0)); s1 = f(s1 + f(t * y1)); s2 = f(s2 + f(t * y2)); s3 = f(s3 + f(t * y3)) }
	if (j++ < len) { t = x[xo++]; y0 = y[yo++]; s0 = f(s0 + f(t * y1)); s1 = f(s1 + f(t * y2)); s2 = f(s2 + f(t * y3)); s3 = f(s3 + f(t * y0)) }
	if (j < len) { t = x[xo++]; y1 = y[yo++]; s0 = f(s0 + f(t * y2)); s1 = f(s1 + f(t * y3)); s2 = f(s2 + f(t * y0)); s3 = f(s3 + f(t * y1)) }
	s[0] = s0; s[1] = s1; s[2] = s2; s[3] = s3
}

const SUM4 = new Float32Array(4)
function pitchXcorr(x, xo, y, yo, xcorr, len, maxPitch) {
	let i = 0
	for (; i < maxPitch - 3; i += 4) {
		SUM4.fill(0)
		xcorrKernel(x, xo, y, yo + i, SUM4, len)
		xcorr[i] = SUM4[0]; xcorr[i + 1] = SUM4[1]; xcorr[i + 2] = SUM4[2]; xcorr[i + 3] = SUM4[3]
	}
	for (; i < maxPitch; i++) xcorr[i] = innerProd(x, xo, y, yo + i, len)
}

// rnn_lpc: Levinson–Durbin, order 4 here, stopping at 30 dB prediction gain
function lpc(out, ac, p) {
	let err = ac[0]
	out.fill(0, 0, p)
	if (ac[0] === 0) return
	for (let i = 0; i < p; i++) {
		let rr = 0
		for (let j = 0; j < i; j++) rr = f(rr + f(out[j] * ac[i - j]))
		rr = f(rr + ac[i + 1])
		let r = f(-rr / err)
		out[i] = r
		for (let j = 0; j < (i + 1) >> 1; j++) {
			let t1 = out[j], t2 = out[i - 1 - j]
			out[j] = f(t1 + f(r * t2)); out[i - 1 - j] = f(t2 + f(r * t1))
		}
		err = f(err - f(f(r * r) * err))
		if (err < f(F.c0001 * ac[0])) break
	}
}

const AC = new Float32Array(5), LPC = new Float32Array(4)

// rnn_pitch_downsample (one channel): 2:1 by [.25 .5 .25], then a 4th-order LPC whitening with a zero
function pitchDownsample(x, xlp, len) {
	let n = len >> 1
	for (let i = 1; i < n; i++) xlp[i] = f(.5 * f(f(.5 * f(x[2 * i - 1] + x[2 * i + 1])) + x[2 * i]))
	xlp[0] = f(.5 * f(f(.5 * x[1]) + x[0]))
	let ac = AC
	pitchXcorr(xlp, 0, xlp, 0, ac, n - 4, 5)
	for (let k = 0; k <= 4; k++) {
		let d = 0
		for (let i = k + n - 4; i < n; i++) d = f(d + f(xlp[i] * xlp[i - k]))
		ac[k] = f(ac[k] + d)
	}
	ac[0] = f(ac[0] * F.c10001)
	for (let i = 1; i <= 4; i++) ac[i] = f(ac[i] - f(f(ac[i] * f(F.c008 * i)) * f(F.c008 * i)))
	let a = LPC, tmp = 1
	lpc(a, ac, 4)
	for (let i = 0; i < 4; i++) { tmp = f(F.c09 * tmp); a[i] = f(a[i] * tmp) }
	let c1 = F.c08
	let n0 = f(a[0] + c1), n1 = f(a[1] + f(c1 * a[0])), n2 = f(a[2] + f(c1 * a[1])), n3 = f(a[3] + f(c1 * a[2])), n4 = f(c1 * a[3])
	let m0 = 0, m1 = 0, m2 = 0, m3 = 0, m4 = 0
	for (let i = 0; i < n; i++) {
		let xi = xlp[i]
		let s = f(xi + f(n0 * m0)); s = f(s + f(n1 * m1)); s = f(s + f(n2 * m2)); s = f(s + f(n3 * m3)); s = f(s + f(n4 * m4))
		m4 = m3; m3 = m2; m2 = m1; m1 = m0; m0 = xi
		xlp[i] = s
	}
}

function findBestPitch(xcorr, y, yo, len, maxPitch, best) {
	let Syy = 1, bn0 = -1, bn1 = -1, bd0 = 0, bd1 = 0
	best[0] = 0; best[1] = 1
	for (let j = 0; j < len; j++) Syy = f(Syy + f(y[yo + j] * y[yo + j]))
	for (let i = 0; i < maxPitch; i++) {
		if (xcorr[i] > 0) {
			let c = f(xcorr[i] * F.c1e12), num = f(c * c)
			if (f(num * bd1) > f(bn1 * Syy)) {
				if (f(num * bd0) > f(bn0 * Syy)) { bn1 = bn0; bd1 = bd0; best[1] = best[0]; bn0 = num; bd0 = Syy; best[0] = i }
				else { bn1 = num; bd1 = Syy; best[1] = i }
			}
		}
		Syy = f(Syy + f(f(y[yo + i + len] * y[yo + i + len]) - f(y[yo + i] * y[yo + i])))
		Syy = 1 > Syy ? 1 : Syy
	}
}

// rnn_pitch_search: coarse at 4:1, refined at 2:1 around the two best, pseudo-interpolated
const X4 = new Float32Array(PFRAME >> 2), Y4 = new Float32Array((PFRAME + PMAX) >> 2), XCORR = new Float32Array(PMAX >> 1), BEST = new Int32Array(2)
function pitchSearch(xlp, xo, y, len, maxPitch) {
	let lag = len + maxPitch, x4 = X4, y4 = Y4, xcorr = XCORR.fill(0), best = BEST
	for (let j = 0; j < len >> 2; j++) x4[j] = xlp[xo + 2 * j]
	for (let j = 0; j < lag >> 2; j++) y4[j] = y[2 * j]
	pitchXcorr(x4, 0, y4, 0, xcorr, len >> 2, maxPitch >> 2)
	findBestPitch(xcorr, y4, 0, len >> 2, maxPitch >> 2, best)
	for (let i = 0; i < maxPitch >> 1; i++) {
		xcorr[i] = 0
		if (Math.abs(i - 2 * best[0]) > 2 && Math.abs(i - 2 * best[1]) > 2) continue
		let s = innerProd(xlp, xo, y, i, len >> 1)
		xcorr[i] = -1 > s ? -1 : s
	}
	findBestPitch(xcorr, y, 0, len >> 1, maxPitch >> 1, best)
	let offset = 0
	if (best[0] > 0 && best[0] < (maxPitch >> 1) - 1) {
		let a = xcorr[best[0] - 1], b = xcorr[best[0]], c = xcorr[best[0] + 1]
		if (f(c - a) > f(F.c07 * f(b - a))) offset = 1
		else if (f(a - c) > f(F.c07 * f(b - c))) offset = -1
	}
	return 2 * best[0] - offset
}

const SECOND_CHECK = [0, 0, 3, 2, 3, 2, 5, 2, 3, 2, 3, 2, 5, 2, 3, 2]
const pitchGain = (xy, xx, yy) => f(xy / Math.sqrt(f(1 + f(xx * yy))))

// rnn_remove_doubling: prefer T/k when its correlation holds up; the gain goes to PG[0]
const YYL = new Float32Array((PMAX >> 1) + 1), PG = new Float32Array(1)
function removeDoubling(x, T0, prevPeriod, prevGain) {
	let maxp = PMAX >> 1, minp = PMIN >> 1, N = PFRAME >> 1, o = maxp
	T0 = (T0 / 2) | 0; prevPeriod = (prevPeriod / 2) | 0
	if (T0 >= maxp) T0 = maxp - 1
	let T = T0, xx = 0, xy = 0
	for (let i = 0; i < N; i++) { xx = f(xx + f(x[o + i] * x[o + i])); xy = f(xy + f(x[o + i] * x[o + i - T0])) }
	let yyl = YYL, yy = xx
	yyl[0] = xx
	for (let i = 1; i <= maxp; i++) {
		yy = f(f(yy + f(x[o - i] * x[o - i])) - f(x[o + N - i] * x[o + N - i]))
		yyl[i] = 0 > yy ? 0 : yy
	}
	yy = yyl[T0]
	let bestXy = xy, bestYy = yy, g0 = pitchGain(xy, xx, yy), g = g0
	for (let k = 2; k <= 15; k++) {
		let T1 = ((2 * T0 + k) / (2 * k)) | 0
		if (T1 < minp) break
		let T1b = k === 2 ? (T1 + T0 > maxp ? T0 : T0 + T1) : ((2 * SECOND_CHECK[k] * T0 + k) / (2 * k)) | 0
		let a = 0, b = 0
		for (let i = 0; i < N; i++) { a = f(a + f(x[o + i] * x[o + i - T1])); b = f(b + f(x[o + i] * x[o + i - T1b])) }
		let xy1 = f(.5 * f(a + b)), yy1 = f(.5 * f(yyl[T1] + yyl[T1b])), g1 = pitchGain(xy1, xx, yy1)
		let cont = Math.abs(T1 - prevPeriod) <= 1 ? prevGain : Math.abs(T1 - prevPeriod) <= 2 && 5 * k * k < T0 ? f(.5 * prevGain) : 0
		let th = f(f(F.c07 * g0) - cont); th = F.c03 > th ? F.c03 : th
		if (T1 < 3 * minp) { th = f(f(F.c085 * g0) - cont); th = F.c04 > th ? F.c04 : th }
		else if (T1 < 2 * minp) { th = f(f(F.c09 * g0) - cont); th = F.c05 > th ? F.c05 : th }
		if (g1 > th) { bestXy = xy1; bestYy = yy1; T = T1; g = g1 }
	}
	bestXy = 0 > bestXy ? 0 : bestXy
	let pg = bestYy <= bestXy ? 1 : f(bestXy / f(bestYy + 1))
	let c0 = innerProd(x, o, x, o - (T - 1), N), c1 = innerProd(x, o, x, o - T, N), c2 = innerProd(x, o, x, o - (T + 1), N)
	let offset = f(c2 - c0) > f(F.c07 * f(c1 - c0)) ? 1 : f(c0 - c2) > f(F.c07 * f(c1 - c2)) ? -1 : 0
	if (pg > g) pg = g
	T0 = 2 * T + offset
	PG[0] = pg
	return T0 < PMIN ? PMIN : T0
}

// pitch() → { push(frame, search = true) → period, gain, buf }: one 10 ms frame at 48 kHz, int16 scale, as
// rnn_compute_frame_features searches it over the last 36 ms (PITCH_BUF_SIZE, held in `buf`): the period in
// samples (60 to 768), `gain` the normalized correlation at it; search false only takes the frame in
export function pitch() {
	let buf = new Float32Array(PBUF), xlp = new Float32Array(PBUF >> 1), track = { buf, period: 0, gain: 0 }
	track.push = (x, search = true) => {
		buf.copyWithin(0, FRAME); buf.set(x, PBUF - FRAME)
		if (!search) return track.period
		pitchDownsample(buf, xlp, PBUF)
		track.period = removeDoubling(xlp, PMAX - pitchSearch(xlp, PMAX >> 1, xlp, PFRAME, PMAX - 3 * PMIN), track.period, track.gain)
		track.gain = PG[0]
		return track.period
	}
	return track
}

// ------------------------------------------------ denoiser state (DenoiseState)

// create(weights) → { process(input480, output480) → VAD probability }: one 10 ms frame at 48 kHz,
// samples at int16 scale (±32768), as rnnoise_process_frame. Output lags input by 960 samples.
export function create(net) {
	if (!net || !net.gru1) net = model(net)
	let analysisMem = new Float32Array(FRAME), synthesisMem = new Float32Array(FRAME)
	let pt = pitch(), pitchBuf = pt.buf, memHp = new Float64Array(2), lastg = new Float32Array(NB)
	let dXr = new Float32Array(FREQ), dXi = new Float32Array(FREQ), dPr = new Float32Array(FREQ), dPi = new Float32Array(FREQ)
	let dEx = new Float32Array(NB), dEp = new Float32Array(NB), dExp = new Float32Array(NB)
	// RNNState
	let conv1 = new Float32Array(130), conv2 = new Float32Array(256)
	let h1 = new Float32Array(384), h2 = new Float32Array(384), h3 = new Float32Array(384)
	// scratch
	let x = new Float32Array(FRAME), buf = new Float32Array(WIN)
	let Xr = new Float32Array(FREQ), Xi = new Float32Array(FREQ), Pr = new Float32Array(FREQ), Pi = new Float32Array(FREQ)
	let Ex = new Float32Array(NB), Ep = new Float32Array(NB), Exp = new Float32Array(NB), Ly = new Float32Array(NB)
	let feat = new Float32Array(NF), g = new Float32Array(NB), gf = new Float32Array(FREQ)
	let vad = new Float32Array(1)
	let t1 = new Float32Array(128), tmp = new Float32Array(1536), cat = new Float32Array(1536)
	let zrh = new Float32Array(1152), recur = new Float32Array(1152)

	// apply_window: the 960-point Vorbis window, symmetric
	function window(v) {
		for (let i = 0; i < WIN; i++) v[i] = f(v[i] * WINDOW[i])
	}

	// rnn_compute_frame_features → true when the frame is silent (E < 0.04)
	function features(x) {
		buf.set(analysisMem); buf.set(x, FRAME); analysisMem.set(x)
		window(buf); rfft(buf, Xr, Xi); bandEnergy(Ex, Xr, Xi)
		let idx = pt.push(x)
		for (let i = 0; i < WIN; i++) buf[i] = pitchBuf[PBUF - WIN - idx + i]
		window(buf); rfft(buf, Pr, Pi)
		bandEnergy(Ep, Pr, Pi); bandEnergy(Exp, Xr, Xi, Pr, Pi)
		for (let i = 0; i < NB; i++) Exp[i] = Exp[i] / Math.sqrt(.001 + f(Ex[i] * Ep[i]))
		dct(feat, NB, Exp)
		feat[2 * NB] = .01 * (idx - 300)
		let logMax = -2, follow = -2, E = 0
		for (let i = 0; i < NB; i++) {
			let ly = f(Math.log10(1e-2 + Ex[i])), a = follow - 1.5, b = f(logMax - 7)
			ly = a > ly ? a : ly
			Ly[i] = ly = f(b > ly ? b : ly)
			logMax = logMax > ly ? logMax : ly
			follow = f(a > ly ? a : ly)
			E = f(E + Ex[i])
		}
		if (E < .04) { feat.fill(0); return true }
		dct(feat, 0, Ly)
		feat[0] = f(feat[0] - 12); feat[1] = f(feat[1] - 4)
		return false
	}

	function network() {
		convLayer(net.conv1, t1, conv1, feat, NF, tanh, tmp)
		convLayer(net.conv2, cat, conv2, t1, 128, tanh, tmp)
		gruLayer(net.gru1, h1, cat, zrh, recur)
		gruLayer(net.gru2, h2, h1, zrh, recur)
		gruLayer(net.gru3, h3, h2, zrh, recur)
		cat.set(h1, 384); cat.set(h2, 768); cat.set(h3, 1152)
		computeLinear(net.dense, g, cat)
		for (let i = 0; i < NB; i++) g[i] = sigmoid(g[i])
		computeLinear(net.vad, vad, cat)
		return sigmoid(vad[0])
	}

	// rnn_pitch_filter on the delayed frame: add back the pitch-periodic part the gains would remove
	let r = new Float32Array(NB), rf = new Float32Array(FREQ), newE = new Float32Array(NB), norm = new Float32Array(NB), nf = new Float32Array(FREQ)
	function pitchFilter() {
		rf.fill(0); nf.fill(0)
		for (let i = 0; i < NB; i++) {
			let e = dExp[i], gi = g[i], v
			if (e > gi) v = 1
			else v = f(f(e * e) * f(1 - f(gi * gi))) / (.001 + f(f(gi * gi) * f(1 - f(e * e))))
			v = f(v)
			v = 0 > v ? 0 : v; v = 1 < v ? 1 : v
			r[i] = f(Math.sqrt(v)) * Math.sqrt(dEx[i] / (1e-8 + dEp[i]))
		}
		interpGain(rf, r)
		for (let i = 0; i < FREQ; i++) { dXr[i] = f(dXr[i] + f(rf[i] * dPr[i])); dXi[i] = f(dXi[i] + f(rf[i] * dPi[i])) }
		bandEnergy(newE, dXr, dXi)
		for (let i = 0; i < NB; i++) norm[i] = Math.sqrt(dEx[i] / (1e-8 + newE[i]))
		interpGain(nf, norm)
		for (let i = 0; i < FREQ; i++) { dXr[i] = f(dXr[i] * nf[i]); dXi[i] = f(dXi[i] * nf[i]) }
	}

	// rnnoise_process_frame
	function process(input, output) {
		// rnn_biquad: DC-blocking high-pass, state updated in double, stored as float
		let m0 = memHp[0], m1 = memHp[1]
		for (let i = 0; i < FRAME; i++) {
			let xi = f(input[i]), yi = f(xi + m0)
			m0 = f(m1 + (-2 * xi - F.hpA0 * yi))
			m1 = f(1 * xi - F.hpA1 * yi)
			x[i] = yi
		}
		memHp[0] = m0; memHp[1] = m1
		let p = 0
		if (!features(x)) {
			p = network()
			pitchFilter()
			for (let i = 0; i < NB; i++) {
				let gi = g[i] > f(F.c06 * lastg[i]) ? g[i] : f(F.c06 * lastg[i])
				g[i] = gi
				let v = gi * (dEx[i] + 1e-3) / (Ex[i] + 1e-3)
				lastg[i] = 1 < v ? 1 : v
			}
			gf.fill(0); interpGain(gf, g)
			for (let i = 0; i < FREQ; i++) { dXr[i] = f(dXr[i] * gf[i]); dXi[i] = f(dXi[i] * gf[i]) }
		}
		irfft(dXr, dXi, buf)
		window(buf)
		for (let i = 0; i < FRAME; i++) output[i] = f(buf[i] + synthesisMem[i])
		synthesisMem.set(buf.subarray(FRAME))
		dXr.set(Xr); dXi.set(Xi); dPr.set(Pr); dPi.set(Pi); dEx.set(Ex); dEp.set(Ep); dExp.set(Exp)
		return p
	}

	return { process, latency: 2 * FRAME }
}
