// The 960-point FFT and 20 ms Vorbis window both models frame speech with (10 ms hop at 48 kHz).
//
// Copyright (c) 2003-2004 Mark Borgerding; 2005-2009 Xiph.Org Foundation; 2008 CSIRO; 2017-2018 Mozilla;
// 2023 Amazon. BSD-3-Clause: the conditions and the disclaimer are in LICENSE.
//
// Changes, 2026, audiojs: translated to JavaScript from xiph/rnnoise at 70f1d25 (src/kiss_fft.c,
// _kiss_fft_guts.h, dump_rnnoise_tables.c): the float build of kiss_fft fixed at 960 points, radices
// 5·3·4·4·4 as kf_factor orders them, twiddles and bit reversal generated as the tables generator does
// (equal to rnnoise_tables.c), every float op rounded as float32.

const f = Math.fround

export const N = 960, HOP = 480, BINS = 481

// sin(π/2 · sin²(π(i + ½)/960)): Vorbis power-complementary window (rnn_half_window, libDF DFState::new)
export const WINDOW = new Float32Array(N)
for (let i = 0; i < HOP; i++) {
	let s = Math.sin(.5 * Math.PI * (i + .5) / HOP)
	WINDOW[i] = WINDOW[N - 1 - i] = Math.sin(.5 * Math.PI * s * s)
}

const FACTORS = [5, 192, 3, 64, 4, 16, 4, 4, 4, 1]
const SCALE = f(0.0010416667) // kiss_fft_state.scale as rnnoise_tables.c prints it: 1/960 in float32
const TW_R = new Float32Array(N), TW_I = new Float32Array(N)
for (let i = 0; i < N; i++) {
	let phase = (-2 * 3.14159265358979323846264338327 / N) * i
	TW_R[i] = Math.cos(phase); TW_I[i] = Math.sin(phase)
}
const BITREV = new Int32Array(N)
;(function bitrev(fout, at, fstride, k) {
	let p = FACTORS[k], m = FACTORS[k + 1]
	if (m === 1) for (let j = 0; j < p; j++, at += fstride) BITREV[at] = fout + j
	else for (let j = 0; j < p; j++, at += fstride, fout += m) bitrev(fout, at, fstride * p, k + 2)
})(0, 0, 1, 0)

// rnn_fft_c: out = FFT(in)/960, complex values as separate re/im arrays of 960
export function fft(ir, ii, or, oi) {
	for (let i = 0; i < N; i++) { or[BITREV[i]] = f(SCALE * ir[i]); oi[BITREV[i]] = f(SCALE * ii[i]) }
	bfly4d(or, oi, 240)
	bfly4(or, oi, 60, 4, 60, 16)
	bfly4(or, oi, 15, 16, 15, 64)
	bfly3(or, oi, 5, 64, 5, 192)
	bfly5(or, oi, 1, 192)
}

const ZERO = new Float32Array(N), FR = new Float32Array(N), FI = new Float32Array(N), IR = new Float32Array(N), II = new Float32Array(N)

// forward_transform: 481 bins of a real 960-sample frame, scaled by 1/960 (libDF's wnorm is the same 1/960)
export function rfft(x, Xr, Xi) {
	fft(x, ZERO, FR, FI)
	for (let i = 0; i < BINS; i++) { Xr[i] = FR[i]; Xi[i] = FI[i] }
}

// inverse_transform: unnormalized inverse of 481 bins (Hermitian extension; imaginary parts of the
// DC and Nyquist bins ignored, as realfft's ComplexToReal does)
export function irfft(Xr, Xi, x) {
	for (let i = 0; i < BINS; i++) { IR[i] = Xr[i]; II[i] = Xi[i] }
	for (let i = BINS; i < N; i++) { IR[i] = IR[N - i]; II[i] = -II[N - i] }
	fft(IR, II, FR, FI)
	x[0] = f(N * FR[0])
	for (let i = 1; i < N; i++) x[i] = f(N * FR[N - i])
}

function bfly4d(r, im, n) {
	for (let i = 0, o = 0; i < n; i++, o += 4) {
		let s0r = f(r[o] - r[o + 2]), s0i = f(im[o] - im[o + 2])
		r[o] = f(r[o] + r[o + 2]); im[o] = f(im[o] + im[o + 2])
		let s1r = f(r[o + 1] + r[o + 3]), s1i = f(im[o + 1] + im[o + 3])
		r[o + 2] = f(r[o] - s1r); im[o + 2] = f(im[o] - s1i)
		r[o] = f(r[o] + s1r); im[o] = f(im[o] + s1i)
		s1r = f(r[o + 1] - r[o + 3]); s1i = f(im[o + 1] - im[o + 3])
		r[o + 1] = f(s0r + s1i); im[o + 1] = f(s0i - s1r)
		r[o + 3] = f(s0r - s1i); im[o + 3] = f(s0i + s1r)
	}
}

function bfly4(r, im, fstride, m, n, mm) {
	let m2 = 2 * m, m3 = 3 * m
	for (let i = 0; i < n; i++) {
		for (let j = 0, o = i * mm; j < m; j++, o++) {
			let t1 = j * fstride, t2 = 2 * t1, t3 = 3 * t1, a = o + m, b = o + m2, c = o + m3
			let s0r = f(f(r[a] * TW_R[t1]) - f(im[a] * TW_I[t1])), s0i = f(f(r[a] * TW_I[t1]) + f(im[a] * TW_R[t1]))
			let s1r = f(f(r[b] * TW_R[t2]) - f(im[b] * TW_I[t2])), s1i = f(f(r[b] * TW_I[t2]) + f(im[b] * TW_R[t2]))
			let s2r = f(f(r[c] * TW_R[t3]) - f(im[c] * TW_I[t3])), s2i = f(f(r[c] * TW_I[t3]) + f(im[c] * TW_R[t3]))
			let s5r = f(r[o] - s1r), s5i = f(im[o] - s1i)
			r[o] = f(r[o] + s1r); im[o] = f(im[o] + s1i)
			let s3r = f(s0r + s2r), s3i = f(s0i + s2i), s4r = f(s0r - s2r), s4i = f(s0i - s2i)
			r[b] = f(r[o] - s3r); im[b] = f(im[o] - s3i)
			r[o] = f(r[o] + s3r); im[o] = f(im[o] + s3i)
			r[a] = f(s5r + s4i); im[a] = f(s5i - s4r)
			r[c] = f(s5r - s4i); im[c] = f(s5i + s4r)
		}
	}
}

function bfly3(r, im, fstride, m, n, mm) {
	let m2 = 2 * m, epi = TW_I[fstride * m]
	for (let i = 0; i < n; i++) {
		for (let k = 0, o = i * mm; k < m; k++, o++) {
			let t1 = k * fstride, t2 = 2 * t1, a = o + m, b = o + m2
			let s1r = f(f(r[a] * TW_R[t1]) - f(im[a] * TW_I[t1])), s1i = f(f(r[a] * TW_I[t1]) + f(im[a] * TW_R[t1]))
			let s2r = f(f(r[b] * TW_R[t2]) - f(im[b] * TW_I[t2])), s2i = f(f(r[b] * TW_I[t2]) + f(im[b] * TW_R[t2]))
			let s3r = f(s1r + s2r), s3i = f(s1i + s2i), s0r = f(s1r - s2r), s0i = f(s1i - s2i)
			r[a] = f(r[o] - f(s3r * .5)); im[a] = f(im[o] - f(s3i * .5))
			s0r = f(s0r * epi); s0i = f(s0i * epi)
			r[o] = f(r[o] + s3r); im[o] = f(im[o] + s3i)
			r[b] = f(r[a] + s0i); im[b] = f(im[a] - s0r)
			r[a] = f(r[a] - s0i); im[a] = f(im[a] + s0r)
		}
	}
}

function bfly5(r, im, fstride, m) {
	let yar = TW_R[fstride * m], yai = TW_I[fstride * m], ybr = TW_R[2 * fstride * m], ybi = TW_I[2 * fstride * m]
	for (let u = 0; u < m; u++) {
		let o0 = u, o1 = u + m, o2 = u + 2 * m, o3 = u + 3 * m, o4 = u + 4 * m
		let t1 = u * fstride, t2 = 2 * t1, t3 = 3 * t1, t4 = 4 * t1
		let s0r = r[o0], s0i = im[o0]
		let s1r = f(f(r[o1] * TW_R[t1]) - f(im[o1] * TW_I[t1])), s1i = f(f(r[o1] * TW_I[t1]) + f(im[o1] * TW_R[t1]))
		let s2r = f(f(r[o2] * TW_R[t2]) - f(im[o2] * TW_I[t2])), s2i = f(f(r[o2] * TW_I[t2]) + f(im[o2] * TW_R[t2]))
		let s3r = f(f(r[o3] * TW_R[t3]) - f(im[o3] * TW_I[t3])), s3i = f(f(r[o3] * TW_I[t3]) + f(im[o3] * TW_R[t3]))
		let s4r = f(f(r[o4] * TW_R[t4]) - f(im[o4] * TW_I[t4])), s4i = f(f(r[o4] * TW_I[t4]) + f(im[o4] * TW_R[t4]))
		let s7r = f(s1r + s4r), s7i = f(s1i + s4i), s10r = f(s1r - s4r), s10i = f(s1i - s4i)
		let s8r = f(s2r + s3r), s8i = f(s2i + s3i), s9r = f(s2r - s3r), s9i = f(s2i - s3i)
		r[o0] = f(r[o0] + f(s7r + s8r)); im[o0] = f(im[o0] + f(s7i + s8i))
		let s5r = f(s0r + f(f(s7r * yar) + f(s8r * ybr))), s5i = f(s0i + f(f(s7i * yar) + f(s8i * ybr)))
		let s6r = f(f(s10i * yai) + f(s9i * ybi)), s6i = -f(f(s10r * yai) + f(s9r * ybi))
		r[o1] = f(s5r - s6r); im[o1] = f(s5i - s6i)
		r[o4] = f(s5r + s6r); im[o4] = f(s5i + s6i)
		let s11r = f(s0r + f(f(s7r * ybr) + f(s8r * yar))), s11i = f(s0i + f(f(s7i * ybr) + f(s8i * yar)))
		let s12r = f(f(s9i * yai) - f(s10i * ybi)), s12i = f(f(s10r * ybi) - f(s9r * yai))
		r[o2] = f(s11r + s12r); im[o2] = f(s11i + s12i)
		r[o3] = f(s11r - s12r); im[o3] = f(s11i - s12i)
	}
}
