// Music guard: speech and noise are enhanced, music passes through untouched, segment by segment.
//
// Both models enhance speech, and to them everything else is noise: music loses 8 to 16 dB in every band at their
// default limits. What tells music from speech here is inaSpeechSegmenter's speech/music/noise CNN (D. Doukhan,
// J. Carrive, F. Vallet, A. Larcher, S. Meignier, "An Open-Source Speaker Gender Detection Framework for Monitoring
// Gender Equality", ICASSP 2018; ina-foss/inaSpeechSegmenter, MIT; its 'smn' engine, keras_speech_music_noise_cnn.hdf5),
// trained on French radio and TV, where it labels speech over a music bed as speech and songs as music. Its input as ina
// computes it (sidekit's mfcc() at ina's settings): 16 kHz audio, pre-emphasis 0.97 within each frame, 25 ms Hann
// frames every 10 ms, a 512-point power spectrum, 24 triangular HTK-mel filters from 100 Hz to 8 kHz, natural log. The
// lowest 21 bands over 68 frames (0.68 s) make a patch, standardized by its own mean and deviation, so the level
// doesn't matter. Four convolutions and four dense layers give speech, music and noise; 782,403 weights, in guard.bin
// as float16 (1.56 MB, scripts/guard.py).
//
// A patch every STEP frames (ina takes one every 2; on the training material every 10 decides the same, at a fifth of
// the work). Offline, the classes become segments as ina makes them, by Viterbi over the whole input: three states,
// a switch costing 10^-80 per 20 ms (ina's SpeechMusicNoise, diag_trans_exp(80)), each patch PRIOR nats in favour of
// speech (from 0.5): a voice under a music bed as loud as itself reaches ina as music half the time, speech at a mean
// probability of 0.47, songs at 0.01; PRIOR is the most that passes every frame of the training songs and
// accompaniments passed without it (README, Music). A stream can only use what has arrived: the forward recursion of
// the same chain, for two states (music, the rest), keeps the log-odds of music, bounded at ±BOUND, a patch moving them
// by at most ln(1/FLOOR), each patch PRIOR nats in favour of speech; it switches to music above HYST and back below 0
// (hysteresis). Either way, music passes (the enhanced signal's gain 0), speech and noise are enhanced (gain 1), and
// the gain moves over RAMP samples, raised cosine: offline centered on the segment boundary, streaming from the
// decision on.
//
// The products of the three upper convolutions and the dense layers run in a WebAssembly SIMD kernel (gemm.wat) or
// its JS form, the same float ops in the same order, so the same bits either way.

const f = Math.fround

export const STEP = 10, RAMP = 9600, BOUND = 20, PRIOR = 1, HYST = 5, FLOOR = 1e-2
const COST = 80 * Math.LN10 * 2 / STEP

// ------------------------------------------------ the network

// conv: [out, kh, kw, in], dense: [out, in], each followed by its bias, batch norms folded into the layer before
const LAYERS = [[64, 7, 4, 1], [128, 3, 3, 64], [128, 3, 3, 128], [256, 3, 3, 128], [256, 256], [256, 256], [256, 256], [256, 256], [3, 256]]
export const WEIGHTS = LAYERS.reduce((n, s) => n + s.reduce((a, b) => a * b) + s[0], 0)

// IEEE 754 binary16 → number
const half = h => { let e = (h >> 10) & 31, m = h & 1023, v = e ? (1 + m / 1024) * 2 ** (e - 15) : m * 2 ** -24; return h >> 15 ? -v : v }

// gemm.wat, compiled by wat2wasm (1,074 bytes): y[p, r] = max(0, b[r] + Σc w[r, c] · x[p, c]), four rows and two
// positions at a time, each row in four float32 lanes summed (l0 + l1) + (l2 + l3)
const GEMM = new Uint8Array([
	0,97,115,109,1,0,0,0,1,11,1,96,7,127,127,127,127,127,127,127,0,3,2,1,0,5,3,1,0,1,7,17,2,6,109,101,109,111,114,121,2,0,
	4,103,101,109,109,0,0,10,254,7,1,251,7,2,7,127,12,123,32,5,65,2,116,33,13,3,64,32,1,32,9,32,13,108,106,33,10,32,10,32,
	13,106,32,10,32,9,65,1,106,32,6,73,27,33,11,65,0,33,7,3,64,253,12,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,33,14,253,12,0,0,0,
	0,0,0,0,0,0,0,0,0,0,0,0,0,33,15,253,12,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,33,16,253,12,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
	33,17,253,12,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,33,18,253,12,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,33,19,253,12,0,0,0,0,0,0,0,
	0,0,0,0,0,0,0,0,0,33,20,253,12,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,33,21,32,0,32,7,32,13,108,106,33,12,65,0,33,8,3,64,32,
	10,32,8,106,253,0,4,0,33,22,32,11,32,8,106,253,0,4,0,33,23,32,12,32,8,106,253,0,4,0,33,24,32,14,32,24,32,22,253,230,1,
	253,228,1,33,14,32,18,32,24,32,23,253,230,1,253,228,1,33,18,32,12,32,13,106,32,8,106,253,0,4,0,33,24,32,15,32,24,32,
	22,253,230,1,253,228,1,33,15,32,19,32,24,32,23,253,230,1,253,228,1,33,19,32,12,32,13,65,1,116,106,32,8,106,253,0,4,0,
	33,24,32,16,32,24,32,22,253,230,1,253,228,1,33,16,32,20,32,24,32,23,253,230,1,253,228,1,33,20,32,12,32,13,65,3,108,
	106,32,8,106,253,0,4,0,33,24,32,17,32,24,32,22,253,230,1,253,228,1,33,17,32,21,32,24,32,23,253,230,1,253,228,1,33,21,
	32,8,65,16,106,33,8,32,8,32,13,73,13,0,11,32,2,32,7,65,2,116,106,253,0,4,0,33,25,32,14,32,15,253,13,0,1,2,3,16,17,18,
	19,0,1,2,3,16,17,18,19,32,14,32,15,253,13,4,5,6,7,20,21,22,23,4,5,6,7,20,21,22,23,253,228,1,32,14,32,15,253,13,8,9,10,
	11,24,25,26,27,8,9,10,11,24,25,26,27,32,14,32,15,253,13,12,13,14,15,28,29,30,31,12,13,14,15,28,29,30,31,253,228,1,253,
	228,1,33,24,32,16,32,17,253,13,0,1,2,3,16,17,18,19,0,1,2,3,16,17,18,19,32,16,32,17,253,13,4,5,6,7,20,21,22,23,4,5,6,7,
	20,21,22,23,253,228,1,32,16,32,17,253,13,8,9,10,11,24,25,26,27,8,9,10,11,24,25,26,27,32,16,32,17,253,13,12,13,14,15,
	28,29,30,31,12,13,14,15,28,29,30,31,253,228,1,253,228,1,33,22,32,3,32,9,32,4,108,32,7,106,65,2,116,106,253,12,0,0,0,0,
	0,0,0,0,0,0,0,0,0,0,0,0,32,25,32,24,32,22,253,13,0,1,2,3,4,5,6,7,16,17,18,19,20,21,22,23,253,228,1,253,233,1,253,11,4,
	0,32,9,65,1,106,32,6,73,4,64,32,18,32,19,253,13,0,1,2,3,16,17,18,19,0,1,2,3,16,17,18,19,32,18,32,19,253,13,4,5,6,7,20,
	21,22,23,4,5,6,7,20,21,22,23,253,228,1,32,18,32,19,253,13,8,9,10,11,24,25,26,27,8,9,10,11,24,25,26,27,32,18,32,19,253,
	13,12,13,14,15,28,29,30,31,12,13,14,15,28,29,30,31,253,228,1,253,228,1,33,24,32,20,32,21,253,13,0,1,2,3,16,17,18,19,0,
	1,2,3,16,17,18,19,32,20,32,21,253,13,4,5,6,7,20,21,22,23,4,5,6,7,20,21,22,23,253,228,1,32,20,32,21,253,13,8,9,10,11,
	24,25,26,27,8,9,10,11,24,25,26,27,32,20,32,21,253,13,12,13,14,15,28,29,30,31,12,13,14,15,28,29,30,31,253,228,1,253,
	228,1,33,22,32,3,32,9,65,1,106,32,4,108,32,7,106,65,2,116,106,253,12,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,32,25,32,24,32,
	22,253,13,0,1,2,3,4,5,6,7,16,17,18,19,20,21,22,23,253,228,1,253,233,1,253,11,4,0,11,32,7,65,4,106,33,7,32,7,32,4,73,
	13,0,11,32,9,65,2,106,33,9,32,9,32,6,73,13,0,11,11,
])

/** net(bytes, { simd }) → the classifier over guard.bin; rest(p1) takes conv1's pooled output, gives [speech, music, noise]. */
export function net(bytes, { simd = true } = {}) {
	let u = new Uint16Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength)), o = 0
	if (u.length !== WEIGHTS) throw new Error(`neural-denoise: guard weights are ${u.length} values, want ${WEIGHTS}`)
	let take = n => { let a = new Float32Array(n); for (let i = 0; i < n; i++) a[i] = half(u[o++]); return a }
	let [c1, c2, c3, c4, ...dense] = LAYERS.map(s => ({ s, w: take(s.reduce((a, b) => a * b)), b: take(s[0]) }))
	let out = dense.pop(), mm = matmul([c2, c3, c4, ...dense], simd), col = new Float32Array(12 * 5 * 1152)
	// conv1's kernel sums: a standardized patch (x − μ)/σ convolves to (conv(x) − μ·Σk)/σ
	let ksum = new Float32Array(64)
	for (let a = 0; a < 64; a++) for (let j = 0; j < 28; j++) ksum[a] += c1.w[a * 28 + j]
	// valid convolution, channels last, through im2col: x [H][W][C] → [H − kh + 1][W − kw + 1][co], ReLU
	function conv(x, H, W, C, l) {
		let [co, kh, kw] = l.s, oh = H - kh + 1, ow = W - kw + 1, q = 0
		for (let i = 0; i < oh; i++) for (let j = 0; j < ow; j++) for (let h = 0; h < kh; h++) {
			let a = ((i + h) * W + j) * C
			col.set(x.subarray(a, a + kw * C), q); q += kw * C
		}
		return [mm(l, col, oh * ow), oh, ow, co]
	}
	return {
		c1, ksum,
		rest(p1) {
			for (let i = 0; i < p1.length; i++) if (p1[i] < 0) p1[i] = 0
			let t = conv(p1, 16, 9, 64, c2)
			t = pool(conv(...t, c3), 2, 2, 0)
			let v = pool(conv(...t, c4), 4, 1, 0)[0]
			for (let l of dense) v = mm(l, v, 1)
			let z = [0, 1, 2].map(a => { let s = out.b[a]; for (let k = 0; k < 256; k++) s += out.w[a * 256 + k] * v[k]; return s })
			let m = Math.max(...z), e = z.map(q => Math.exp(q - m)), s = e[0] + e[1] + e[2]
			return e.map(q => q / s)
		},
	}
}

// matmul(layers) → mm(l, x, n): the kernel over layer l's weights for n rows of x; wasm where WebAssembly SIMD runs
function matmul(layers, simd = true) {
	let mod = null
	try { mod = simd && typeof WebAssembly === 'object' ? new WebAssembly.Module(GEMM) : null } catch {}
	if (!mod) return (l, x, n) => gemm(l.w, x, l.b, l.s[0], l.w.length / l.s[0], n)
	let inst = new WebAssembly.Instance(mod), mem = inst.exports.memory, top = 0, at = new Map()
	let alloc = n => { let p = top; top += n * 4; return p }
	for (let l of layers) at.set(l, [alloc(l.w.length), alloc(l.b.length)])
	let X = alloc(12 * 5 * 1152), Y = alloc(14 * 7 * 128)
	mem.grow(Math.ceil(top / 65536))
	let F = new Float32Array(mem.buffer)
	for (let l of layers) { F.set(l.w, at.get(l)[0] / 4); F.set(l.b, at.get(l)[1] / 4) }
	return (l, x, n) => {
		let rows = l.s[0], k = l.w.length / rows, [w, b] = at.get(l)
		F.set(x.subarray(0, n * k), X / 4)
		inst.exports.gemm(w, X, b, Y, rows, k, n)
		return F.slice(Y / 4, Y / 4 + n * rows)
	}
}

// gemm.wat in JS
function gemm(w, x, b, rows, k, n) {
	let y = new Float32Array(n * rows)
	for (let p = 0; p < n; p++) for (let r = 0; r < rows; r++) {
		let a0 = 0, a1 = 0, a2 = 0, a3 = 0, wo = r * k, xo = p * k
		for (let c = 0; c < k; c += 4) {
			a0 = f(a0 + f(w[wo + c] * x[xo + c])); a1 = f(a1 + f(w[wo + c + 1] * x[xo + c + 1]))
			a2 = f(a2 + f(w[wo + c + 2] * x[xo + c + 2])); a3 = f(a3 + f(w[wo + c + 3] * x[xo + c + 3]))
		}
		let s = f(b[r] + f(f(a0 + a1) + f(a2 + a3)))
		y[p * rows + r] = s > 0 ? s : 0
	}
	return y
}

// max pooling, window = stride = ph × pw, `pt` rows of padding on top (keras 'same' pads the first pool's rows 1 + 1)
function pool([x, H, W, C], ph, pw, pt) {
	let oh = Math.ceil(H / ph), ow = Math.ceil(W / pw), y = new Float32Array(oh * ow * C).fill(-Infinity)
	for (let i = 0; i < H; i++) for (let j = 0; j < W; j++) {
		let a = Math.floor((i + pt) / ph), c = Math.floor(j / pw), xo = (i * W + j) * C, yo = (a * ow + c) * C
		for (let k = 0; k < C; k++) if (x[xo + k] > y[yo + k]) y[yo + k] = x[xo + k]
	}
	return [y, oh, ow, C]
}

// ------------------------------------------------ features

// 48 → 16 kHz: a 31-tap low-pass at 8 kHz (sinc under a Kaiser window, β = 6), causal, every third sample kept. Its
// stop band (60 dB down from 10.6 kHz, whose alias lands at 5.4 kHz) covers the 21 bands used, which end at 5.4 kHz.
const TAPS = 31, LP = new Float32Array(TAPS)
{
	let I0 = x => { let s = 1, t = 1; for (let k = 1; k < 30; k++) { t *= (x / 2 / k) ** 2; s += t } return s }, sum = 0
	for (let i = 0; i < TAPS; i++) {
		let t = i - (TAPS - 1) / 2, r = 2 * t / (TAPS - 1)
		LP[i] = (t ? Math.sin(Math.PI * t / 3) / (Math.PI * t) : 1 / 3) * I0(6 * Math.sqrt(1 - r * r)) / I0(6); sum += LP[i]
	}
	for (let i = 0; i < TAPS; i++) LP[i] /= sum
}

// sidekit's trfbank(16000, 512, 100, 8000, 0, 24): triangles evenly spaced in HTK mel, each of area 1 (the lowest 21)
const MELS = 24, BANDS = 21, NFFT = 512, WIN = 400, HOP = 160, PATCH = 68
const mel = hz => 2595 * Math.log10(1 + hz / 700), hz = m => 700 * (10 ** (m / 2595) - 1)
const FB = Array.from({ length: BANDS }, (_, i) => {
	let edge = k => hz(mel(100) + k * (mel(8000) - mel(100)) / (MELS + 1)), [lo, c, hi] = [edge(i), edge(i + 1), edge(i + 2)]
	let h = 2 / (hi - lo), w = [], start = Math.floor(lo * NFFT / 16000) + 1, mid = Math.floor(c * NFFT / 16000)
	for (let k = start; k <= mid; k++) w.push(h / (c - lo) * (k * 16000 / NFFT - lo))
	for (let k = mid + 1; k < Math.min(Math.floor(hi * NFFT / 16000) + 1, NFFT) - 1; k++) w.push(h / (hi - c) * (hi - k * 16000 / NFFT))
	return { start, w: Float64Array.from(w) }
})
// numpy.hanning(400)
const HANN = Float64Array.from({ length: WIN }, (_, n) => .5 - .5 * Math.cos(2 * Math.PI * n / (WIN - 1)))

// radix-2 FFT of 512 points, the frame zero-padded → power of bins 0…256
const REV = new Uint16Array(NFFT), COS = new Float64Array(NFFT / 2), SIN = new Float64Array(NFFT / 2)
for (let i = 0; i < NFFT; i++) { let r = 0; for (let b = 0; b < 9; b++) r |= ((i >> b) & 1) << (8 - b); REV[i] = r }
for (let i = 0; i < NFFT / 2; i++) { COS[i] = Math.cos(2 * Math.PI * i / NFFT); SIN[i] = -Math.sin(2 * Math.PI * i / NFFT) }
function power(x, re, im, P) {
	for (let i = 0; i < NFFT; i++) { re[REV[i]] = i < WIN ? x[i] : 0; im[i] = 0 }
	for (let size = 2; size <= NFFT; size <<= 1) {
		let half = size >> 1, step = NFFT / size
		for (let s = 0; s < NFFT; s += size) for (let j = 0; j < half; j++) {
			let a = s + j, b = a + half, c = COS[j * step], d = SIN[j * step]
			let tr = re[b] * c - im[b] * d, ti = re[b] * d + im[b] * c
			re[b] = re[a] - tr; im[b] = im[a] - ti; re[a] += tr; im[a] += ti
		}
	}
	for (let k = 0; k <= NFFT / 2; k++) P[k] = re[k] * re[k] + im[k] * im[k]
}

/**
 * analyzer(nn) → push(x): 48 kHz samples in, the [speech, music, noise] of each patch completed out, in order (null
 * for a patch with digital silence in it: its log-mel is −∞, what ina calls not finite and scores as neither class).
 * Patch k covers the 16 kHz frames STEP·k … STEP·k + 67, frame j the 16 kHz samples 160j … 160j + 399; at 48 kHz
 * (the low-pass delays 15 samples) it is centered on sample 480·(STEP·k + 34.716) and complete once sample
 * 480·(STEP·k + 67) + 1197 is in, in 48 kHz frame STEP·k + 69. push.end() → for an input shorter than a patch, its
 * one patch, padded with its least log-mel value as ina pads.
 */
export function analyzer(nn) {
	let hist = new Float32Array(TAPS), hn = 0, n48 = 0, s16 = new Float32Array(WIN), n16 = 0
	let frame = new Float64Array(WIN), re = new Float64Array(NFFT), im = new Float64Array(NFFT), P = new Float64Array(NFFT / 2 + 1)
	// the last 68 frames' log-mel, and conv1 over the 7 frames from each: [18][64], no bias
	let M = new Float32Array(PATCH * BANDS), C = new Float32Array(PATCH * 1152), mels = new Float32Array(BANDS)
	let p1 = new Float32Array(16 * 9 * 64), nf = 0, low = Infinity, out = [], { c1, ksum } = nn
	let row = j => (j % PATCH) * BANDS
	function addFrame() {
		// pre-emphasis within the frame, its first sample against itself
		for (let i = 0; i < WIN; i++) frame[i] = (s16[(n16 + i) % WIN] - .97 * s16[(n16 + Math.max(i - 1, 0)) % WIN]) * HANN[i]
		power(frame, re, im, P)
		for (let b = 0; b < BANDS; b++) {
			let { start, w } = FB[b], s = 0
			for (let k = 0; k < w.length; k++) s += P[start + k] * w[k]
			mels[b] = Math.log(s); low = Math.min(low, mels[b])
		}
		addRow(mels)
	}
	function addRow(v) {
		M.set(v, row(nf++))
		if (nf >= 7) {
			let o = ((nf - 7) % PATCH) * 1152
			for (let j = 0; j < 18; j++) for (let a = 0; a < 64; a++) {
				let s = 0
				for (let h = 0; h < 7; h++) {
					let m = row(nf - 7 + h) + j, w = (a * 7 + h) * 4
					s += M[m] * c1.w[w] + M[m + 1] * c1.w[w + 1] + M[m + 2] * c1.w[w + 2] + M[m + 3] * c1.w[w + 3]
				}
				C[o + j * 64 + a] = s
			}
		}
		if (nf >= PATCH && (nf - PATCH) % STEP === 0) out.push(patch(nf - PATCH))
	}
	// the patch from frame t: standardized by the mean and deviation of its 68 × 21 values; conv1 from the shared rows,
	// pooled 4 × 2 (the max commutes with the patch's increasing affine map), then the rest of the network
	function patch(t) {
		let s = 0, ss = 0
		for (let j = t; j < t + PATCH; j++) for (let b = 0, r = row(j); b < BANDS; b++) s += M[r + b]
		let mu = s / (PATCH * BANDS)
		if (!isFinite(mu)) return null
		for (let j = t; j < t + PATCH; j++) for (let b = 0, r = row(j); b < BANDS; b++) ss += (M[r + b] - mu) ** 2
		let sd = Math.sqrt(ss / (PATCH * BANDS))
		if (!(sd > 0)) return null
		for (let m = 0; m < 16; m++) for (let c = 0; c < 9; c++) for (let a = 0; a < 64; a++) {
			let v = -Infinity
			for (let i = Math.max(0, 4 * m - 1); i <= Math.min(61, 4 * m + 2); i++) {
				let q = ((t + i) % PATCH) * 1152 + 2 * c * 64 + a
				v = Math.max(v, C[q], C[q + 64])
			}
			p1[(m * 9 + c) * 64 + a] = (v - mu * ksum[a]) / sd + c1.b[a]
		}
		return nn.rest(p1)
	}
	function push(x) {
		out = []
		for (let i = 0; i < x.length; i++) {
			hist[hn = (hn + 1) % TAPS] = x[i]
			if (n48++ % 3) continue
			let v = 0
			for (let k = 0; k < TAPS; k++) v += LP[k] * hist[(hn - k + TAPS) % TAPS]
			s16[n16++ % WIN] = v
			if (n16 >= WIN && (n16 - WIN) % HOP === 0) addFrame()
		}
		return out
	}
	push.end = () => {
		out = []
		if (nf && nf < PATCH) { mels.fill(low); while (nf < PATCH) addRow(mels) }
		return out
	}
	return push
}

// ------------------------------------------------ decisions

const lp = (p, floor) => Math.log(Math.max(p, floor))

/** offline(x, nn) → per 48 kHz frame of x (480 samples), 1 where it is music: ina's Viterbi over the whole input. */
export function offline(x, nn) {
	let a = analyzer(nn), P = [...a(x), ...a.end()], T = P.length, n = Math.ceil(x.length / 480), pass = new Uint8Array(n)
	if (!T) return pass
	// V: the best path's log-likelihood ending in each state (speech, music, noise); B: the state it came from. Every
	// switch costs COST, speech gains PRIOR a patch; a patch of silence scores all states alike. Ties go to the lower
	// state, as numpy's argmax.
	let V = [0, 0, 0], B = new Uint8Array(T * 3)
	for (let t = 0; t < T; t++) {
		let e = P[t] ? P[t].map((p, s) => lp(p, 1e-10) + (s ? 0 : PRIOR)) : [0, 0, 0], j = V.indexOf(Math.max(...V))
		V = V.map((v, s) => { let b = V[j] - COST > v ? j : s; B[t * 3 + s] = b; return (b === s ? v : V[j] - COST) + e[s] })
	}
	let music = new Uint8Array(T)
	for (let t = T - 1, s = V.indexOf(Math.max(...V)); t >= 0; s = B[t * 3 + s], t--) music[t] = +(s === 1)
	// each frame takes the patch centered nearest it
	for (let g = 0; g < n; g++) pass[g] = music[Math.min(T - 1, Math.max(0, Math.round((g + .5 - 34.716) / STEP)))]
	return pass
}

/**
 * online(nn) → frame(x480): the stream's decision, one 48 kHz frame at a time: true once it holds music, from the
 * patches complete in the frames so far. The output frame two before (RNNoise's delay) takes it.
 */
export function online(nn) {
	let a = analyzer(nn), l = 0, on = false
	return x => {
		for (let p of a(x)) {
			if (!p) continue
			l = Math.min(BOUND, Math.max(-BOUND, l + lp(p[1], FLOOR) - lp(p[0] + p[2], FLOOR) - PRIOR))
			on = l > (on ? 0 : HYST)
		}
		return on
	}
}

/** ramp(enhance) → next(enhance): the enhanced signal's gain for the next sample, toward 1 (enhance) or 0 (pass) over
 *  RAMP, starting at 1 or 0. */
export function ramp(start = true) {
	let phi = +start
	return enhance => (phi = Math.min(1, Math.max(0, phi + (enhance ? 1 : -1) / RAMP)), .5 - .5 * Math.cos(Math.PI * phi))
}

/** blend(g, y, x): the enhanced y at gain g, the input x at 1 − g; at 1 and 0 the one signal itself, bit for bit. */
export const blend = (g, y, x) => g === 1 ? y : g === 0 ? x : g * y + (1 - g) * x

/** gains(pass, n, centered) → Float32Array(n), the enhanced signal's gain per sample from a pass flag per 480: offline,
 *  from the first frame's and the ramps centered on each change; streaming, from 1 (nothing decided yet), the ramps
 *  starting at each change. */
export function gains(pass, n, centered) {
	let g = new Float32Array(n), next = ramp(!(centered && pass[0])), lead = centered ? RAMP / 2 : 0
	for (let i = 0; i < n; i++) g[i] = next(!pass[Math.min(pass.length - 1, Math.floor((i + lead) / 480))])
	return g
}

// guard.bin, the network's weights (scripts/guard.py writes them from upstream's export)
let bundled
export async function guardNet() {
	return bundled ??= (async () => {
		let url = new URL('./guard.bin', import.meta.url)
		if (url.protocol === 'file:') return net(new Uint8Array((await import('node:fs')).readFileSync(url)))
		let res = await fetch(url)
		if (!res.ok) throw new Error(`neural-denoise: can't fetch ${url}: ${res.status}`)
		return net(new Uint8Array(await res.arrayBuffer()))
	})()
}
