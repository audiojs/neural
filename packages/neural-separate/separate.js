// Source separation (stems) through @audio/neural-runtime's ONNX adapter. Three model families:
//
// 'openunmix' | 'mask': spectrogram models of the Open-Unmix class (Stöter, Uhlich, Liutkus,
//   Mitsufuji, "Open-Unmix - A Reference Implementation for Music Source Separation", JOSS 2019).
//   STFT (4096/1024, Hann, center=True, torch.stft-compatible) → magnitude → one ONNX run per
//   target → multichannel Wiener EM in 300-frame windows (Duong, Vincent, Gribonval,
//   "Under-determined reverberant audio source separation using a full-rank spatial covariance
//   model", IEEE TASLP 2010; algorithm and defaults from sigsep/norbert and open-unmix-pytorch's
//   filtering.py and model.Separator) → iSTFT.
// 'hybrid': Hybrid Transformer Demucs (Rouard, Massa, Défossez, "Hybrid Transformers for Music
//   Source Separation", ICASSP 2023) with its STFT and iSTFT moved out of the graph, the split
//   of sevagh/demucs.onnx. 7.8 s segments overlap-added as demucs.apply.apply_model does; each
//   runs as waveform + complex-as-channels spectrogram in, frequency- and time-branch estimates
//   out, summed after the iSTFT.
// 'complex': complex spectrogram models (SCNet: Tong, Zhang, Liu, Li, Yu, "SCNet: Sparse Compression
//   Network for Music Source Separation", ICASSP 2024; the Band-Split RoFormer family has the same
//   contract) with their STFT and iSTFT out of the graph: each channel's STFT in, re and im as
//   channels, each source's out. 11 s segments overlap-added under linear fades, as
//   ZFTurbo/Music-Source-Separation-Training's demix() runs them.
// 'multires': masking models over several STFT resolutions (MRX: Petermann, Wichern, Wang, Le Roux, "The Cocktail
//   Fork Problem: Three-Stem Audio Separation for Real-World Soundtracks", ICASSP 2022), their STFTs and iSTFTs out of
//   the graph: each channel's magnitudes in, a real mask per source and resolution out; a source is the sum of the
//   iSTFTs of its masked spectrograms. The input at the loudness the model trained at, as upstream's separate.py sets it.
// 'waveform': any [1, C, N] → [1, S, C, N] graph.
//
// Long inputs run in overlapping chunks, so memory stays bounded by the chunk, not the file.

import { fft, ifft } from 'fourier-transform'
import resampleSinc from '@audio/resample-sinc'
import { load as neuralLoad, tensor } from '@audio/neural-runtime'

const PI2 = Math.PI * 2
const isNode = typeof process !== 'undefined' && !!process.versions?.node

// ---------------------------------------------------------------- STFT/iSTFT

const hannCache = new Map()
function hann(n) {
	let w = hannCache.get(n)
	if (!w) {
		w = new Float64Array(n)
		for (let i = 0; i < n; i++) w[i] = 0.5 - 0.5 * Math.cos(PI2 * i / n)
		hannCache.set(n, w)
	}
	return w
}

// numpy/torch 'reflect' pad index: mirrors without repeating the edge sample, period
// 2(n-1); any pad width, even wider than the signal.
function reflectIndex(i, n) {
	if (n === 1) return 0
	let period = 2 * (n - 1)
	i = ((i % period) + period) % period
	return i < n ? i : period - i
}

// Frames as rows of `bins` values: views into one flat frame-major buffer.
function rows(flat, bins) {
	let T = flat.length / bins, r = new Array(T)
	for (let t = 0; t < T; t++) r[t] = flat.subarray(t * bins, (t + 1) * bins)
	return r
}

// Flat complex STFT: re, im = Float64Array(T·bins), frame t at [t·bins, (t+1)·bins).
function stftFlat(x, n, hop, win, center) {
	let N = x.length, bins = (n >> 1) + 1, pad = center ? n >> 1 : 0
	let T = center ? 1 + Math.floor(N / hop) : (N < n ? 0 : 1 + Math.floor((N - n) / hop))
	let re = new Float64Array(T * bins), im = new Float64Array(T * bins), frame = new Float64Array(n)
	for (let m = 0; m < T; m++) {
		let pos = m * hop - pad
		if (pos < 0 || pos + n > N) for (let i = 0; i < n; i++) frame[i] = (center ? x[reflectIndex(pos + i, N)] : x[pos + i] ?? 0) * win[i]
		else for (let i = 0; i < n; i++) frame[i] = x[pos + i] * win[i]
		let [r, j] = fft(frame) // internal buffers, copied out before the next call
		re.set(r, m * bins); im.set(j, m * bins)
	}
	return { re, im, T, bins }
}

// stft(x, opts): mono complex STFT, torch.stft(center=True, pad_mode='reflect') compatible:
// 1 + floor(N/hop) frames, whatever n (the n/2 reflect pad on each side cancels out).
export function stft(x, opts = {}) {
	let n = opts.n ?? 4096, hop = opts.hop ?? 1024, center = opts.center ?? true
	let s = stftFlat(x, n, hop, opts.window ?? hann(n), center)
	return { re: rows(s.re, s.bins), im: rows(s.im, s.bins), n, hop, bins: s.bins, center }
}

// Weighted overlap-add of frame rows into out from frame t0: the synthesis half of istft,
// callable in consecutive batches; olaFinish divides by the window envelope once.
function ola(re, im, out, t0, n, hop, win) {
	for (let m = 0; m < re.length; m++) {
		let f = ifft(re[m], im[m]) // internal buffer, consumed before the next call
		for (let i = 0, pos = (t0 + m) * hop; i < n; i++) out[pos + i] += f[i] * win[i]
	}
}

// Squared-window envelope of T frames: the WOLA denominator (torch.istft's window envelope).
function envelope(T, n, hop, win) {
	let env = new Float64Array(T > 0 ? (T - 1) * hop + n : 0)
	for (let m = 0; m < T; m++) for (let i = 0; i < n; i++) env[m * hop + i] += win[i] * win[i]
	return env
}

// Divide by the envelope, drop the center padding, fit to `length`.
function olaFinish(out, env, n, center, length) {
	let pad = center ? n >> 1 : 0
	length ??= Math.max(0, out.length - 2 * pad)
	let result = new Float32Array(length)
	for (let i = 0, j = pad; i < length && j < out.length; i++, j++) if (env[j] > 1e-8) result[i] = out[j] / env[j]
	return result
}

// istft(frames, opts): exact inverse of stft by weighted overlap-add with squared-window
// normalization (WOLA), as torch.istft does: perfect reconstruction when the squared window
// satisfies NOLA (Hann at 75% overlap does). `length` crops or pads, as torch.istft's.
export function istft(fr, opts = {}) {
	let { re, im, n, hop, center = true } = fr
	let win = opts.window ?? hann(n), T = re.length
	let out = new Float64Array(T > 0 ? (T - 1) * hop + n : 0)
	ola(re, im, out, 0, n, hop, win)
	return olaFinish(out, envelope(T, n, hop, win), n, center, opts.length)
}

// ------------------------------------------------------------- Wiener filter

// wienerFilter(mixStft, estimates, opts): norbert's algorithm (Liutkus & Stöter,
// github.com/sigsep/norbert) with open-unmix-pytorch's defaults (filtering.py: eps=1e-10,
// softmask=False, scale_factor=10). One EM window: the spatial covariances average over all
// given frames (separate() windows by 300 frames, as open-unmix's Separator does).
//
//   mixStft: per-channel stft() output.
//   estimates: { [target]: magnitude[channel][frame] } (Float64Array(bins) per frame); a
//     single-channel estimate broadcasts to all mix channels (norbert's convention).
//   opts.iterations (1): EM steps; 0 returns the initial estimate ("raw masks").
//   opts.softmask (false): true = ratio mask, summing to the mixture by construction; false =
//     target magnitude with the mixture's phase (open-unmix's recommendation).
//   opts.residual (false): appends 'residual' = mixture minus the other targets, before EM.
//   opts.eps (1e-10): regularization floor and division guard.
//
// Returns { [target]: per-channel { re, im, n, hop, bins, center } }, ready for istft().
export function wienerFilter(mixStft, estimates, opts = {}) {
	let C = mixStft.length, T = mixStft[0].re.length, bins = mixStft[0].bins
	let names = Object.keys(estimates)
	if (!names.length) throw new Error('wienerFilter: estimates has no targets')
	let flat = rs => { let a = new Float64Array(T * bins); rs.forEach((r, t) => a.set(r, t * bins)); return a }
	let V = names.map(name => {
		let e = estimates[name]
		if (e.length !== C && e.length !== 1) throw new Error(`wienerFilter: estimate '${name}' has ${e.length} channel(s), mixture has ${C}`)
		return Array.from({ length: C }, (_, c) => flat(e[e.length === 1 ? 0 : c]))
	})
	let { Yr, Yi } = wiener(mixStft.map(s => flat(s.re)), mixStft.map(s => flat(s.im)), V, T, bins, opts)
	let { n, hop, center } = mixStft[0], result = {}
	Yr.forEach((y, j) => { result[names[j] ?? 'residual'] = y.map((re, c) => ({ re: rows(re, bins), im: rows(Yi[j][c], bins), n, hop, bins, center })) })
	return result
}

// The filter on flat frame-major spectra (T·bins per channel): X mixture, V[j][c] target
// magnitudes. Returns Y[j][c], the residual (when asked) last. EM runs on the mixture and
// estimates divided by max(1, max|x| / scaleFactor), filtering.py's guard for numerical
// stability: the initial estimate is built scaled, the mixture scales on the fly, and the
// last EM pass writes the stems back at full scale.
function wiener(Xr, Xi, V, T, bins, { iterations = 1, softmask = false, residual = false, eps = 1e-10, scaleFactor = 10 } = {}) {
	let C = Xr.length, S0 = V.length, S = S0 + (residual ? 1 : 0), TB = T * bins
	let k = 1
	if (iterations > 0) {
		let maxPow = 0
		for (let c = 0; c < C; c++) { let xr = Xr[c], xi = Xi[c]; for (let i = 0; i < TB; i++) { let p = xr[i] * xr[i] + xi[i] * xi[i]; if (p > maxPow) maxPow = p } }
		k = 1 / Math.max(1, Math.sqrt(maxPow) / scaleFactor)
	}
	let Yr = Array.from({ length: S }, () => Array.from({ length: C }, () => new Float64Array(TB)))
	let Yi = Array.from({ length: S }, () => Array.from({ length: C }, () => new Float64Array(TB)))
	let a = new Float64Array(TB), b = new Float64Array(TB)
	for (let c = 0; c < C; c++) {
		let xr = Xr[c], xi = Xi[c]
		if (softmask) {
			// ratio mask: a = eps + Σ_j v_j
			a.fill(eps)
			for (let j = 0; j < S0; j++) { let v = V[j][c]; for (let i = 0; i < TB; i++) a[i] += v[i] }
			for (let j = 0; j < S0; j++) {
				let v = V[j][c], yr = Yr[j][c], yi = Yi[j][c]
				for (let i = 0; i < TB; i++) { let r = v[i] / a[i] * k; yr[i] = r * xr[i]; yi[i] = r * xi[i] }
			}
		} else {
			// the mixture's phase as (cos, sin) in a, b; openunmix's atan2 maps 0/0 to angle 0
			for (let i = 0; i < TB; i++) {
				let m = Math.sqrt(xr[i] * xr[i] + xi[i] * xi[i])
				a[i] = (m > 0 ? xr[i] / m : 1) * k; b[i] = m > 0 ? xi[i] / m * k : 0
			}
			for (let j = 0; j < S0; j++) {
				let v = V[j][c], yr = Yr[j][c], yi = Yi[j][c]
				for (let i = 0; i < TB; i++) { yr[i] = v[i] * a[i]; yi[i] = v[i] * b[i] }
			}
		}
		if (residual) {
			let rr = Yr[S0][c], ri = Yi[S0][c]
			for (let i = 0; i < TB; i++) { rr[i] = xr[i] * k; ri[i] = xi[i] * k }
			for (let j = 0; j < S0; j++) { let yr = Yr[j][c], yi = Yi[j][c]; for (let i = 0; i < TB; i++) { rr[i] -= yr[i]; ri[i] -= yi[i] } }
		}
	}
	if (iterations > 0) em(Yr, Yi, Xr, Xi, T, bins, iterations, eps, k)
	return { Yr, Yi }
}

// expectation_maximization, ported from norbert / openunmix's filtering.py: re-estimate each
// source's power spectral density v_j and spatial covariance R_j, rebuild the mixture
// covariance Cxx = Σ v_j R_j + √eps·I, apply the multichannel Wiener gain v_j R_j Cxx⁻¹.
// Applied as y_j = v_j R_j (Cxx⁻¹ x): the inverse times the mixture once per bin, then one
// C×C product per source, the same algebra as forming each gain first. R_j and Cxx are
// Hermitian (sums of y yᴴ), so stereo takes the closed-form 2×2 inverse; other channel counts
// a Gauss-Jordan inverse. Y arrives scaled by k; X is scaled by k where read.
function em(Yr, Yi, Xr, Xi, T, bins, iterations, eps, k) {
	let S = Yr.length, C = Xr.length, TB = T * bins
	let pairs = [] // upper triangle (c1 ≤ c2) of each C×C covariance
	for (let c1 = 0; c1 < C; c1++) for (let c2 = c1; c2 < C; c2++) pairs.push([c1, c2])
	let V = Array.from({ length: S }, () => new Float64Array(TB))
	let Rr = Array.from({ length: S }, () => pairs.map(() => new Float64Array(bins)))
	let Ri = Array.from({ length: S }, () => pairs.map(() => new Float64Array(bins)))
	let w = new Float64Array(bins), sqrtEps = Math.sqrt(eps)

	for (let iter = 0; iter < iterations; iter++) {
		for (let j = 0; j < S; j++) {
			// 1. v_j = mean over channels of |y_j|²
			let v = V[j]
			v.fill(0)
			for (let c = 0; c < C; c++) { let yr = Yr[j][c], yi = Yi[j][c]; for (let i = 0; i < TB; i++) v[i] += yr[i] * yr[i] + yi[i] * yi[i] }
			for (let i = 0; i < TB; i++) v[i] /= C
			// 2. R_j = Σ_t y yᴴ / (eps + Σ_t v_j), per bin
			w.fill(eps)
			for (let o = 0; o < TB; o += bins) for (let f = 0; f < bins; f++) w[f] += v[o + f]
			pairs.forEach(([c1, c2], p) => {
				let ar = Yr[j][c1], ai = Yi[j][c1], br = Yr[j][c2], bi = Yi[j][c2], rr = Rr[j][p], ri = Ri[j][p]
				rr.fill(0); ri.fill(0)
				for (let o = 0; o < TB; o += bins) for (let f = 0, i = o; f < bins; f++, i++) {
					rr[f] += ar[i] * br[i] + ai[i] * bi[i]
					ri[f] += ai[i] * br[i] - ar[i] * bi[i]
				}
				for (let f = 0; f < bins; f++) { rr[f] /= w[f]; ri[f] /= w[f] }
			})
		}
		// 3. per (frame, bin): w = Cxx⁻¹ x, y_j = v_j R_j w; the last pass writes at full scale
		let out = iter === iterations - 1 ? 1 / k : 1
		if (C === 2) apply2(Yr, Yi, Xr, Xi, V, Rr, Ri, TB, bins, sqrtEps, k, out)
		else applyN(Yr, Yi, Xr, Xi, V, Rr, Ri, pairs, TB, bins, sqrtEps, k, out)
	}
}

// Stereo: Cxx = [[a, b], [b̄, d]] with a, d real, inverse [[d, −b], [−b̄, a]] / (ad − |b|²).
// Sources interleave per bin (R as P[(f·S + j)·4 + k], v as VP[i·S + j], y as YP[(i·S + j)·4 + k]):
// one typed array per quantity instead of one per source, twice as fast.
function apply2(Yr, Yi, Xr, Xi, V, Rr, Ri, TB, bins, sqrtEps, xs, ys) {
	let S = V.length, P = new Float64Array(bins * S * 4), VP = new Float64Array(TB * S), YP = new Float64Array(TB * S * 4)
	for (let j = 0; j < S; j++) {
		let r00 = Rr[j][0], r01r = Rr[j][1], r01i = Ri[j][1], r11 = Rr[j][2], v = V[j]
		for (let f = 0, q = j * 4; f < bins; f++, q += S * 4) { P[q] = r00[f]; P[q + 1] = r01r[f]; P[q + 2] = r01i[f]; P[q + 3] = r11[f] }
		for (let i = 0, q = j; i < TB; i++, q += S) VP[q] = v[i]
	}
	let [x0r, x1r] = Xr, [x0i, x1i] = Xi
	for (let o = 0; o < TB; o += bins) for (let f = 0, i = o; f < bins; f++, i++) {
		let a = sqrtEps, d = sqrtEps, br = 0, bi = 0, pb = f * S * 4, vb = i * S
		for (let j = 0, q = pb; j < S; j++, q += 4) { let v = VP[vb + j]; a += v * P[q]; br += v * P[q + 1]; bi += v * P[q + 2]; d += v * P[q + 3] }
		let det = a * d - br * br - bi * bi, k = det !== 0 ? 1 / det : 0
		let xr0 = x0r[i] * xs, xi0 = x0i[i] * xs, xr1 = x1r[i] * xs, xi1 = x1i[i] * xs
		let w0r = (d * xr0 - br * xr1 + bi * xi1) * k, w0i = (d * xi0 - br * xi1 - bi * xr1) * k
		let w1r = (a * xr1 - br * xr0 - bi * xi0) * k, w1i = (a * xi1 - br * xi0 + bi * xr0) * k
		for (let j = 0, q = pb, y = vb * 4; j < S; j++, q += 4, y += 4) {
			let v = VP[vb + j] * ys, p = P[q], qr = P[q + 1], qi = P[q + 2], s = P[q + 3]
			YP[y] = v * (p * w0r + qr * w1r - qi * w1i)
			YP[y + 1] = v * (p * w0i + qr * w1i + qi * w1r)
			YP[y + 2] = v * (qr * w0r + qi * w0i + s * w1r)
			YP[y + 3] = v * (qr * w0i - qi * w0r + s * w1i)
		}
	}
	for (let j = 0; j < S; j++) {
		let y0r = Yr[j][0], y0i = Yi[j][0], y1r = Yr[j][1], y1i = Yi[j][1]
		for (let i = 0, y = j * 4; i < TB; i++, y += S * 4) { y0r[i] = YP[y]; y0i[i] = YP[y + 1]; y1r[i] = YP[y + 2]; y1i[i] = YP[y + 3] }
	}
}

// Any channel count: Cxx assembled from the Hermitian pairs, Gauss-Jordan inverse.
function applyN(Yr, Yi, Xr, Xi, V, Rr, Ri, pairs, TB, bins, sqrtEps, xs, ys) {
	let S = V.length, C = Xr.length, CC = C * C
	let cr = new Float64Array(CC), ci = new Float64Array(CC), ir = new Float64Array(CC), ii = new Float64Array(CC)
	let sr = new Float64Array(2 * CC), si = new Float64Array(2 * CC), wr = new Float64Array(C), wi = new Float64Array(C)
	let rr = new Float64Array(CC), ri = new Float64Array(CC)
	let full = (j, f) => pairs.forEach(([c1, c2], p) => {
		rr[c1 * C + c2] = Rr[j][p][f]; ri[c1 * C + c2] = Ri[j][p][f]
		rr[c2 * C + c1] = Rr[j][p][f]; ri[c2 * C + c1] = -Ri[j][p][f]
	})
	for (let o = 0; o < TB; o += bins) for (let f = 0, i = o; f < bins; f++, i++) {
		cr.fill(0); ci.fill(0)
		for (let j = 0; j < S; j++) { let v = V[j][i]; full(j, f); for (let k = 0; k < CC; k++) { cr[k] += v * rr[k]; ci[k] += v * ri[k] } }
		for (let c = 0; c < C; c++) cr[c * C + c] += sqrtEps
		complexInverse(cr, ci, ir, ii, C, sr, si)
		for (let c1 = 0; c1 < C; c1++) {
			let ar = 0, ai = 0
			for (let c2 = 0; c2 < C; c2++) { let gr = ir[c1 * C + c2], gi = ii[c1 * C + c2], xr = Xr[c2][i] * xs, xi = Xi[c2][i] * xs; ar += gr * xr - gi * xi; ai += gr * xi + gi * xr }
			wr[c1] = ar; wi[c1] = ai
		}
		for (let j = 0; j < S; j++) {
			let v = V[j][i] * ys
			full(j, f)
			for (let c1 = 0; c1 < C; c1++) {
				let ar = 0, ai = 0
				for (let c2 = 0; c2 < C; c2++) { let gr = rr[c1 * C + c2], gi = ri[c1 * C + c2]; ar += gr * wr[c2] - gi * wi[c2]; ai += gr * wi[c2] + gi * wr[c2] }
				Yr[j][c1][i] = v * ar; Yi[j][c1][i] = v * ai
			}
		}
	}
}

// Complex Gauss-Jordan inverse with partial pivoting, n×n row-major; scratch re/im are n×2n.
function complexInverse(Are, Aim, outRe, outIm, n, re, im) {
	let m = n * 2
	for (let i = 0; i < n; i++) {
		for (let j = 0; j < n; j++) { re[i * m + j] = Are[i * n + j]; im[i * m + j] = Aim[i * n + j] }
		for (let j = n; j < m; j++) { re[i * m + j] = 0; im[i * m + j] = 0 }
		re[i * m + n + i] = 1
	}
	for (let col = 0; col < n; col++) {
		let piv = col, best = re[col * m + col] ** 2 + im[col * m + col] ** 2
		for (let r = col + 1; r < n; r++) {
			let mag = re[r * m + col] ** 2 + im[r * m + col] ** 2
			if (mag > best) { best = mag; piv = r }
		}
		if (piv !== col) for (let k = 0; k < m; k++) {
			let tr = re[col * m + k]; re[col * m + k] = re[piv * m + k]; re[piv * m + k] = tr
			let ti = im[col * m + k]; im[col * m + k] = im[piv * m + k]; im[piv * m + k] = ti
		}
		let pr = re[col * m + col], pi = im[col * m + col]
		let d = pr * pr + pi * pi || 1e-300
		for (let k = 0; k < m; k++) {
			let vr = re[col * m + k], vi = im[col * m + k]
			re[col * m + k] = (vr * pr + vi * pi) / d
			im[col * m + k] = (vi * pr - vr * pi) / d
		}
		for (let r = 0; r < n; r++) {
			if (r === col) continue
			let fr = re[r * m + col], fi = im[r * m + col]
			if (fr === 0 && fi === 0) continue
			for (let k = 0; k < m; k++) {
				let vr = re[col * m + k], vi = im[col * m + k]
				re[r * m + k] -= fr * vr - fi * vi
				im[r * m + k] -= fr * vi + fi * vr
			}
		}
	}
	for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) { outRe[i * n + j] = re[i * m + n + j]; outIm[i * n + j] = im[i * m + n + j] }
}

// ------------------------------------------------------------------ models

// Demucs source order (demucs/pretrained.py SOURCES), the S axis of its outputs.
const DEMUCS = ['drums', 'bass', 'other', 'vocals']

// SCNet's STFT (torch.stft's window=None: ones; normalized) and segment, 476 frames of 11 s
const SCNET = { modelType: 'complex', sampleRate: 44100, targets: DEMUCS, n: 4096, hop: 1024, window: 'ones', normalized: true, segment: 485100, frames: 476 }

// MRX's STFTs (mrx.py _stft: Hann, hop 256, centered, normalized) at three resolutions; its sources in its order
// (music, speech, sfx), named as a soundtrack's stems are; channels encoded apart, so mono stays mono; the input at
// -27 LUFS (separate.py's input_lufs, the stems scaled back); 20 s chunks (Node's peak 1.9 GB on 60 s, whole 4.4 GB)
const MRX = {
	modelType: 'multires', sampleRate: 44100, targets: ['dialogue', 'music', 'effects'], sources: ['music', 'dialogue', 'effects'],
	windows: [1024, 2048, 8192], hop: 256, loudness: -27, mono: true, chunk: 20, script: 'export-mrx.py',
}

// TIGER's DnR model as scripts/export-tiger.py exports it (three band-split models, each its kept source, one graph): its
// STFT (Hann, unnormalized, centered), 12 s segments of 1034 frames every 4 s, as TIGERDNR.wav_chunk_inference cuts
// them; mono, channels apart
const TIGER = {
	modelType: 'complex', sampleRate: 44100, targets: ['dialogue', 'music', 'effects'], sources: ['dialogue', 'music', 'effects'],
	n: 2048, hop: 512, window: 'hann', normalized: false, segment: 529200, frames: 1034, step: 1 / 3, fade: 0, pad: 'zero',
	standardize: false, mono: true, script: 'export-tiger.py',
}

// Presets: files are <weights>/<name>/<target>.onnx (one graph per target) or
// <weights>/<name>/<name>.onnx, as scripts/export-*.py write them.
export const models = {
	umxhq: { modelType: 'openunmix', sampleRate: 44100, targets: ['vocals', 'drums', 'bass', 'other'], perTarget: true },
	htdemucs: { modelType: 'hybrid', sampleRate: 44100, targets: DEMUCS },
	htdemucs_ft: { modelType: 'hybrid', sampleRate: 44100, targets: DEMUCS, perTarget: true },
	// SCNet-large and SCNet (starrytong/SCNet, MIT weights) as scripts/export-scnet.py exports them
	'scnet-large': SCNET,
	scnet: SCNET,
	// MRX (merlresearch/cocktail-fork-separation, MIT weights) as scripts/export-mrx.py exports it: dialogue, music
	// and effects of a soundtrack (Divide and Remaster)
	mrx: MRX,
	// TIGER (JusperLee/TIGER, Apache-2.0 weights): the same three stems, three band-split models, about 50 times MRX's time
	tiger: TIGER,
}

async function cacheDir() {
	if (process.env.AUDIO_NEURAL_CACHE) return process.env.AUDIO_NEURAL_CACHE
	let [path, os] = await Promise.all([import('node:path'), import('node:os')])
	return path.join(os.homedir(), '.cache', 'audiojs', 'neural')
}

// Base URL of a preset's files: opts.weights (URL or, in Node, a directory), else the neural cache.
async function weightsBase(opts, name) {
	let w = opts.weights
	if (w == null && !isNode) throw new Error(`neural-separate: model '${name}' needs opts.weights in the browser: the URL its files are served from (${name}/<file>.onnx)`)
	if (w == null) w = await cacheDir()
	// a URL scheme has two characters at least, so a Windows drive (C:\) stays a path
	if (/^[a-z][a-z0-9+.-]+:/i.test(w)) return w.endsWith('/') ? w : w + '/'
	if (!isNode) throw new Error(`neural-separate: opts.weights must be a URL in the browser, got '${w}'`)
	let { pathToFileURL } = await import('node:url')
	return pathToFileURL(w.endsWith('/') ? w : w + '/').href
}

// Weights produced locally must exist before a session loads them: name the missing file and
// how to make it, rather than surfacing ENOENT from deep inside the runtime.
async function checkLocal(graphs, name) {
	let files = graphs.map(g => g.spec).filter(s => typeof s === 'string' && s.startsWith('file:'))
	if (!files.length) return
	let [{ existsSync }, { fileURLToPath }] = await Promise.all([import('node:fs'), import('node:url')])
	let missing = files.map(f => fileURLToPath(f)).filter(f => !existsSync(f))
	if (!missing.length) return
	let script = models[name].script ?? { hybrid: 'export-htdemucs.py', complex: 'export-scnet.py' }[models[name].modelType] ?? 'export-openunmix.py'
	throw new Error(`neural-separate: ${name} weights not found: ${missing.join(', ')}. ` +
		`Produce them with python3 node_modules/@audio/neural-separate/scripts/${script} --model ${name}, ` +
		`or pass opts.weights: a URL or directory holding ${name}/<file>.onnx`)
}

// opts.model → { graphs: [{ spec, sources, keep }], modelType, sampleRate, targets }. A graph's
// output stacks `sources` on its S axis (size 1 for a single target, the same flat layout as no
// axis); `keep` are the sources it contributes.
async function resolveModel(opts) {
	let model = opts.model
	if (model == null) throw new Error('neural-separate: opts.model is required (preset name | url | bytes | {target: url, ...} | {url, targets})')
	let want = opts.targets
	let pick = names => want ? names.filter(n => want.includes(n)) : names

	if (typeof model === 'string' && Object.hasOwn(models, model)) {
		let p = models[model], base = await weightsBase(opts, model)
		if (want) for (let t of want) if (!p.targets.includes(t)) throw new Error(`neural-separate: ${model} has no target '${t}' (has ${p.targets.join(', ')})`)
		let targets = pick(p.targets)
		let sources = p.sources ?? (p.modelType === 'hybrid' || p.modelType === 'complex' ? DEMUCS : null)
		let graphs = p.perTarget
			? targets.map(t => ({ spec: `${base}${model}/${t}.onnx`, sources: sources ?? [t], keep: [t] }))
			: [{ spec: `${base}${model}/${model}.onnx`, sources: sources ?? p.targets, keep: targets }]
		if (isNode) await checkLocal(graphs, model)
		return { graphs, modelType: p.modelType, sampleRate: p.sampleRate, targets, preset: p }
	}

	let graphs
	if (typeof model === 'string' || model instanceof Uint8Array) graphs = [{ spec: model, sources: ['stem'], keep: ['stem'] }]
	else if (model.url != null && Array.isArray(model.targets)) graphs = [{ spec: model.url, sources: model.targets, keep: pick(model.targets) }]
	else {
		let names = Object.keys(model)
		if (!names.length) throw new Error('neural-separate: opts.model target map is empty')
		graphs = pick(names).map(t => {
			let g = model[t]
			return g?.url != null && Array.isArray(g.targets) ? { spec: g.url, sources: g.targets, keep: [t] } : { spec: g, sources: [t], keep: [t] }
		})
	}
	graphs = graphs.filter(g => g.keep.length)
	if (!graphs.length) throw new Error(`neural-separate: none of targets ${want} is in the model`)
	return { graphs, modelType: opts.modelType ?? 'openunmix', sampleRate: null, targets: graphs.flatMap(g => g.keep) }
}

// No pooled arena: one htdemucs segment peaks at 2.7 GB RSS in onnxruntime-node instead of 3.2 GB.
const sessionOptions = { enableCpuMemArena: false, enableMemPattern: false }

function loader(opts) {
	return spec => opts.session ? opts.session(spec, opts) : neuralLoad(spec, { backend: opts.device, sessionOptions })
}

function outputOf(session, out, i = 0) {
	let name = session.outputs?.[i]?.name
	if (name && out[name]) return out[name]
	let vals = Object.values(out)
	if (!vals[i]) throw new Error(`neural-separate: model returned ${vals.length} output(s), expected at least ${i + 1}`)
	return vals[i]
}

// ------------------------------------------------------------------ chunking

// Overlap-add over spans [start, start + len): run(start, len) → { [target]: channel arrays of
// len samples }; each span weighted by weight(k, len), the sum normalized by the summed weights.
async function overlapAdd(N, C, starts, size, weight, run, progress) {
	let outputs = null, wsum = new Float64Array(N)
	for (let i = 0; i < starts.length; i++) {
		let start = starts[i], len = Math.min(size, N - start)
		progress?.({ chunk: i + 1, totalChunks: starts.length })
		let res = await run(start, len)
		if (!outputs) {
			outputs = {}
			for (let name in res) outputs[name] = Array.from({ length: C }, () => new Float64Array(N))
		}
		let w = new Float64Array(len)
		for (let k = 0; k < len; k++) { w[k] = weight(k, len); wsum[start + k] += w[k] }
		for (let name in res) for (let c = 0; c < C; c++) {
			let src = res[name][c], dst = outputs[name][c]
			for (let k = 0; k < len; k++) dst[start + k] += src[k] * w[k]
		}
	}
	let result = {}
	for (let name in outputs) result[name] = outputs[name].map(ch => {
		let out = new Float32Array(N)
		for (let i = 0; i < N; i++) if (wsum[i] > 0) out[i] = ch[i] / wsum[i]
		return out
	})
	return result
}

// Chunks of `chunk` seconds overlapping by `overlap`, linear crossfades (the spectral and
// waveform paths). One chunk when the input fits; the last runs on what remains.
function crossfadeChunks(N, rate, opts) {
	let chunkSec = opts.chunk ?? 30, overlapSec = opts.overlap ?? 2
	if (overlapSec >= chunkSec) throw new Error('neural-separate: opts.overlap must be smaller than opts.chunk')
	let size = Math.max(1, Math.round(chunkSec * rate))
	let ov = Math.max(0, Math.round(overlapSec * rate))
	let starts = [0]
	if (N > size) { starts = []; for (let s = 0; s < N; s += size - ov) { starts.push(s); if (s + size >= N) break } }
	let weight = (k, len) => Math.min(1, (k + 0.5) / ov, (len - k - 0.5) / ov)
	return { starts, size, weight: ov ? weight : () => 1 }
}

const slice = (channels, start, len) => channels.map(ch => ch.subarray(start, start + len))

// ------------------------------------------------------------------ spectral

// Magnitude tensor [1, C, F, T], T innermost (Open-Unmix's own layout), from flat spectra.
function packSpectral(mix, T, bins) {
	let C = mix.length, data = new Float32Array(C * bins * T)
	mix.forEach(({ re, im }, c) => {
		for (let t = 0, base = c * bins * T; t < T; t++) for (let f = 0, i = t * bins; f < bins; f++, i++) data[base + f * T + t] = Math.sqrt(re[i] * re[i] + im[i] * im[i])
	})
	return tensor(data, [1, C, bins, T], 'float32')
}

// Frames [t0, t1) of one target's model output ([C, F, T] from `base`) per channel, flat
// frame-major; modelType 'mask' multiplies the [0, 1] mask by the mixture magnitude (the
// packed input, same layout).
function estimate(data, base, mag, C, T, bins, t0, t1, mask) {
	return Array.from({ length: C }, (_, c) => {
		let v = new Float64Array((t1 - t0) * bins), b = base + c * bins * T, m = c * bins * T
		if (mask) for (let f = 0; f < bins; f++) for (let t = t0, i = f; t < t1; t++, i += bins) v[i] = data[b + f * T + t] * mag[m + f * T + t]
		else for (let f = 0; f < bins; f++) for (let t = t0, i = f; t < t1; t++, i += bins) v[i] = data[b + f * T + t]
		return v
	})
}

async function separateSpectral(channels, rate, opts, graphs, modelType, load) {
	let n = opts.n ?? 4096, hop = opts.hop ?? 1024, win = hann(n)
	let iterations = opts.wiener ?? 1
	let W = opts.wienerWindow ?? 300 // frames per EM window: openunmix Separator's wiener_win_len
	let names = graphs.flatMap(g => g.keep)
	// EM needs two sources: one target runs against the residual, open-unmix's residual=True
	let residual = names.length === 1 && iterations > 0
	let wopts = { iterations, softmask: opts.softmask ?? false, eps: opts.eps, residual }
	let sessions = []
	try {
		for (let g of graphs) sessions.push(await load(g.spec))
		let C = channels.length, N = channels[0].length
		let { starts, size, weight } = crossfadeChunks(N, rate, opts)
		return await overlapAdd(N, C, starts, size, weight, async (start, len) => {
			let mix = slice(channels, start, len).map(x => stftFlat(x, n, hop, win, true))
			let { T, bins } = mix[0]
			let input = packSpectral(mix, T, bins)
			let raw = {}
			for (let i = 0; i < graphs.length; i++) {
				let s = sessions[i], out = outputOf(s, await s.run({ [s.inputs?.[0]?.name ?? 'input']: input }))
				for (let name of graphs[i].keep) raw[name] = { data: out.data, base: graphs[i].sources.indexOf(name) * C * bins * T }
			}
			let acc = names.map(() => Array.from({ length: C }, () => new Float64Array((T - 1) * hop + n)))
			for (let t0 = 0; t0 < T; t0 += W) {
				let t1 = Math.min(T, t0 + W), view = x => x.subarray(t0 * bins, t1 * bins)
				let V = names.map(name => estimate(raw[name].data, raw[name].base, input.data, C, T, bins, t0, t1, modelType === 'mask'))
				let { Yr, Yi } = wiener(mix.map(m => view(m.re)), mix.map(m => view(m.im)), V, t1 - t0, bins, wopts)
				names.forEach((_, j) => { for (let c = 0; c < C; c++) ola(rows(Yr[j][c], bins), rows(Yi[j][c], bins), acc[j][c], t0, n, hop, win) })
			}
			let env = envelope(T, n, hop, win), res = {}
			names.forEach((name, j) => { res[name] = acc[j].map(o => olaFinish(o, env, n, true, len)) })
			return res
		}, opts.progress)
	} finally {
		for (let s of sessions) s.free?.()
	}
}

// ------------------------------------------------------------------ hybrid

// The input over its whole length: (x − mean) / deviation of the channels' mean (demucs.api, and demix's callers
// with inference.normalize); back(ch) scales a source back and adds the mean, as both do to every source; scale(ch)
// only scales it back, the mean (the mixture's DC) left to no source.
function normalize(channels) {
	let C = channels.length, N = channels[0].length, mono = new Float64Array(N), mean = 0, sq = 0
	for (let x of channels) for (let i = 0; i < N; i++) mono[i] += x[i] / C
	for (let i = 0; i < N; i++) mean += mono[i]
	mean /= N
	for (let i = 0; i < N; i++) sq += (mono[i] - mean) ** 2
	let std = Math.sqrt(sq / Math.max(1, N - 1)) + 1e-8
	return {
		norm: channels.map(x => Float32Array.from(x, v => (v - mean) / std)),
		back: ch => { for (let i = 0; i < ch.length; i++) ch[i] = ch[i] * std + mean; return ch },
		scale: ch => { for (let i = 0; i < ch.length; i++) ch[i] *= std; return ch },
	}
}

// One segment through a demucs.onnx-contract graph: STFT and iSTFT as HTDemucs._spec,
// _magnitude, _mask and _ispec do them (demucs/htdemucs.py, spec.py: torch.stft with
// normalized=True, n_fft 4096, hop 1024; the Nyquist bin dropped; reflect re-padding by 3/4
// hop so frames align with the time branch). mix: C channels of L samples; returns the
// sources at indices `want` (of the graph's S), C channels each.
async function hybridSegment(session, mix, want, n, hop) {
	let C = mix.length, L = mix[0].length, win = hann(n)
	let le = Math.ceil(L / hop), pad = (hop >> 1) * 3, F = n >> 1, bins = F + 1, scale = 1 / Math.sqrt(n)
	let spec = new Float32Array(C * 2 * F * le), wave = new Float32Array(C * L)
	mix.forEach((x, c) => {
		let padded = new Float64Array(le * hop + 2 * pad)
		for (let i = 0; i < padded.length; i++) padded[i] = x[reflectIndex(i - pad, L)]
		let { re, im } = stftFlat(padded, n, hop, win, true)
		let zr = spec.subarray(2 * c * F * le), zi = spec.subarray((2 * c + 1) * F * le)
		for (let f = 0, k = 0; f < F; f++) for (let t = 0, i = 2 * bins + f; t < le; t++, i += bins, k++) { zr[k] = re[i] * scale; zi[k] = im[i] * scale }
		wave.set(x, c * L)
	})
	let inputs = session.inputs ?? [{ name: 'mix' }, { name: 'mix_spec' }]
	let out = await session.run({ [inputs[0].name]: tensor(wave, [1, C, L], 'float32'), [inputs[1].name]: tensor(spec, [1, 2 * C, F, le], 'float32') })
	let zs = outputOf(session, out, 0).data, xt = outputOf(session, out, 1).data

	// per source and channel: zero Nyquist bin and two zero frames each side, iSTFT (× √n
	// undoes normalized=True), crop the re-padding, add the time branch
	let T = le + 4, len = hop * le + 2 * pad, unscale = Math.sqrt(n), env = envelope(T, n, hop, win)
	return want.map(s => Array.from({ length: C }, (_, c) => {
		let re = new Float64Array(T * bins), im = new Float64Array(T * bins), b = (s * 2 * C + 2 * c) * F * le
		// read the [F, T] planes in order, write frames strided
		for (let f = 0, k = b; f < F; f++) for (let t = 0, i = 2 * bins + f; t < le; t++, i += bins, k++) { re[i] = zs[k] * unscale; im[i] = zs[k + F * le] * unscale }
		let out = new Float64Array((T - 1) * hop + n)
		ola(rows(re, bins), rows(im, bins), out, 0, n, hop, win)
		let x = olaFinish(out, env, n, true, len)
		let y = new Float32Array(L), tb = (s * C + c) * L
		for (let i = 0; i < L; i++) y[i] = x[pad + i] + xt[tb + i]
		return y
	}))
}

// demucs.apply.apply_model(split=True, overlap=0.25, shifts=0) with demucs.api's whole-input
// normalization: segments of L samples every floor(0.75 L), each centered in L with the
// neighbouring input as context (TensorChunk.padded), center-trimmed, overlap-added under a
// triangular window normalized by its sum.
async function separateHybrid(channels, rate, opts, graphs, load) {
	let n = opts.n ?? 4096, hop = opts.hop ?? 1024
	let C = channels.length, N = channels[0].length, { norm, back } = normalize(channels)

	let result = {}
	for (let g of graphs) {
		let session = await load(g.spec)
		try {
			let L = session.inputs?.[0]?.dims?.[2]
			L = typeof L === 'number' && L > 0 ? L : opts.segment ?? 343980 // htdemucs: int(7.8 · 44100)
			let stride = Math.floor(0.75 * L) // apply_model's overlap=0.25
			let starts = []
			for (let s = 0; s < N; s += stride) starts.push(s)
			let half = L >> 1, tri = k => (k < half ? k + 1 : L - k) / half
			let keep = g.keep.map(name => g.sources.indexOf(name))
			let out = await overlapAdd(N, C, starts, L, tri, async (start, len) => {
				// the chunk centered in L samples, context from the input around it, zeros past its ends
				let from = start - ((L - len) >> 1)
				let seg = norm.map(x => { let s = new Float32Array(L), a = Math.max(0, from); s.set(x.subarray(a, Math.min(N, from + L)), a - from); return s })
				let stems = await hybridSegment(session, seg, keep, n, hop)
				let off = (L - len) >> 1, res = {}
				g.keep.forEach((name, i) => { res[name] = stems[i].map(ch => ch.subarray(off, off + len)) })
				return res
			}, opts.progress)
			for (let name in out) result[name] = out[name].map(back)
		} finally {
			session.free?.()
		}
	}
	return result
}

// ------------------------------------------------------------------ complex

// One segment through a complex-spectrogram graph, [1, 2C, F, T] in (channel c's STFT at 2c, re, and 2c + 1, im),
// [1, 2·S·C, F, T] out (source s, channel c at 2(sC + c), re, and the next, im): the segment zero-padded to
// (T − 1)·hop samples when shorter (SCNet.forward pads so), centered reflect-padded frames, × `scale` (normalized:
// 1/√n) and back. seg: C channels of L samples; returns the sources at indices `want`, C channels of L samples each.
async function complexSegment(session, seg, want, n, hop, win, scale, T) {
	let C = seg.length, L = seg[0].length, F = (n >> 1) + 1, spec = new Float32Array(2 * C * F * T)
	seg.forEach((x, c) => {
		let padded = new Float64Array(Math.max(L, (T - 1) * hop))
		padded.set(x)
		let { re, im } = stftFlat(padded, n, hop, win, true)
		let zr = spec.subarray(2 * c * F * T), zi = spec.subarray((2 * c + 1) * F * T)
		for (let f = 0, k = 0; f < F; f++) for (let t = 0, i = f; t < T; t++, i += F, k++) { zr[k] = re[i] * scale; zi[k] = im[i] * scale }
	})
	let name = session.inputs?.[0]?.name ?? 'mix_spec'
	let z = outputOf(session, await session.run({ [name]: tensor(spec, [1, 2 * C, F, T], 'float32') })).data
	let env = envelope(T, n, hop, win)
	return want.map(s => Array.from({ length: C }, (_, c) => {
		let re = new Float64Array(T * F), im = new Float64Array(T * F), b = 2 * (s * C + c) * F * T
		// the [F, T] planes read in order, frames written strided
		for (let f = 0, k = b; f < F; f++) for (let t = 0, i = f; t < T; t++, i += F, k++) { re[i] = z[k] / scale; im[i] = z[k + F * T] / scale }
		let out = new Float64Array((T - 1) * hop + n)
		ola(rows(re, F), rows(im, F), out, 0, n, hop, win)
		return olaFinish(out, env, n, true, L)
	}))
}

// demix() of Music-Source-Separation-Training (utils/model_utils.py, generic mode): segments of L samples every
// `step` of L, each weighted by a window fading in and out linearly over L/10 (the first not in, the last not out),
// the sum divided by the summed weights. An input longer than two segments less a step is first reflected L − step
// samples out on each side; a segment running past the end is reflected out when more than half of it is input,
// zero-padded when not. The input is normalized by its mean and deviation, and the sources scaled back; demix's
// callers add the mean back to every source, here to none: the stems sum to the input less its DC, which stays in
// the residual. And two departures: the segments stop at the first to reach the end (demix runs on while one
// starts before it: a 7 s song took three 11 s segments, now one, its SDR within 0.02 dB), and the first and last
// fades are set per segment (demix sets them per batch of 8: its first segment fades in, every one of a last
// batch holds).
// TIGER's DnR model (its preset's `step` 1/3, `fade` 0, `pad` 'zero', `standardize` false, `mono`) runs the segments
// TIGERDNR.wav_chunk_inference cuts: 12 s every 4 s, unweighted, the input zero-padded by 8 s at each end and the last
// segments with zeros, each channel apart, nothing normalized; its sum divided by the three segments over every
// sample, as the summed weights divide it here.
async function separateComplex(channels, rate, opts, graphs, load, p) {
	let n = opts.n ?? p?.n ?? 4096, hop = opts.hop ?? p?.hop ?? 1024
	let win = (opts.window ?? p?.window) === 'ones' ? new Float64Array(n).fill(1) : hann(n), scale = (opts.normalized ?? p?.normalized ?? true) ? 1 / Math.sqrt(n) : 1
	let C = channels.length, N = channels[0].length, zero = (opts.pad ?? p?.pad) === 'zero'
	let { norm, scale: back } = opts.standardize ?? p?.standardize ?? true ? normalize(channels) : { norm: channels, scale: c => c }
	let result = {}
	for (let g of graphs) {
		let session = await load(g.spec)
		try {
			let T = session.inputs?.[0]?.dims?.[3]
			T = typeof T === 'number' && T > 0 ? T : opts.frames ?? p?.frames
			let L = opts.segment ?? p?.segment ?? (T - 1) * hop, step = Math.max(1, Math.floor(L * (opts.step ?? p?.step ?? 0.25)))
			let border = L - step, wide = border > 0 && (zero || N > 2 * border)
			let x = wide ? norm.map(c => Float32Array.from({ length: N + 2 * border }, (_, i) => zero ? (i < border || i >= border + N ? 0 : c[i - border]) : c[reflectIndex(i - border, N)])) : norm
			let M = x[0].length, fade = Math.floor(L * (opts.fade ?? p?.fade ?? 0.1)), starts = []
			for (let s = 0; ; s += step) { starts.push(s); if (s + L >= M) break }
			let keep = g.keep.map(name => g.sources.indexOf(name))
			let acc = g.keep.map(() => Array.from({ length: C }, () => new Float64Array(M))), wsum = new Float64Array(M)
			for (let k = 0; k < starts.length; k++) {
				let start = starts[k], len = Math.min(L, M - start), reflect = !zero && len > L / 2
				opts.progress?.({ chunk: k + 1, totalChunks: starts.length })
				let seg = x.map(c => { let s = new Float32Array(L); for (let i = 0; i < L; i++) s[i] = i < len ? c[start + i] : reflect ? c[start + reflectIndex(i, len)] : 0; return s })
				// a mono model hears each channel apart
				let stems = p?.mono ? [] : await complexSegment(session, seg, keep, n, hop, win, scale, T)
				if (p?.mono) for (let c = 0; c < C; c++) (await complexSegment(session, [seg[c]], keep, n, hop, win, scale, T)).forEach((st, j) => (stems[j] ??= [])[c] = st[0])
				for (let i = 0; i < len; i++) {
					let w = 1
					if (k > 0 && i < fade) w = i / (fade - 1)
					if (k < starts.length - 1 && i >= L - fade) w = Math.min(w, (L - 1 - i) / (fade - 1))
					wsum[start + i] += w
					for (let j = 0; j < keep.length; j++) for (let c = 0; c < C; c++) acc[j][c][start + i] += stems[j][c][i] * w
				}
			}
			g.keep.forEach((name, j) => {
				result[name] = acc[j].map(a => back(Float32Array.from({ length: N }, (_, i) => wsum[i + (wide ? border : 0)] > 0 ? a[i + (wide ? border : 0)] / wsum[i + (wide ? border : 0)] : 0)))
			})
		} finally {
			session.free?.()
		}
	}
	return result
}

// ------------------------------------------------------------------ multires

// Each channel apart (B = 1), chunk by chunk: its STFT at every resolution (Hann windows of `windows` samples, one
// hop, centered), the magnitudes scaled 1/√n (torch.stft's normalized=True) in as mag_<n> [1, F, T], masks out as
// mask_<n> [1, S, F, T] in the graph's source order; a source is the sum over resolutions of the iSTFT of its mask
// times the spectrogram (the masks are real: the 1/√n and the iSTFT's √n cancel).
async function separateMultires(channels, rate, opts, graphs, load, p) {
	let windows = opts.windows ?? p?.windows, hop = opts.hop ?? p?.hop ?? 256
	let C = channels.length, N = channels[0].length, result = {}
	for (let g of graphs) {
		let session = await load(g.spec)
		try {
			let { starts, size, weight } = crossfadeChunks(N, rate, { ...opts, chunk: opts.chunk ?? p?.chunk })
			Object.assign(result, await overlapAdd(N, C, starts, size, weight, async (start, len) => {
				let res = Object.fromEntries(g.keep.map(name => [name, []]))
				for (let c = 0; c < C; c++) {
					let x = channels[c].subarray(start, start + len), feeds = {}
					// the spectra kept in float32 (the 8192 window's alone 113 MB a 20 s chunk), one masked copy reused
					let specs = windows.map((n, r) => {
						let { re, im, T, bins: F } = stftFlat(x, n, hop, hann(n), true), mag = new Float32Array(F * T), k = 1 / Math.sqrt(n)
						for (let t = 0, i = 0; t < T; t++) for (let f = 0; f < F; f++, i++) mag[f * T + t] = Math.hypot(re[i], im[i]) * k
						feeds[session.inputs?.[r]?.name ?? `mag_${n}`] = tensor(mag, [1, F, T], 'float32')
						return { re: Float32Array.from(re), im: Float32Array.from(im), T, F }
					})
					let out = await session.run(feeds), acc = g.keep.map(() => new Float64Array(len))
					specs.forEach(({ re, im, T, F }, r) => {
						let n = windows[r], m = outputOf(session, out, r).data, mr = new Float64Array(T * F), mi = new Float64Array(T * F)
						g.keep.forEach((name, j) => {
							let b = g.sources.indexOf(name) * F * T
							for (let f = 0; f < F; f++) for (let t = 0, i = f, q = b + f * T; t < T; t++, i += F, q++) { mr[i] = re[i] * m[q]; mi[i] = im[i] * m[q] }
							let o = new Float64Array((T - 1) * hop + n)
							ola(rows(mr, F), rows(mi, F), o, 0, n, hop, hann(n))
							let y = olaFinish(o, envelope(T, n, hop, hann(n)), n, true, len)
							for (let i = 0; i < len; i++) acc[j][i] += y[i]
						})
					})
					g.keep.forEach((name, j) => res[name].push(acc[j]))
				}
				return res
			}, opts.progress))
		} finally {
			session.free?.()
		}
	}
	return result
}

// ------------------------------------------------------------------ loudness

// Integrated loudness, ITU-R BS.1770-4, as pyloudnorm 0.2 measures it (upstream MRX's separate.py sets its input with
// it): K-weighting by pyloudnorm's two biquads at any rate (a 4 dB shelf over 1.5 kHz, a 38 Hz high-pass), 400 ms
// blocks every 100 ms, gated at -70 LUFS, then 10 LU under the mean of the rest; -Infinity for silence.
function lufs(channels, rate) {
	let shelf = (G, Q, fc) => {
		let A = 10 ** (G / 40), w = 2 * Math.PI * fc / rate, al = Math.sin(w) / (2 * Q), c = Math.cos(w), r = 2 * Math.sqrt(A) * al
		return [A * ((A + 1) + (A - 1) * c + r), -2 * A * ((A - 1) + (A + 1) * c), A * ((A + 1) + (A - 1) * c - r), (A + 1) - (A - 1) * c + r, 2 * ((A - 1) - (A + 1) * c), (A + 1) - (A - 1) * c - r]
	}
	let pass = (Q, fc) => { let w = 2 * Math.PI * fc / rate, al = Math.sin(w) / (2 * Q), c = Math.cos(w); return [(1 + c) / 2, -(1 + c), (1 + c) / 2, 1 + al, -2 * c, 1 - al] }
	let filters = [shelf(4, 1 / Math.sqrt(2), 1500), pass(0.5, 38)], N = channels[0].length, T = 0.4, step = 0.25
	// numpy's round, half to even, as pyloudnorm counts its blocks
	let even = v => { let f = Math.floor(v), d = v - f; return d > 0.5 || (d === 0.5 && f % 2) ? f + 1 : f }
	let blocks = even((N / rate - T) / (T * step)) + 1
	if (!(blocks > 0)) return -Infinity
	let z = new Float64Array(blocks), G = [1, 1, 1, 1.41, 1.41]
	channels.forEach((x, c) => {
		let y = Float64Array.from(x)
		for (let [b0, b1, b2, a0, a1, a2] of filters) {
			// scipy.signal.lfilter from rest: direct form II transposed, coefficients over a0
			b0 /= a0; b1 /= a0; b2 /= a0; a1 /= a0; a2 /= a0
			let z1 = 0, z2 = 0
			for (let i = 0; i < N; i++) { let v = y[i], o = b0 * v + z1; z1 = b1 * v - a1 * o + z2; z2 = b2 * v - a2 * o; y[i] = o }
		}
		for (let j = 0; j < blocks; j++) {
			let lo = Math.trunc(T * (j * step) * rate), hi = Math.min(N, Math.trunc(T * (j * step + 1) * rate)), s = 0
			for (let i = lo; i < hi; i++) s += y[i] * y[i]
			z[j] += (G[c] ?? 1) * s / (T * rate)
		}
	})
	let L = v => -0.691 + 10 * Math.log10(v), mean = v => v.reduce((a, b) => a + b, 0) / v.length
	let abs = [...z].filter(v => L(v) >= -70)
	if (!abs.length) return -Infinity
	let rel = L(mean(abs)) - 10, gated = [...z].filter(v => L(v) > rel && L(v) > -70)
	return gated.length ? L(mean(gated)) : -Infinity
}

// ------------------------------------------------------------------ waveform

// [1, C, N] in, [1, S, C, N] out, contiguous per (source, channel).
async function separateWaveform(channels, rate, opts, graphs, load) {
	let C = channels.length, N = channels[0].length
	let { starts, size, weight } = crossfadeChunks(N, rate, opts)
	let result = {}
	for (let g of graphs) {
		let session = await load(g.spec)
		try {
			let out = await overlapAdd(N, C, starts, size, weight, async (start, len) => {
				let data = new Float32Array(C * len)
				slice(channels, start, len).forEach((ch, c) => data.set(ch, c * len))
				let y = outputOf(session, await session.run({ [session.inputs?.[0]?.name ?? 'input']: tensor(data, [1, C, len], 'float32') })).data
				let res = {}
				for (let name of g.keep) { let s = g.sources.indexOf(name); res[name] = Array.from({ length: C }, (_, c) => y.subarray((s * C + c) * len, (s * C + c + 1) * len)) }
				return res
			}, opts.progress)
			Object.assign(result, out)
		} finally {
			session.free?.()
		}
	}
	return result
}

// -------------------------------------------------------------------- input

function normalizeAudio(audio, opts) {
	let channelData, sampleRate
	if (Array.isArray(audio)) { channelData = audio; sampleRate = opts.sampleRate }
	else if (audio && audio.channelData) { channelData = audio.channelData; sampleRate = audio.sampleRate ?? opts.sampleRate }
	else throw new Error('neural-separate: audio must be Float32Array[] or { channelData, sampleRate }')
	if (!sampleRate) throw new Error('neural-separate: sampleRate is required (opts.sampleRate, or audio.sampleRate)')
	if (!channelData.length) throw new Error('neural-separate: audio has no channels')
	return { channelData, sampleRate }
}

function fitLength(arr, len) {
	if (arr.length === len) return arr
	let out = new Float32Array(len)
	out.set(arr.subarray(0, Math.min(len, arr.length)))
	return out
}

// ------------------------------------------------------------------- default

// separate(audio, opts) → { stems: { [target]: Float32Array[] }, sampleRate, residual }
export default async function separate(audio, opts = {}) {
	if (opts.dtype && opts.dtype !== 'float32') throw new Error(`neural-separate: dtype '${opts.dtype}' not supported: only 'float32' tensor marshalling is implemented`)

	let { channelData, sampleRate: rate } = normalizeAudio(audio, opts)
	let { graphs, modelType, sampleRate: modelRate, targets, preset } = await resolveModel(opts)
	// mono → duplicated stereo, as openunmix.utils.preprocess does; a model hearing channels apart takes it as it is
	if (channelData.length === 1 && !preset?.mono) channelData = [channelData[0], channelData[0]]
	let empty = () => channelData.map(() => new Float32Array(0))
	if (!channelData[0].length) return { stems: Object.fromEntries(targets.map(t => [t, empty()])), sampleRate: rate, residual: empty() }
	let targetRate = opts.targetRate ?? modelRate ?? rate
	let proc = targetRate !== rate ? channelData.map(c => resampleSinc(c, { from: rate, to: targetRate })) : channelData
	let load = loader(opts)
	// the input at the loudness the model trained at, its stems scaled back (a silent input as it is)
	let level = preset?.loudness == null ? -Infinity : lufs(proc, targetRate), gain = Number.isFinite(level) ? 10 ** ((preset.loudness - level) / 20) : 1
	if (gain !== 1) proc = proc.map(c => c.map(v => v * gain))

	let stemsAtRate = modelType === 'hybrid' ? await separateHybrid(proc, targetRate, opts, graphs, load)
		: modelType === 'complex' ? await separateComplex(proc, targetRate, opts, graphs, load, preset)
		: modelType === 'multires' ? await separateMultires(proc, targetRate, opts, graphs, load, preset)
		: modelType === 'waveform' ? await separateWaveform(proc, targetRate, opts, graphs, load)
		: await separateSpectral(proc, targetRate, opts, graphs, modelType, load)

	let N = channelData[0].length
	let stems = {}
	for (let name of targets) stems[name] = stemsAtRate[name].map(c => {
		if (gain !== 1) c = Float32Array.from(c, v => v / gain)
		return fitLength(targetRate !== rate ? resampleSinc(c, { from: targetRate, to: rate }) : c, N)
	})

	let residual = channelData.map((c, ci) => {
		let r = Float32Array.from(c)
		for (let name of targets) { let s = stems[name][ci]; for (let i = 0; i < N; i++) r[i] -= s[i] }
		return r
	})

	return { stems, sampleRate: rate, residual }
}
