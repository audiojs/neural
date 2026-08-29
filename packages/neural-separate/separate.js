// Source separation (stems) — spectrogram-mask models of the Open-Unmix class
// (Stöter, Uhlich, Liutkus, Mitsufuji, "Open-Unmix - A Reference Implementation
// for Music Source Separation", JOSS 2019) plus Demucs-class waveform models,
// through the model-agnostic ONNX adapter @audio/neural-runtime.
//
// Pipeline (spectral / 'openunmix' | 'mask'):
//   STFT (n 4096, hop 1024, Hann, center=True — torch.stft-compatible) → per-
//   channel magnitude → one ONNX run per target → multichannel Wiener EM
//   refinement (Duong, Vincent, Gribonval, "Under-determined reverberant
//   audio source separation using a full-rank spatial covariance model",
//   IEEE TASLP 2010; algorithm and defaults ported from sigsep/norbert and
//   open-unmix-pytorch's openunmix/filtering.py) → iSTFT.
//
// Pipeline ('waveform', Demucs-class): chunked raw waveform → ONNX → stacked
// per-target waveforms directly, no STFT/Wiener step (Demucs operates in the
// time domain by design).
//
// Long files are processed in overlapping chunks (bi-LSTM/model memory is
// bounded per chunk) and stitched back with a linear crossfade.

import { fft, ifft } from 'fourier-transform'
import resampleSinc from '@audio/resample-sinc'
import { load as neuralLoad, tensor } from '@audio/neural-runtime'

const PI2 = Math.PI * 2

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

// numpy/torch 'reflect' pad index map: mirrors the signal without repeating
// the edge sample, period 2(n-1) — handles any pad width via modulo, so it
// stays correct even when pad exceeds the signal length.
function reflectIndex(i, n) {
	if (n === 1) return 0
	let period = 2 * (n - 1)
	i = ((i % period) + period) % period
	return i < n ? i : period - i
}

// stft(x, opts) — mono complex STFT, torch.stft(center=True, pad_mode='reflect')
// compatible: frame count = 1 + floor(N/hop), independent of n (matches
// torch's centered framing formula since the n_fft/2 reflect-pad on each side
// cancels out of the frame-count arithmetic).
export function stft(x, opts = {}) {
	let n = opts.n ?? 4096
	let hop = opts.hop ?? 1024
	let win = opts.window ?? hann(n)
	let center = opts.center ?? true
	let N = x.length
	let bins = (n >> 1) + 1
	let pad = center ? n >> 1 : 0
	let nFrames = center
		? 1 + Math.floor(N / hop)
		: (N < n ? 0 : 1 + Math.floor((N - n) / hop))

	let padded
	if (center) {
		padded = new Float64Array(N + 2 * pad)
		for (let i = 0; i < padded.length; i++) padded[i] = x[reflectIndex(i - pad, N)]
	}

	let re = new Array(nFrames), im = new Array(nFrames)
	let frame = new Float64Array(n)
	for (let m = 0; m < nFrames; m++) {
		let pos = m * hop
		if (center) for (let i = 0; i < n; i++) frame[i] = padded[pos + i] * win[i]
		else for (let i = 0; i < n; i++) frame[i] = (x[pos + i] ?? 0) * win[i]
		let [r, ii] = fft(frame) // fourier-transform: [re,im] each length n/2+1, cached — copy before next call
		re[m] = Float64Array.from(r)
		im[m] = Float64Array.from(ii)
	}
	return { re, im, n, hop, bins, center }
}

// istft(frames, opts) — exact inverse: weighted overlap-add with the
// squared-window normalization (WOLA), the same scheme torch.istft uses —
// perfect reconstruction whenever the window/hop combination satisfies NOLA
// for the squared window (Hann at 75% overlap, n/hop = 4, does). `opts.length`
// crops/pads the output to an exact sample count (mirrors torch.istft's
// `length` argument); default drops the center padding only.
export function istft(frames, opts = {}) {
	let { re, im, n, hop, center = true } = frames
	let win = opts.window ?? hann(n)
	let nFrames = re.length
	let paddedLen = nFrames > 0 ? (nFrames - 1) * hop + n : 0
	let out = new Float64Array(paddedLen)
	let norm = new Float64Array(paddedLen)
	for (let m = 0; m < nFrames; m++) {
		let f = ifft(re[m], im[m]) // Float64Array(n), internal cached buffer — consumed before next call
		let pos = m * hop
		for (let i = 0; i < n; i++) {
			out[pos + i] += f[i] * win[i]
			norm[pos + i] += win[i] * win[i]
		}
	}
	for (let i = 0; i < paddedLen; i++) if (norm[i] > 1e-8) out[i] /= norm[i]

	let pad = center ? n >> 1 : 0
	let length = opts.length ?? Math.max(0, paddedLen - 2 * pad)
	let result = new Float32Array(length)
	for (let i = 0; i < length; i++) {
		let j = i + pad
		if (j >= 0 && j < paddedLen) result[i] = out[j]
	}
	return result
}

// ------------------------------------------------------------- Wiener filter

// wienerFilter(mixStft, estimates, opts) — norbert's algorithm (Liutkus &
// Stöter, github.com/sigsep/norbert) with open-unmix-pytorch's defaults
// (openunmix/filtering.py: eps=1e-10, softmask=False, scale_factor=10).
//
//   mixStft: array of per-channel {re, im, n, hop, center} — stft() output
//     per channel.
//   estimates: { [target]: magnitude[channel][frame] } (Float64Array(bins)
//     per frame); a single-channel estimate broadcasts to all mix channels
//     (spectrograms may be mono — norbert's convention).
//   opts.iterations (default 1): EM refinement steps; 0 returns the initial
//     (softmask or phase-substituted) estimate untouched — "raw masks".
//   opts.softmask (default false): true = ratio mask (sums to the mixture by
//     construction); false = target magnitude with the mixture's own phase
//     (open-unmix's recommended default — better once the model estimates
//     are themselves good).
//   opts.residual (default false): appends a 'residual' target equal to the
//     mixture minus the sum of the other targets (computed before EM).
//   opts.eps (default 1e-10): regularization floor, also the softmask/EM
//     division guard.
//
// Returns { [target]: per-channel {re, im, n, hop, center} } — ready for
// istft() per channel.
export function wienerFilter(mixStft, estimates, opts = {}) {
	let { iterations = 1, softmask = false, residual = false, eps = 1e-10, scaleFactor = 10 } = opts
	let C = mixStft.length
	let T = mixStft[0].re.length
	let bins = mixStft[0].bins
	let names = Object.keys(estimates)
	if (!names.length) throw new Error('wienerFilter: estimates has no targets')

	let v = names.map(name => {
		let e = estimates[name]
		if (e.length === C) return e
		if (e.length === 1) return Array.from({ length: C }, () => e[0])
		throw new Error(`wienerFilter: estimate '${name}' has ${e.length} channel(s), mixture has ${C}`)
	})

	let mixRe = mixStft.map(s => s.re), mixIm = mixStft.map(s => s.im)
	let S0 = names.length

	let yre = alloc(S0, C, T, bins), yim = alloc(S0, C, T, bins)

	if (softmask) {
		for (let c = 0; c < C; c++) for (let t = 0; t < T; t++) {
			let mr = mixRe[c][t], mi = mixIm[c][t]
			for (let f = 0; f < bins; f++) {
				let total = eps
				for (let j = 0; j < S0; j++) total += v[j][c][t][f]
				let re = mr[f], im = mi[f]
				for (let j = 0; j < S0; j++) {
					let ratio = v[j][c][t][f] / total
					yre[j][c][t][f] = ratio * re
					yim[j][c][t][f] = ratio * im
				}
			}
		}
	} else {
		for (let c = 0; c < C; c++) for (let t = 0; t < T; t++) {
			let mr = mixRe[c][t], mi = mixIm[c][t]
			for (let f = 0; f < bins; f++) {
				let re = mr[f], im = mi[f]
				let mag = Math.sqrt(re * re + im * im)
				let cos = mag > 0 ? re / mag : 1, sin = mag > 0 ? im / mag : 0
				for (let j = 0; j < S0; j++) {
					let m = v[j][c][t][f]
					yre[j][c][t][f] = m * cos
					yim[j][c][t][f] = m * sin
				}
			}
		}
	}

	let targetNames = names.slice()
	if (residual) {
		targetNames.push('residual')
		let rre = alloc(1, C, T, bins)[0], rim = alloc(1, C, T, bins)[0]
		for (let c = 0; c < C; c++) for (let t = 0; t < T; t++) {
			let out_re = rre[c][t], out_im = rim[c][t]
			for (let f = 0; f < bins; f++) {
				let sre = 0, sim = 0
				for (let j = 0; j < S0; j++) { sre += yre[j][c][t][f]; sim += yim[j][c][t][f] }
				out_re[f] = mixRe[c][t][f] - sre
				out_im[f] = mixIm[c][t][f] - sim
			}
		}
		yre.push(rre); yim.push(rim)
	}
	let S = targetNames.length

	if (iterations > 0) runEM(yre, yim, mixRe, mixIm, C, T, bins, S, iterations, eps, scaleFactor)

	let result = {}
	for (let j = 0; j < S; j++) {
		result[targetNames[j]] = []
		for (let c = 0; c < C; c++) result[targetNames[j]].push({ re: yre[j][c], im: yim[j][c], n: mixStft[0].n, hop: mixStft[0].hop, center: mixStft[0].center })
	}
	return result
}

function alloc(S, C, T, bins) {
	let out = new Array(S)
	for (let j = 0; j < S; j++) {
		let cs = new Array(C)
		for (let c = 0; c < C; c++) {
			let ts = new Array(T)
			for (let t = 0; t < T; t++) ts[t] = new Float64Array(bins)
			cs[c] = ts
		}
		out[j] = cs
	}
	return out
}

// expectation_maximization, ported from norbert (numpy) / openunmix's torch
// filtering.py: re-estimate each source's power spectral density + spatial
// covariance matrix, rebuild the mixture covariance model, apply the
// resulting multichannel Wiener gain — iterated. Complex CxC linear algebra
// via a generic Gauss-Jordan inverse (norbert/umx hand-special-case 1 and 2
// channels only; this stays correct for any channel count, at the cost the
// reference implementations chose to avoid — negligible for C ≤ a handful).
function runEM(yre, yim, mixRe, mixIm, C, T, bins, S, iterations, eps, scaleFactor) {
	let maxPow = 0
	for (let c = 0; c < C; c++) for (let t = 0; t < T; t++) {
		let re = mixRe[c][t], im = mixIm[c][t]
		for (let f = 0; f < bins; f++) { let p = re[f] * re[f] + im[f] * im[f]; if (p > maxPow) maxPow = p }
	}
	let maxAbs = Math.max(1, Math.sqrt(maxPow) / scaleFactor)
	let invMax = 1 / maxAbs

	for (let j = 0; j < S; j++) for (let c = 0; c < C; c++) for (let t = 0; t < T; t++) {
		let re = yre[j][c][t], im = yim[j][c][t]
		for (let f = 0; f < bins; f++) { re[f] *= invMax; im[f] *= invMax }
	}

	let vv = alloc(S, 1, T, bins).map(x => x[0]) // vv[j][t] = Float64Array(bins) — PSD, mean power over channels
	let R_re = new Float64Array(S * bins * C * C), R_im = new Float64Array(S * bins * C * C)
	let weight = new Float64Array(bins)
	let Cxx_re = new Float64Array(C * C), Cxx_im = new Float64Array(C * C)
	let inv_re = new Float64Array(C * C), inv_im = new Float64Array(C * C)
	let G_re = new Float64Array(C * C), G_im = new Float64Array(C * C)
	let mxRe = new Float64Array(C), mxIm = new Float64Array(C)
	let scratchRe = new Float64Array(C * 2 * C), scratchIm = new Float64Array(C * 2 * C)
	let sqrtEps = Math.sqrt(eps)

	for (let iter = 0; iter < iterations; iter++) {
		// 1. power spectral density per source: mean over channels of |y_j|^2
		for (let j = 0; j < S; j++) for (let t = 0; t < T; t++) {
			let out = vv[j][t]
			out.fill(0)
			for (let c = 0; c < C; c++) {
				let re = yre[j][c][t], im = yim[j][c][t]
				for (let f = 0; f < bins; f++) out[f] += re[f] * re[f] + im[f] * im[f]
			}
			for (let f = 0; f < bins; f++) out[f] /= C
		}

		// 2. spatial covariance R[j] per bin (C×C complex), weighted average
		// over frames: R_j = (Σ_t y_j[t]⊗conj(y_j[t])) / (eps + Σ_t v_j[t])
		R_re.fill(0); R_im.fill(0)
		for (let j = 0; j < S; j++) {
			weight.fill(eps)
			for (let t = 0; t < T; t++) { let vt = vv[j][t]; for (let f = 0; f < bins; f++) weight[f] += vt[f] }
			let jBase = j * bins * C * C
			for (let c1 = 0; c1 < C; c1++) for (let c2 = 0; c2 < C; c2++) {
				let cIdx = c1 * C + c2
				for (let t = 0; t < T; t++) {
					let re1 = yre[j][c1][t], im1 = yim[j][c1][t]
					let re2 = yre[j][c2][t], im2 = yim[j][c2][t]
					for (let f = 0; f < bins; f++) {
						let idx = jBase + f * C * C + cIdx
						// y1 * conj(y2)
						R_re[idx] += re1[f] * re2[f] + im1[f] * im2[f]
						R_im[idx] += im1[f] * re2[f] - re1[f] * im2[f]
					}
				}
			}
			for (let f = 0; f < bins; f++) {
				let w = weight[f]
				let base = jBase + f * C * C
				for (let idx = 0; idx < C * C; idx++) { R_re[base + idx] /= w; R_im[base + idx] /= w }
			}
		}

		// 3. per (frame, bin): mixture covariance model, invert, apply the
		// resulting Wiener gain per source
		for (let t = 0; t < T; t++) for (let f = 0; f < bins; f++) {
			Cxx_re.fill(0); Cxx_im.fill(0)
			for (let j = 0; j < S; j++) {
				let vjt = vv[j][t][f]
				if (vjt === 0) continue
				let base = j * bins * C * C + f * C * C
				for (let idx = 0; idx < C * C; idx++) { Cxx_re[idx] += vjt * R_re[base + idx]; Cxx_im[idx] += vjt * R_im[base + idx] }
			}
			for (let c = 0; c < C; c++) Cxx_re[c * C + c] += sqrtEps
			complexInverse(Cxx_re, Cxx_im, inv_re, inv_im, C, scratchRe, scratchIm)

			for (let c = 0; c < C; c++) { mxRe[c] = mixRe[c][t][f] * invMax; mxIm[c] = mixIm[c][t][f] * invMax }

			for (let j = 0; j < S; j++) {
				let vjt = vv[j][t][f]
				let base = j * bins * C * C + f * C * C
				// G = v_j · R_j @ inv(Cxx)
				for (let i1 = 0; i1 < C; i1++) for (let i2 = 0; i2 < C; i2++) {
					let sre = 0, sim = 0
					for (let i3 = 0; i3 < C; i3++) {
						let rre = R_re[base + i1 * C + i3], rim = R_im[base + i1 * C + i3]
						let ire = inv_re[i3 * C + i2], iim = inv_im[i3 * C + i2]
						sre += rre * ire - rim * iim
						sim += rre * iim + rim * ire
					}
					G_re[i1 * C + i2] = sre * vjt
					G_im[i1 * C + i2] = sim * vjt
				}
				// y_j = G @ mix
				for (let c1 = 0; c1 < C; c1++) {
					let sre = 0, sim = 0
					for (let c2 = 0; c2 < C; c2++) {
						let gre = G_re[c1 * C + c2], gim = G_im[c1 * C + c2]
						sre += gre * mxRe[c2] - gim * mxIm[c2]
						sim += gre * mxIm[c2] + gim * mxRe[c2]
					}
					yre[j][c1][t][f] = sre
					yim[j][c1][t][f] = sim
				}
			}
		}
	}

	for (let j = 0; j < S; j++) for (let c = 0; c < C; c++) for (let t = 0; t < T; t++) {
		let re = yre[j][c][t], im = yim[j][c][t]
		for (let f = 0; f < bins; f++) { re[f] *= maxAbs; im[f] *= maxAbs }
	}
}

// Complex Gauss-Jordan inverse with partial pivoting, on n×n matrices packed
// as flat row-major Float64Array(n*n). scratchRe/scratchIm are caller-owned
// n×2n work buffers (reused across calls — this runs once per (frame, bin)).
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

// ------------------------------------------------------ ONNX tensor packing

// Spectral I/O contract (modelType 'openunmix' | 'mask'): magnitude tensor
// shape [1, C, F, T] (batch, channels, freq bins, time frames), row-major,
// T innermost — see scripts/export-openunmix.py for the export-side wrapper
// that produces this layout from OpenUnmix's native (frames, batch, channels,
// bins) tensor. Multi-target graphs stack an extra S axis after batch:
// [1, S, C, F, T].

function packSpectral(mixStft, C, T, bins) {
	let data = new Float32Array(C * bins * T)
	for (let c = 0; c < C; c++) {
		let re = mixStft[c].re, im = mixStft[c].im, base = c * bins * T
		for (let t = 0; t < T; t++) {
			let rt = re[t], it = im[t]
			for (let f = 0; f < bins; f++) { let r = rt[f], i = it[f]; data[base + f * T + t] = Math.sqrt(r * r + i * i) }
		}
	}
	return tensor(data, [1, C, bins, T], 'float32')
}

function unpackSpectral(data, sBase, C, T, bins) {
	let result = new Array(C)
	for (let c = 0; c < C; c++) {
		let base = sBase + c * bins * T
		let chan = new Array(T)
		for (let t = 0; t < T; t++) {
			let arr = new Float64Array(bins)
			for (let f = 0; f < bins; f++) arr[f] = data[base + f * T + t]
			chan[t] = arr
		}
		result[c] = chan
	}
	return result
}

function applyMask(mask, mixStft, C, T, bins) {
	let result = new Array(C)
	for (let c = 0; c < C; c++) {
		let re = mixStft[c].re, im = mixStft[c].im
		let chan = new Array(T)
		for (let t = 0; t < T; t++) {
			let mrow = mask[c][t], rrow = re[t], irow = im[t]
			let out = new Float64Array(bins)
			for (let f = 0; f < bins; f++) { let r = rrow[f], i = irow[f]; out[f] = mrow[f] * Math.sqrt(r * r + i * i) }
			chan[t] = out
		}
		result[c] = chan
	}
	return result
}

// Waveform I/O contract (modelType 'waveform', Demucs-class): [1, C, N] in,
// [1, S, C, N] out — contiguous per (source, channel), no transpose needed.

function packWaveform(chunkChannels, C, N) {
	let data = new Float32Array(C * N)
	for (let c = 0; c < C; c++) data.set(chunkChannels[c], c * N)
	return tensor(data, [1, C, N], 'float32')
}

function unpackWaveform(data, S, C, N) {
	let result = new Array(S)
	for (let s = 0; s < S; s++) {
		let chans = new Array(C)
		for (let c = 0; c < C; c++) { let base = (s * C + c) * N; chans[c] = Float32Array.from(data.subarray(base, base + N)) }
		result[s] = chans
	}
	return result
}

function feedName(session) { return session.inputs?.[0]?.name ?? 'input' }
function firstOutput(session, out) {
	let name = session.outputs?.[0]?.name
	if (name && out[name]) return out[name]
	let vals = Object.values(out)
	if (!vals.length) throw new Error('neural-separate: model returned no outputs')
	return vals[0]
}

// ------------------------------------------------------------------ chunking

// Long files are processed in overlapping chunks (chunk seconds, overlap
// seconds) and stitched back with a linear crossfade — trades a fixed memory/
// latency ceiling per chunk against a slightly less globally-informed
// estimate at chunk boundaries (the bi-LSTM/model only sees `chunk` seconds
// of context, not the whole track). `processChunk(channels, length) →
// { [target]: Float32Array[] }` runs the model (+ STFT/Wiener, or not) on one
// chunk; chunkedProcess owns the outer loop, padding and crossfade.
async function chunkedProcess(channelData, rate, opts, processChunk) {
	let C = channelData.length, N = channelData[0].length
	let chunkSec = opts.chunk ?? 30, overlapSec = opts.overlap ?? 2
	if (overlapSec >= chunkSec) throw new Error('neural-separate: opts.overlap must be smaller than opts.chunk')
	let chunkLen = Math.max(1, Math.round(chunkSec * rate))
	let overlapLen = Math.max(0, Math.round(overlapSec * rate))
	let stepLen = chunkLen - overlapLen

	let starts = [0]
	if (N > chunkLen) { starts = []; for (let s = 0; s < N; s += stepLen) { starts.push(s); if (s + chunkLen >= N) break } }
	let numChunks = starts.length

	let outputs = null, weightSum = new Float64Array(N)

	for (let i = 0; i < numChunks; i++) {
		let start = starts[i]
		let len = numChunks === 1 ? N : chunkLen // single chunk: exact length, no wasted padding
		let chunkChannels = channelData.map(ch => {
			let seg = new Float32Array(len)
			for (let k = 0; k < len; k++) { let idx = start + k; if (idx < N) seg[k] = ch[idx] }
			return seg
		})

		opts.progress?.({ chunk: i + 1, totalChunks: numChunks })
		let chunkStems = await processChunk(chunkChannels, len)

		if (!outputs) {
			outputs = {}
			for (let name in chunkStems) outputs[name] = channelData.map(() => new Float64Array(N))
		}

		let hasPrev = i > 0, hasNext = i < numChunks - 1
		for (let k = 0; k < len; k++) {
			let idx = start + k
			if (idx >= N) break
			let w = 1
			if (hasPrev && k < overlapLen) w = (k + 0.5) / overlapLen
			else if (hasNext && k >= len - overlapLen) w = (len - k - 0.5) / overlapLen
			weightSum[idx] += w
			for (let name in chunkStems) {
				let src = chunkStems[name], dst = outputs[name]
				for (let c = 0; c < C; c++) dst[c][idx] += src[c][k] * w
			}
		}
	}

	let result = {}
	for (let name in outputs) {
		result[name] = outputs[name].map(chan => {
			let out = new Float32Array(N)
			for (let i = 0; i < N; i++) { let w = weightSum[i]; out[i] = w > 1e-9 ? chan[i] / w : 0 }
			return out
		})
	}
	return result
}

// ------------------------------------------------------------------ sessions

async function resolveSessions(opts) {
	let model = opts.model
	if (model == null) throw new Error('neural-separate: opts.model is required (url | bytes | {target: url, ...} | {url, targets})')
	let loadOne = spec => opts.session ? opts.session(spec, opts) : neuralLoad(spec, { backend: opts.device })

	if (typeof model === 'string' || model instanceof Uint8Array) {
		return { sessions: { stem: await loadOne(model) }, targets: ['stem'], multi: false }
	}
	if (model.url != null && Array.isArray(model.targets)) {
		return { sessions: { __multi__: await loadOne(model.url) }, targets: model.targets.slice(), multi: true }
	}
	let names = Object.keys(model)
	if (!names.length) throw new Error('neural-separate: opts.model target map is empty')
	let sessions = {}
	for (let name of names) sessions[name] = await loadOne(model[name])
	return { sessions, targets: names, multi: false }
}

// ------------------------------------------------------------------ pipeline

async function separateSpectral(channelData, rate, opts, sessions, targets, multi, modelType) {
	let n = opts.n ?? 4096, hop = opts.hop ?? 1024
	let win = hann(n)
	let wienerOpts = { iterations: opts.wiener ?? 1, softmask: opts.softmask ?? false, eps: opts.eps, residual: false }

	return chunkedProcess(channelData, rate, opts, async (chunkChannels, chunkLen) => {
		let mixStft = chunkChannels.map(c => stft(c, { n, hop, window: win }))
		let C = mixStft.length, T = mixStft[0].re.length, bins = mixStft[0].bins

		let estimates = {}
		if (multi) {
			let session = sessions.__multi__
			let out = await session.run({ [feedName(session)]: packSpectral(mixStft, C, T, bins) })
			let outTensor = firstOutput(session, out)
			for (let i = 0; i < targets.length; i++) {
				let raw = unpackSpectral(outTensor.data, i * C * bins * T, C, T, bins)
				estimates[targets[i]] = modelType === 'mask' ? applyMask(raw, mixStft, C, T, bins) : raw
			}
		} else {
			for (let name of targets) {
				let session = sessions[name]
				let out = await session.run({ [feedName(session)]: packSpectral(mixStft, C, T, bins) })
				let outTensor = firstOutput(session, out)
				let raw = unpackSpectral(outTensor.data, 0, C, T, bins)
				estimates[name] = modelType === 'mask' ? applyMask(raw, mixStft, C, T, bins) : raw
			}
		}

		let complex = wienerFilter(mixStft, estimates, wienerOpts)
		let out = {}
		for (let name of targets) out[name] = complex[name].map(chStft => istft(chStft, { length: chunkLen }))
		return out
	})
}

async function separateWaveform(channelData, rate, opts, sessions, targets, multi) {
	return chunkedProcess(channelData, rate, opts, async (chunkChannels, chunkLen) => {
		let C = chunkChannels.length
		if (multi) {
			let session = sessions.__multi__
			let out = await session.run({ [feedName(session)]: packWaveform(chunkChannels, C, chunkLen) })
			let outTensor = firstOutput(session, out)
			let stacked = unpackWaveform(outTensor.data, targets.length, C, chunkLen)
			let result = {}
			for (let i = 0; i < targets.length; i++) result[targets[i]] = stacked[i]
			return result
		}
		let result = {}
		for (let name of targets) {
			let session = sessions[name]
			let out = await session.run({ [feedName(session)]: packWaveform(chunkChannels, C, chunkLen) })
			let outTensor = firstOutput(session, out)
			result[name] = unpackWaveform(outTensor.data, 1, C, chunkLen)[0] // [1,1,C,N] convention — single target still carries the S axis
		}
		return result
	})
}

// -------------------------------------------------------------------- input

function normalizeAudio(audio, opts) {
	let channelData, sampleRate
	if (Array.isArray(audio)) { channelData = audio; sampleRate = opts.sampleRate }
	else if (audio && audio.channelData) { channelData = audio.channelData; sampleRate = audio.sampleRate ?? opts.sampleRate }
	else throw new Error('neural-separate: audio must be Float32Array[] or { channelData, sampleRate }')
	if (!sampleRate) throw new Error('neural-separate: sampleRate is required (opts.sampleRate, or audio.sampleRate)')
	if (!channelData.length) throw new Error('neural-separate: audio has no channels')
	// mono → duplicated stereo, matching open-unmix-pytorch's own preprocess()
	// (openunmix/utils.py: "if we have mono, we duplicate it to get stereo")
	if (channelData.length === 1) channelData = [channelData[0], channelData[0]]
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
	if (opts.dtype && opts.dtype !== 'float32') throw new Error(`neural-separate: dtype '${opts.dtype}' not supported — only 'float32' tensor marshalling is implemented`)

	let { channelData, sampleRate: rate } = normalizeAudio(audio, opts)
	let targetRate = opts.targetRate ?? rate
	let modelType = opts.modelType ?? 'openunmix'

	let procData = targetRate !== rate ? channelData.map(c => resampleSinc(c, { from: rate, to: targetRate })) : channelData

	let { sessions, targets, multi } = await resolveSessions(opts)

	let stemsAtRate
	try {
		stemsAtRate = modelType === 'waveform'
			? await separateWaveform(procData, targetRate, opts, sessions, targets, multi)
			: await separateSpectral(procData, targetRate, opts, sessions, targets, multi, modelType)
	} finally {
		for (let name in sessions) sessions[name].free?.()
	}

	let N = channelData[0].length
	let stems = {}
	for (let name of targets) {
		let chans = stemsAtRate[name]
		stems[name] = chans.map(c => fitLength(targetRate !== rate ? resampleSinc(c, { from: targetRate, to: rate }) : c, N))
	}

	let residual = channelData.map((c, ci) => {
		let r = Float32Array.from(c)
		for (let name of targets) { let s = stems[name][ci]; for (let i = 0; i < N; i++) r[i] -= s[i] }
		return r
	})

	return { stems, sampleRate: rate, residual }
}
