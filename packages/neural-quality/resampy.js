// The resampler dnsmos_local.py ran with when 82f1b17e77 was committed (2022-05-24): librosa.resample(aud, input_fs,
// fs) on librosa 0.9.1, the release of the day, whose default res_type 'kaiser_best' is resampy's precomputed filter,
// from resampy 0.2.2, the one release between August 2019 and resampy 0.3.0 (2022-06-29), which changed the filter; then
// librosa's fix_length to ceil(n · target / orig). Translated from resampy 0.2.2 (ISC License, Copyright (c) 2016
// Brian McFee): core.py resample(), interpn.py resample_f(), filters.py sinc_window(); and from librosa 0.9.1 (ISC
// License, Copyright (c) 2013--2017, librosa development team) core/audio.py resample(), the fix_length step.
//
// Changes, 2026, audiojs: kaiser_best's table (data/kaiser_best.npz, 32,769 values) is not shipped but built by
// sinc_window from the parameters filters.py states (64 zero crossings, a Kaiser window of beta 14.769656459379492, a
// rolloff of 0.9475937167399596 of Nyquist) and the file's 2^9 values per crossing; so built, 27,650 of its values
// differ from the file's, by up to 1.0e-15 (13 ulps; I0 is its power series, not Cephes' Chebyshev expansion), and the
// 16 kHz signal by 2.8e-17 at most: 15 of 9,968,612 samples as float32, on 54 files at 8 to 96 kHz (measured once, by
// scripts outside the package). One channel; the output float64, as resampy's for float64 input.

const ZEROS = 64, NUM_TABLE = 2 ** 9, ROLLOFF = 0.9475937167399596, BETA = 14.769656459379492

// I0(x) = Σ ((x/2)^k / k!)², summed until a term no longer changes the sum
function i0(x) {
	let s = 1, t = 1, q = x * x / 4
	for (let k = 1; k < 500; k++) { t *= q / (k * k); let s1 = s + t; if (s1 === s) break; s = s1 }
	return s
}

// sinc_window(num_zeros=64, precision=9, window=kaiser(beta), rolloff): the filter's right wing,
// rolloff · sinc(rolloff · t) · kaiser(2n + 1, beta)[n:], t = linspace(0, 64, n + 1)
let table
function wing() {
	if (table) return table
	let n = NUM_TABLE * ZEROS, w = new Float64Array(n + 1), step = ZEROS / n, ib = i0(BETA)
	for (let k = 0; k <= n; k++) {
		let t = k === n ? ZEROS : k * step, x = Math.PI * (t === 0 ? 1e-20 : ROLLOFF * t)  // np.sinc: π · where(x == 0, 1e-20, x)
		let r = k / n, taper = i0(BETA * Math.sqrt(1 - r * r)) / ib  // scipy.signal.windows.kaiser: (n − α) / α
		w[k] = taper * (ROLLOFF * (Math.sin(x) / x))
	}
	return table = w
}

// the wing scaled by the rate's ratio where it falls (core.py: interp_win *= sample_ratio), and its differences
// (interp_delta[:-1] = np.diff(interp_win), the last 0), kept for the last two ratios
const scaled = new Map()
function filters(ratio) {
	let key = ratio < 1 ? ratio : 1, f = scaled.get(key)
	if (f) return f
	let w = wing()
	if (ratio < 1) w = w.map(v => v * ratio)
	let d = new Float64Array(w.length)
	for (let i = 0; i < w.length - 1; i++) d[i] = w[i + 1] - w[i]
	if (scaled.size >= 2) scaled.delete(scaled.keys().next().value)
	scaled.set(key, f = { w, d })
	return f
}

/**
 * x (one channel at rate `from`) at rate `to`, as librosa 0.9.1's resample(x, from, to) gives it with resampy 0.2.2's
 * kaiser_best: Float64Array of ceil(n · (to / from)) samples.
 */
export default function resampy(x, from, to) {
	let n = x.length, ratio = to / from
	let size = Math.ceil(n * ratio)                     // librosa: int(np.ceil(y.shape[-1] * ratio))
	let nOut = Math.trunc(n * ratio)                    // resampy: int(shape[axis] * sample_ratio)
	if (nOut < 1) throw new RangeError(`neural-quality: ${n} samples are too few to resample from ${from} to ${to} Hz`)
	let { w, d } = filters(ratio), y = new Float64Array(size)
	let scale = Math.min(1, ratio), step = Math.trunc(scale * NUM_TABLE), inc = 1 / ratio, nwin = w.length, time = 0
	for (let t = 0; t < Math.min(nOut, size); t++, time += inc) {  // the time register summed, as resample_f sums it
		let m = Math.trunc(time), frac = scale * (time - m)
		let index = frac * NUM_TABLE, offset = Math.trunc(index), eta = index - offset, acc = 0
		for (let i = 0, e = Math.min(m + 1, Math.floor((nwin - offset) / step)); i < e; i++) {
			let j = offset + i * step
			acc += (w[j] + eta * d[j]) * x[m - i]
		}
		frac = scale - frac
		index = frac * NUM_TABLE; offset = Math.trunc(index); eta = index - offset
		for (let k = 0, e = Math.min(n - m - 1, Math.floor((nwin - offset) / step)); k < e; k++) {
			let j = offset + k * step
			acc += (w[j] + eta * d[j]) * x[m + k + 1]
		}
		y[t] = acc
	}
	return y
}
