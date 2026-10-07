// Fixtures: fixtures/identity-mask.onnx (fixtures/make.py, onnx only) and fixtures/wiener.json
// (fixtures/make-wiener.py: open-unmix-pytorch's own wiener()).
// Real-weight tests run when the exported weights and their reference separations exist under
// $AUDIO_NEURAL_CACHE or ~/.cache/audiojs/neural (scripts/export-*.py --verify writes both).
import test, { ok, is, rejects } from 'tst'
import { existsSync, readFileSync, mkdtempSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import separate, { stft, istft, wienerFilter, models } from './separate.js'

// ---------------------------------------------------------------- utilities

function seededRand(seed) {
	let s = seed >>> 0
	return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296 }
}

// Energy ratio 10·log10(Σref² / Σ(est−ref)²): the SDR of Vincent, Gribonval, Févotte (IEEE TASLP
// 2006) without its distortion-filter search, adequate where estimate and reference are aligned
// by construction.
function snr(est, ref, from = 0, to = Math.min(est.length, ref.length)) {
	let num = 0, den = 0
	for (let i = from; i < to; i++) { num += ref[i] * ref[i]; let d = est[i] - ref[i]; den += d * d }
	return 10 * Math.log10(num / (den + 1e-300))
}

function magOf(chStft) {
	return chStft.re.map((re, t) => {
		let im = chStft.im[t], a = new Float64Array(chStft.bins)
		for (let f = 0; f < a.length; f++) a[f] = Math.hypot(re[f], im[f])
		return a
	})
}

// Mock session with @audio/neural-runtime's Session shape.
function mockSession(run, ioNames = { in: 'input', out: 'output' }, dims) {
	return {
		inputs: [{ name: ioNames.in, dims }],
		outputs: [{ name: ioNames.out }],
		async run(feeds) { return { [ioNames.out]: await run(feeds[ioNames.in]) } },
		free() {},
	}
}

// modelType 'mask' oracle: 1 inside [lo, hi] Hz, 0 outside; exact for spectrally disjoint sources.
function bandMaskSession(lo, hi, n, rate) {
	return mockSession(async input => {
		let [, C, F, T] = input.dims
		let data = new Float32Array(C * F * T)
		for (let f = 0; f < F; f++) {
			let m = (f * rate / n >= lo && f * rate / n <= hi) ? 1 : 0
			for (let c = 0; c < C; c++) data.fill(m, c * F * T + f * T, c * F * T + (f + 1) * T)
		}
		return { data, dims: input.dims, type: 'float32' }
	}, { in: 'mag', out: 'mask' })
}

function sine(freqs, amp, fs, N, vibratoHz = 0, vibratoDepth = 0) {
	let x = new Float32Array(N)
	for (let i = 0; i < N; i++) {
		let t = i / fs, vib = vibratoHz ? 1 + vibratoDepth * Math.sin(2 * Math.PI * vibratoHz * t) : 1, s = 0
		for (let f of freqs) s += Math.sin(2 * Math.PI * f * vib * t)
		x[i] = amp * s / freqs.length
	}
	return x
}

// ---------------------------------------------------------- 1. STFT/iSTFT

test('stft/istft — identity round trip, SNR > 120 dB (random stereo, 3 s)', () => {
	let fs = 44100, N = Math.round(fs * 3), rand = seededRand(42)
	for (let ch = 0; ch < 2; ch++) {
		let x = new Float32Array(N)
		for (let i = 0; i < N; i++) x[i] = (rand() * 2 - 1) * 0.5
		let y = istft(stft(x, { n: 4096, hop: 1024 }), { length: N })
		is(y.length, N)
		let db = snr(y, x)
		ok(db > 120, `channel ${ch}: SNR ${db.toFixed(1)} dB`)
	}
})

test('stft — frame count matches torch.stft(center=True) formula: 1 + floor(N/hop)', () => {
	for (let durMs of [50, 700, 1000, 3333, 30000]) {
		let N = Math.round(44100 * durMs / 1000)
		is(stft(new Float32Array(N), { n: 4096, hop: 1024 }).re.length, 1 + Math.floor(N / 1024), `N=${N}`)
	}
})

// ------------------------------------------------------- 2. Oracle separation

// The true magnitude of a known stereo target (modelType 'openunmix'), one chunk only.
function trueMagnitudeSession(targetChannels, n, hop, perturb) {
	let mags = targetChannels.map(ch => magOf(stft(ch, { n, hop })))
	return mockSession(async input => {
		let [, C, F, T] = input.dims
		let data = new Float32Array(C * F * T)
		for (let c = 0; c < C; c++) for (let t = 0; t < T; t++) for (let f = 0; f < F; f++) {
			let v = mags[c][t][f]
			data[c * F * T + f * T + t] = perturb ? perturb(v) : v
		}
		return { data, dims: input.dims, type: 'float32' }
	}, { in: 'mag', out: 'mag' })
}

// Genuinely stereo bass/vocal: identical channels make every spatial covariance rank 1, a
// degenerate case for multichannel Wiener EM (README, Wiener filter).
function stereoSources(fs, N) {
	let bassL = sine([60, 90, 120, 150, 180], 0.5, fs, N), bassR = sine([60, 90, 120, 150, 180], 0.45, fs, N)
	let vocalL = sine([300, 500, 800, 1200, 1800, 2500], 0.4, fs, N, 5, 0.003)
	let vocalR = sine([300, 500, 800, 1200, 1800, 2500], 0.4, fs, N, 5.2, 0.0025)
	let mixL = new Float32Array(N), mixR = new Float32Array(N)
	for (let i = 0; i < N; i++) { mixL[i] = bassL[i] + vocalL[i]; mixR[i] = bassR[i] + vocalR[i] }
	return { bass: [bassL, bassR], vocal: [vocalL, vocalR], mix: [mixL, mixR] }
}

test('separate — oracle (true stem magnitude), disjoint bass/vocal bands: SDR > 15 dB (wiener:0); wiener:1 does not regress (≥ −0.1 dB)', async () => {
	let fs = 44100, N = Math.round(fs * 4)
	let { bass, vocal, mix } = stereoSources(fs, N)
	let session = async spec => trueMagnitudeSession(spec === 'b' ? bass : vocal, 4096, 1024)
	let opts = { sampleRate: fs, modelType: 'openunmix', model: { bass: 'b', vocal: 'v' }, session }
	let r0 = await separate(mix, { ...opts, wiener: 0 }), r1 = await separate(mix, { ...opts, wiener: 1 })
	for (let [name, ref] of [['bass', bass], ['vocal', vocal]]) {
		let a = snr(r0.stems[name][0], ref[0]), b = snr(r1.stems[name][0], ref[0])
		ok(a > 15, `wiener:0 ${name} ${a.toFixed(1)} dB`)
		ok(b >= a - 0.1, `wiener:1 ${name} ${b.toFixed(1)} vs wiener:0 ${a.toFixed(1)} dB`)
	}
})

test('separate — noisy magnitude estimates (×(1+0.3·noise)): wiener:1 beats wiener:0 by ≥ 1 dB', async () => {
	let fs = 44100, N = Math.round(fs * 4)
	let { bass, vocal, mix } = stereoSources(fs, N)
	let rand = seededRand(99), perturb = v => Math.max(0, v * (1 + 0.3 * (rand() * 2 - 1)))
	let session = async spec => trueMagnitudeSession(spec === 'b' ? bass : vocal, 4096, 1024, perturb)
	let opts = { sampleRate: fs, modelType: 'openunmix', model: { bass: 'b', vocal: 'v' }, session }
	let r0 = await separate(mix, { ...opts, wiener: 0 }), r1 = await separate(mix, { ...opts, wiener: 1 })
	for (let [name, ref] of [['bass', bass], ['vocal', vocal]]) {
		let a = snr(r0.stems[name][0], ref[0]), b = snr(r1.stems[name][0], ref[0])
		ok(b >= a + 1, `wiener:1 ${b.toFixed(1)} vs wiener:0 ${a.toFixed(1)} dB (${name})`)
	}
})

test('separate: one target runs EM against the residual (open-unmix residual=True): oracle bass SDR > 15 dB', async () => {
	let fs = 44100, N = Math.round(fs * 4)
	let { bass, vocal, mix } = stereoSources(fs, N)
	let session = async spec => trueMagnitudeSession(spec === 'b' ? bass : vocal, 4096, 1024)
	let r = await separate(mix, { sampleRate: fs, modelType: 'openunmix', model: { bass: 'b', vocal: 'v' }, targets: ['bass'], session, wiener: 1 })
	is(Object.keys(r.stems), ['bass'])
	let db = snr(r.stems.bass[0], bass[0])
	ok(db > 15, `bass ${db.toFixed(1)} dB`)
})

// --------------------------------------------------------- 3. wienerFilter

// Upstream's atan2 adds π as 2·asin(torch.tensor(1.0)), a float32 constant (8.7e-8 below π), in
// the left half-plane; the phase-based initial estimate carries that, the ratio mask does not.
test('wienerFilter: matches open-unmix-pytorch wiener() (fixtures/wiener.json, float64): softmask < 1e-9, mixture phase < 1e-6 of max |y|', () => {
	let fx = JSON.parse(readFileSync(new URL('./fixtures/wiener.json', import.meta.url), 'utf8'))
	// fixture layout: mix[channel][frame][bin][re/im], mags[source][channel][frame][bin]
	let bins = fx.mix[0][0].length
	let mixStft = fx.mix.map(ch => ({ re: ch.map(fr => Float64Array.from(fr, v => v[0])), im: ch.map(fr => Float64Array.from(fr, v => v[1])), n: (bins - 1) * 2, hop: 1, bins, center: true }))
	let estimates = Object.fromEntries(fx.mags.map((src, j) => [`s${j}`, src.map(ch => ch.map(fr => Float64Array.from(fr)))]))
	for (let [name, { opts, y }] of Object.entries(fx.cases)) {
		let got = Object.values(wienerFilter(mixStft, estimates, opts)), err = 0, scale = 0
		is(got.length, y.length, `${name}: source count`)
		y.forEach((src, j) => src.forEach((ch, c) => ch.forEach((fr, t) => fr.forEach(([re, im], f) => {
			scale = Math.max(scale, Math.abs(re), Math.abs(im))
			err = Math.max(err, Math.abs(got[j][c].re[t][f] - re), Math.abs(got[j][c].im[t][f] - im))
		}))))
		let tol = opts.softmask ? 1e-9 : 1e-6
		ok(err < tol * scale, `${name}: max |diff| ${err.toExponential(2)}, max |y| ${scale.toFixed(2)}`)
	}
})

test('wienerFilter — softmask ratio mask sums exactly to the mixture (iterations:0)', () => {
	let fs = 44100, N = fs
	let bass = sine([80, 160], 0.3, fs, N), vocal = sine([500, 1200], 0.15, fs, N)
	let mix = bass.map((v, i) => v + vocal[i])
	let mixStft = [stft(mix)]
	let out = wienerFilter(mixStft, { bass: [magOf(stft(bass))], vocal: [magOf(stft(vocal))] }, { iterations: 0, softmask: true })
	let maxErr = 0
	mixStft[0].re.forEach((re, t) => { for (let f = 0; f < re.length; f++) {
		maxErr = Math.max(maxErr, Math.abs(out.bass[0].re[t][f] + out.vocal[0].re[t][f] - re[f]), Math.abs(out.bass[0].im[t][f] + out.vocal[0].im[t][f] - mixStft[0].im[t][f]))
	} })
	ok(maxErr < 1e-4, `max |Σ stems − mix| = ${maxErr}`)
})

test('wienerFilter — residual:true makes the sum exact (iterations:0, any softmask)', () => {
	let fs = 44100, N = fs
	let bass = sine([80, 160], 0.3, fs, N), vocal = sine([500, 1200], 0.15, fs, N)
	let mix = bass.map((v, i) => v + vocal[i])
	let mixStft = [stft(mix)], rand = seededRand(3)
	let est = x => [magOf(stft(x)).map(row => row.map(v => v * (1 + 0.5 * rand())))]
	let out = wienerFilter(mixStft, { bass: est(bass), vocal: est(vocal) }, { iterations: 0, residual: true })
	ok('residual' in out, 'residual target present')
	let maxErr = 0
	mixStft[0].re.forEach((re, t) => { for (let f = 0; f < re.length; f++) {
		let sre = out.bass[0].re[t][f] + out.vocal[0].re[t][f] + out.residual[0].re[t][f]
		let sim = out.bass[0].im[t][f] + out.vocal[0].im[t][f] + out.residual[0].im[t][f]
		maxErr = Math.max(maxErr, Math.abs(sre - re[f]), Math.abs(sim - mixStft[0].im[t][f]))
	} })
	ok(maxErr < 1e-9, `max |Σ stems + residual − mix| = ${maxErr}`)
})

test('wienerFilter — eps stability: all-silent input and estimates produce finite output (no NaN/Inf)', () => {
	let mixStft = [stft(new Float32Array(8192))], zero = magOf(mixStft[0])
	for (let iterations of [0, 1, 2]) {
		let out = wienerFilter(mixStft, { a: [zero], b: [zero] }, { iterations })
		let finite = ['a', 'b'].every(name => out[name][0].re.every((re, t) => re.every((v, f) => Number.isFinite(v) && Number.isFinite(out[name][0].im[t][f]))))
		ok(finite, `iterations=${iterations}: all finite`)
	}
})

test('wienerFilter: mono and three-channel mixtures take the general inverse: oracle SDR > 15 dB', () => {
	let fs = 44100, N = fs * 2
	let bass = sine([60, 90, 120], 0.5, fs, N), vocal = sine([500, 800, 1200], 0.3, fs, N)
	for (let C of [1, 3]) {
		let gains = [1, 0.8, 0.6].slice(0, C)
		let mixStft = gains.map((g, c) => stft(bass.map((v, i) => g * v + (1 - 0.2 * c) * vocal[i])))
		let est = (x, gain) => gains.map((g, c) => magOf(stft(x.map(v => gain(g, c) * v))))
		let out = wienerFilter(mixStft, { bass: est(bass, g => g), vocal: est(vocal, (g, c) => 1 - 0.2 * c) }, { iterations: 1 })
		let db = snr(istft(out.bass[0], { length: N }), bass)
		ok(db > 15, `C=${C}: bass ${db.toFixed(1)} dB`)
	}
})

// ------------------------------------------------------------------ 4. Chunking

test('separate — chunked (30s/2s overlap) matches single-chunk (1000s) within 60 dB in overlap regions', async () => {
	let fs = 44100, N = Math.round(fs * 70)
	let mix = sine([60, 90, 120, 150, 180], 0.5, fs, N), vocal = sine([300, 500, 800, 1200, 1800, 2500], 0.4, fs, N)
	for (let i = 0; i < N; i++) mix[i] += vocal[i]
	let session = async spec => spec === 'b' ? bandMaskSession(0, 250, 4096, fs) : bandMaskSession(250, fs / 2, 4096, fs)
	let opts = { sampleRate: fs, modelType: 'mask', model: { bass: 'b', vocal: 'v' }, session, wiener: 1 }
	let size = 30 * fs, step = size - 2 * fs, starts = []
	for (let s = 0; s < N; s += step) { starts.push(s); if (s + size >= N) break }
	ok(starts.length >= 2, `sanity: ${starts.length} chunks for 70s/30s/2s`)
	let r1 = await separate([mix, mix], { ...opts, chunk: 30, overlap: 2 })
	let r2 = await separate([mix, mix], { ...opts, chunk: 1000, overlap: 2 })
	for (let name of ['bass', 'vocal']) for (let i = 1; i < starts.length; i++) {
		let db = snr(r1.stems[name][0], r2.stems[name][0], starts[i], Math.min(N, starts[i] + 2 * fs))
		ok(db > 60, `${name} overlap region ${i}: ${db.toFixed(1)} dB`)
	}
})

// -------------------------------------------------------------- 5. Waveform model

test('separate — waveform modelType (Demucs v2-class): mock returns true stems, SDR > 40 dB', async () => {
	let fs = 44100, N = fs * 3
	let bass = sine([90], 0.4, fs, N), vocal = sine([600], 0.3, fs, N)
	let mix = bass.map((v, i) => v + vocal[i])
	let truth = { bass, vocal }
	let session = mockSession(async input => {
		let [, C, len] = input.dims, names = ['bass', 'vocal'], data = new Float32Array(names.length * C * len)
		names.forEach((name, s) => { for (let c = 0; c < C; c++) data.set(truth[name].subarray(0, len), (s * C + c) * len) })
		return { data, dims: [1, names.length, C, len], type: 'float32' }
	}, { in: 'mix', out: 'stems' })
	let result = await separate([mix, mix], { sampleRate: fs, modelType: 'waveform', model: { url: 'fake', targets: ['bass', 'vocal'] }, session: async () => session })
	for (let name of ['bass', 'vocal']) { let db = snr(result.stems[name][0], truth[name]); ok(db > 40, `${name} SDR ${db.toFixed(1)} dB`) }
})

// ---------------------------------------------------------------- 6. Hybrid model

// demucs.onnx contract mock: segment L; `branch` passes the mixture through as source 0
// (stems_spec = mix_spec, or stems_wave = mix), the other sources zero. HTDemucs's framing
// drops the first and last two STFT frames and the iSTFT pads them back as zeros, so its own
// spectral round trip loses about 1.5k samples at each segment edge (demucs _spec/_ispec in torch:
// max error 0.19 there, 3e-7 inside); the triangular overlap-add weighs those edges down inside
// the file, and the file's own ends keep them.
function hybridSession(L, branch, S = 4) {
	return {
		inputs: [{ name: 'mix', dims: [1, 2, L] }, { name: 'mix_spec' }],
		outputs: [{ name: 'stems_spec' }, { name: 'stems_wave' }],
		async run(feeds) {
			let spec = feeds.mix_spec.data, wave = feeds.mix.data
			let zs = new Float32Array(S * spec.length), xt = new Float32Array(S * wave.length)
			if (branch === 'spec') zs.set(spec); else xt.set(wave)
			let [, C2, F, T] = feeds.mix_spec.dims
			return { stems_spec: { data: zs, dims: [1, S, C2, F, T] }, stems_wave: { data: xt, dims: [1, S, 2, L] } }
		},
		free() {},
	}
}

for (let branch of ['spec', 'wave']) test(`separate: hybrid modelType, ${branch} branch passing the mix through: reconstructs it > 60 dB over overlapping segments`, async () => {
	// 4 s segments every 3 s over 12 s (the edge loss weighs less the longer the segment; htdemucs's
	// is 7.8 s); tones that stay far below Nyquist, whose bin HTDemucs drops (sine()'s vibrato
	// scales phase, so its frequency sweep grows with time: here under 10 kHz by 12 s)
	let fs = 44100, N = fs * 12, L = fs * 4, edge = branch === 'spec' ? 2048 : 0
	let L0 = sine([220, 440, 1100, 5000], 0.4, fs, N, 3, 0.002), R0 = sine([330, 660, 2500, 4000], 0.3, fs, N, 4, 0.002)
	let chunks = []
	let r = await separate([L0, R0], {
		sampleRate: fs, modelType: 'hybrid', model: { url: 'fake', targets: ['a', 'b', 'c', 'd'] },
		session: async () => hybridSession(L, branch), progress: p => chunks.push(p),
	})
	is(chunks.length, Math.ceil(N / Math.floor(0.75 * L)), 'segments every floor(0.75 L), as demucs apply_model')
	for (let [c, x] of [[0, L0], [1, R0]]) { let db = snr(r.stems.a[c], x, edge, N - edge); ok(db > 60, `ch${c}: ${db.toFixed(1)} dB`) }
	// demucs.api denormalizes every source by the input's mean: an all-zero source comes back as that mean
	let mean = 0
	for (let i = 0; i < N; i++) mean += (L0[i] + R0[i]) / 2 / N
	let off = Math.max(...['b', 'c', 'd'].flatMap(s => r.stems[s].map(ch => ch.reduce((m, v) => Math.max(m, Math.abs(v - mean)), 0))))
	ok(off < 1e-6, `other sources at the mean: max |x − mean| ${off.toExponential(1)}`)
})

// ---------------------------------------------------------------- 6b. Complex-spectrogram model

// SCNet's contract mock: frames T, each source's spectrogram a fixed share of the mixture's (shares summing to 1), so
// the stems are the input scaled, and add back to it. Segments of L every L/4, linear fades, the input reflected out
// at both ends: what the overlap-add and the STFT around the graph lose shows in the sum.
function complexSession(T, shares) {
	return {
		inputs: [{ name: 'mix_spec', dims: [1, 4, 2049, T] }], outputs: [{ name: 'stems_spec' }],
		async run(feeds) {
			let x = feeds.mix_spec.data, n = x.length, z = new Float32Array(shares.length * n)
			shares.forEach((w, s) => { for (let i = 0; i < n; i++) z[s * n + i] = w * x[i] })
			return { stems_spec: { data: z, dims: [1, 4 * shares.length, 2049, T] } }
		},
		free() {},
	}
}

for (let window of ['ones', 'hann']) test(`separate: complex modelType (${window} window): stems are the mixture's shares, > 100 dB over segments and fades`, async () => {
	// 4 s segments (T = 173 frames of hop 1024), every 1 s over 13 s: longer than two segments less a step, its ends
	// reflected out; the last segment past the end
	let fs = 44100, T = 173, L = 172 * 1024 - 300, N = fs * 13 + 77, shares = [0.1, 0.2, 0.3, 0.4], chunks = []
	let L0 = sine([220, 440, 1100, 5000], 0.4, fs, N, 3, 0.002), R0 = sine([330, 660, 2500, 4000], 0.3, fs, N, 4, 0.002)
	let r = await separate([L0, R0], {
		sampleRate: fs, modelType: 'complex', model: { url: 'fake', targets: ['a', 'b', 'c', 'd'] }, segment: L, window, normalized: true,
		session: async () => complexSession(T, shares), progress: p => chunks.push(p),
	})
	let step = Math.floor(L / 4), M = N + 2 * (L - step)
	is(chunks.length, Math.ceil((M - L) / step) + 1, 'segments every L/4 until one reaches the end of the reflected input')
	// the input's mean (its DC) goes to no source
	let mean = 0
	for (let i = 0; i < N; i++) mean += (L0[i] + R0[i]) / 2 / N
	;['a', 'b', 'c', 'd'].forEach((t, s) => [L0, R0].forEach((x, c) => {
		let db = snr(r.stems[t][c], x.map(v => shares[s] * (v - mean))); ok(db > 100, `${t} ch${c}: ${db.toFixed(1)} dB`)
	}))
	// a short input: one segment, its rest reflected
	chunks = []
	let short = await separate([L0.subarray(0, fs * 3), R0.subarray(0, fs * 3)], {
		sampleRate: fs, modelType: 'complex', model: { url: 'fake', targets: ['a', 'b', 'c', 'd'] }, segment: L, window,
		session: async () => complexSession(T, shares), progress: p => chunks.push(p),
	})
	is(chunks.length, 1, 'one segment')
	let x3 = [L0, R0].map(x => x.subarray(0, fs * 3)), m3 = 0
	for (let i = 0; i < fs * 3; i++) m3 += (x3[0][i] + x3[1][i]) / 2 / (fs * 3)
	let db = snr(short.stems.d[0], x3[0].map(v => 0.4 * (v - m3))); ok(db > 100, `short: ${db.toFixed(1)} dB`)
})

// ------------------------------------------- 7. Mono duplication + resampling

test('separate — mono input duplicates to stereo output', async () => {
	let fs = 44100, N = fs
	let session = mockSession(async input => ({ data: Float32Array.from(input.data), dims: input.dims, type: 'float32' }))
	let result = await separate([sine([440], 0.3, fs, N)], { sampleRate: fs, model: 'x', modelType: 'openunmix', wiener: 0, session: async () => session })
	is(result.stems.stem.length, 2, 'stereo output')
	is(result.stems.stem[0].length, N)
})

test('separate: empty input returns empty stems, no inference', async () => {
	let runs = 0, session = mockSession(async input => (runs++, { data: input.data, dims: input.dims, type: 'float32' }))
	for (let modelType of ['openunmix', 'hybrid', 'waveform']) {
		let r = await separate([new Float32Array(0), new Float32Array(0)], { sampleRate: 44100, model: { url: 'x', targets: ['a', 'b'] }, modelType, session: async () => session })
		is(Object.keys(r.stems), ['a', 'b'], modelType)
		is(r.stems.a.map(ch => ch.length), [0, 0], `${modelType}: empty channels`)
	}
	is(runs, 0, 'no model run')
})

test('separate — targetRate resampling round trip (48k in, 44.1k model): output stays at input rate, duration ±1 ms', async () => {
	let fs = 48000, N = fs * 2, x = sine([440], 0.3, fs, N)
	let session = mockSession(async input => ({ data: Float32Array.from(input.data), dims: input.dims, type: 'float32' }))
	let result = await separate([x, x], { sampleRate: fs, targetRate: 44100, model: 'x', modelType: 'openunmix', wiener: 0, session: async () => session })
	is(result.sampleRate, fs, 'output reported at input rate')
	ok(Math.abs(result.stems.stem[0].length - N) / fs * 1000 <= 1, 'duration')
	let db = snr(result.stems.stem[0], x)
	ok(db > 30, `round-trip SDR ${db.toFixed(1)} dB`)
})

// ------------------------------------------------- 8. Real ONNX (no weights)

const IDENTITY = new URL('./fixtures/identity-mask.onnx', import.meta.url)

;(existsSync(IDENTITY) ? test : test.skip)('separate — real @audio/neural-runtime session (identity-mask.onnx): returns mixture as the single stem, SDR > 60 dB', async () => {
	let fs = 44100, N = fs, rand = seededRand(11)
	let L = new Float32Array(N), R = new Float32Array(N)
	for (let i = 0; i < N; i++) { L[i] = 0.3 * Math.sin(2 * Math.PI * 220 * i / fs) + 0.05 * (rand() * 2 - 1); R[i] = 0.3 * Math.sin(2 * Math.PI * 330 * i / fs) + 0.05 * (rand() * 2 - 1) }
	let result = await separate([L, R], { sampleRate: fs, model: readFileSync(IDENTITY), modelType: 'openunmix', wiener: 0 })
	is(Object.keys(result.stems).length, 1)
	let db = snr(result.stems.stem[0], L)
	ok(db > 60, `SDR ${db.toFixed(1)} dB`)
})

// ------------------------------------------------------------- 9. Presets

test('models: presets name their contract', () => {
	is(models.umxhq.modelType, 'openunmix')
	is(models.umxhq.targets, ['vocals', 'drums', 'bass', 'other'])
	is(models.htdemucs.modelType, 'hybrid')
	is(models.htdemucs.targets, ['drums', 'bass', 'other', 'vocals'])
	is(models['scnet-large'].modelType, 'complex')
	is(models['scnet-large'].targets, ['drums', 'bass', 'other', 'vocals'])
	is(models.scnet.modelType, 'complex')
})

test('separate: preset without its weights: names the missing file and the export script', async () => {
	let empty = mkdtempSync(path.join(os.tmpdir(), 'neural-separate-'))
	let x = new Float32Array(4410)
	await rejects(() => separate([x, x], { sampleRate: 44100, model: 'umxhq', targets: ['vocals'], weights: empty }), /umxhq weights not found: .*umxhq\/vocals\.onnx.*export-openunmix\.py/, 'umxhq')
	await rejects(() => separate([x, x], { sampleRate: 44100, model: 'htdemucs', weights: empty }), /htdemucs weights not found: .*htdemucs\/htdemucs\.onnx.*export-htdemucs\.py/, 'htdemucs')
	await rejects(() => separate([x, x], { sampleRate: 44100, model: 'scnet-large', weights: empty }), /scnet-large weights not found: .*scnet-large\/scnet-large\.onnx.*export-scnet\.py/, 'scnet-large')
	await rejects(() => separate([x, x], { sampleRate: 44100, model: 'umxhq', targets: ['piano'], weights: empty }), /umxhq has no target 'piano'/, 'unknown target')
})

// ------------------------------------------- 10. Real weights vs the Python reference

const CACHE = process.env.AUDIO_NEURAL_CACHE || path.join(os.homedir(), '.cache', 'audiojs', 'neural')

// scripts/reference.py's mix and the original implementation's stems, when exported
function reference(name) {
	let dir = path.join(CACHE, name), p = models[name]
	let files = [path.join(dir, 'test.f32'), ...p.targets.map(t => path.join(dir, `test.${t}.f32`))]
	let graphs = p.perTarget ? p.targets.map(t => `${t}.onnx`) : [`${name}.onnx`]
	if (![...files, ...graphs.map(g => path.join(dir, g))].every(f => existsSync(f))) return null
	let planar = f => { let b = readFileSync(f), a = new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4), n = a.length / 2; return [a.subarray(0, n), a.subarray(n)] }
	return { mix: planar(files[0]), stems: Object.fromEntries(p.targets.map((t, i) => [t, planar(files[i + 1])])) }
}

// umxhq: open-unmix-pytorch's Separator (niter=1, 300-frame Wiener windows) run in float64, which
// this port matches to 112-134 dB; upstream's float32 run differs from its own float64 run by 71-93 dB.
// htdemucs, htdemucs_ft: demucs.apply.apply_model (split, overlap 0.25, shifts 0), matched to
// 78-89 dB; ONNX Runtime and PyTorch differ by ~1e-4 relative inside the transformer
// (export-htdemucs.py --verify).
// scnet-large, scnet: SCNet.forward on the segments scripts/reference.py's chunked() cuts (separate.js's), 20 s,
// matched to 114-134 dB (export-scnet.py reduces its GroupNorm statistics axis by axis: as exported, 52-55 dB)
for (let [name, minDb] of [['umxhq', 100], ['htdemucs', 70], ['htdemucs_ft', 70], ['scnet-large', 100], ['scnet', 100]]) {
	let ref = reference(name)
	;(ref ? test : test.skip)(`separate: ${name} matches the Python reference on scripts/reference.py's mix: SNR > ${minDb} dB per stem`, async () => {
		let { stems } = await separate(ref.mix, { sampleRate: 44100, model: name })
		for (let t in ref.stems) for (let c of [0, 1]) {
			let db = snr(stems[t][c], ref.stems[t][c])
			ok(db > minDb, `${t} ch${c}: ${db.toFixed(1)} dB`)
		}
	}, { timeout: 600_000 }) // htdemucs_ft: four graphs, two segments each, on a CPU
}

// ------------------------------------------------------------------- Speed

// CPU time of this process: on a shared machine wall time measures the scheduler as much as the
// code (the same run takes 8 to 20 s of wall time at a load average of 50).
test('speed — mock pipeline (STFT + Wiener + iSTFT), 60 s stereo < 4 s of CPU', async () => {
	let fs = 44100, N = fs * 60, x = sine([90, 300], 0.3, fs, N)
	let session = async spec => spec === 'b' ? bandMaskSession(0, 250, 4096, fs) : bandMaskSession(250, fs / 2, 4096, fs)
	let c0 = process.cpuUsage(), t0 = performance.now()
	await separate([x, x], { sampleRate: fs, modelType: 'mask', model: { bass: 'b', vocal: 'v' }, session, wiener: 1, chunk: 30, overlap: 2 })
	let { user, system } = process.cpuUsage(c0), cpu = (user + system) / 1e6
	console.log(`  (60 s stereo, 2 targets, wiener:1: ${cpu.toFixed(2)} s CPU, ${((performance.now() - t0) / 1000).toFixed(2)} s wall)`)
	ok(cpu < 4, `${cpu.toFixed(2)} s CPU`)
})

// --------------------------------------------------------------- input errors

test('separate — rejects on missing sampleRate / model', async () => {
	await rejects(() => separate([new Float32Array(4)], {}), /sampleRate/, 'missing sampleRate')
	await rejects(() => separate([new Float32Array(4)], { sampleRate: 44100 }), /model/, 'missing model')
})
