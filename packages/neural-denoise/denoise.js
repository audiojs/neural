// Neural speech enhancement: RNNoise (pure JS, weights bundled) and DeepFilterNet3 (ONNX through
// @audio/neural-runtime, weights fetched from upstream and cached). Both run at 48 kHz on 10 ms hops;
// other rates are resampled in and out with @audio/resample-sinc. Offline results are aligned with the
// input: the models' delay is compensated. Music passes through untouched (guard.js) unless music: 'enhance'.

import resample from '@audio/resample-sinc'
import { model as parseRNNoise, create, FRAME, LIMIT as RNNOISE_LIMIT } from './rnnoise.js'
import { loadDeepFilter, enhance, level, bandEdge, MODEL, LEVEL, LIMIT as DFN_LIMIT, FLOOR } from './deepfilter.js'
import { guardNet, offline, online, gains, blend } from './guard.js'

export { create as rnnoise, FRAME } from './rnnoise.js'
export { MODEL } from './deepfilter.js'

const RATE = 48000, SMOOTH = 0.5
// the attenuation limit (dB) each model gets when opts.limit is not given; 0 is upstream's output
const LIMITS = { rnnoise: RNNOISE_LIMIT, deepfilternet3: DFN_LIMIT }

// weights() → the bundled RNNoise weights (upstream's weight blob, 3,544,320 bytes), for the worklet
export async function weights() {
	let url = new URL('./rnnoise.bin', import.meta.url)
	if (url.protocol === 'file:') return new Uint8Array((await import('node:fs')).readFileSync(url))
	let res = await fetch(url)
	if (!res.ok) throw new Error(`neural-denoise: can't fetch ${url}: ${res.status}`)
	return new Uint8Array(await res.arrayBuffer())
}

let bundled
// load(model?, opts?) → a handle denoise() reuses: 'rnnoise' (default) or 'deepfilternet3'
export async function load(model = 'rnnoise', opts = {}) {
	if (model === 'rnnoise') {
		let net = opts.weights ? parseRNNoise(opts.weights) : await (bundled ??= weights().then(parseRNNoise))
		return { model, net, latency: 2 * FRAME, free() {} }
	}
	if (model === 'deepfilternet3') {
		// ORT's arena keeps the largest chunk's activations: off, a 180 s file peaks at 563 MB instead of 665
		let net = await loadDeepFilter(opts.weights ?? MODEL, { ...opts, sessionOptions: { enableCpuMemArena: false, enableMemPattern: false, ...opts.sessionOptions } })
		return { model, net, latency: (net.params.n - net.params.hop) + net.params.hop * Math.max(net.params.convLa, net.params.dfLa), free: () => net.free() }
	}
	throw new Error(`neural-denoise: unknown model '${model}' (want 'rnnoise' or 'deepfilternet3')`)
}

// denoise(audio, opts) → the same shape back: Float32Array, Float32Array[] or { channelData, sampleRate }
export default async function denoise(audio, opts = {}) {
	let channels, rate = opts.sampleRate
	if (audio instanceof Float32Array) channels = [audio]
	else if (Array.isArray(audio)) channels = audio
	else if (audio?.channelData) { channels = audio.channelData; rate = audio.sampleRate ?? rate }
	else throw new TypeError('neural-denoise: audio must be a Float32Array, Float32Array[] or { channelData, sampleRate }')
	if (!(rate > 0)) throw new TypeError('neural-denoise: sampleRate is required (opts.sampleRate, or audio.sampleRate)')
	let music = opts.music ?? 'pass'
	if (music !== 'pass' && music !== 'enhance') throw new TypeError(`neural-denoise: music is 'pass' or 'enhance', not ${music}`)
	let handle = typeof opts.model === 'object' && opts.model ? opts.model : await load(opts.model, opts)
	try {
		let out = [], limit = opts.limit ?? LIMITS[handle.model], nn = music === 'pass' ? await guardNet() : null, dfn = handle.model !== 'rnnoise'
		for (let ch of channels) {
			let x = rate === RATE ? Float32Array.from(ch) : resample(Float32Array.from(ch), { from: rate, to: RATE }), y, pass
			// RNNoise decides as its stream does (guard.js online()), DeepFilterNet3 with the whole input in view
			if (!dfn) ({ y, pass } = rnnoiseOffline(x, handle.net, limit, nn))
			else {
				pass = nn && offline(x, nn)
				y = pass?.every(Boolean) ? x : await enhance(x, handle.net, { ...opts, limit: 0, gain: 10 ** ((LEVEL - level(x)) / 20), edge: bandEdge(ch, rate), voice: true })
			}
			if (rate !== RATE) y = fit(resample(y, { from: RATE, to: rate }), ch.length)
			if (pass) y = mix(y, ch, gains(pass, x.length, dfn), rate)
			out.push(dfn ? mixback(ch, y, { limit, floor: opts.floor, sampleRate: rate }) : y)
		}
		return audio instanceof Float32Array ? out[0] : Array.isArray(audio) ? out : { ...audio, channelData: out, sampleRate: rate }
	} finally {
		if (handle !== opts.model) handle.free()
	}
}

/**
 * mixback(x, y, { limit, floor, sampleRate }) → y + a·(x − y): the input x mixed back into a model's output y, the noise
 * the model took (x − y) back at gain a. a = 10^(−limit/20), so the noise drops by `limit` dB and room tone stays; where
 * the noise stands higher than `floor` dB under the voice, a falls until it lies there, so noise loud against the voice
 * is not merely turned down. Per 10 ms frame: the noise's mean power over the 0.5 s around it against the voice's
 * level, the louder half of y's 50 ms frames (level()); a interpolated between frames. Where x and y agree (music the
 * guard passed) the output is y bit for bit. limit 0: y itself, the model's whole removal; floor 0: a constant
 * 10^(−limit/20), as before 0.5. denoise() applies it to DeepFilterNet3; `audio`'s deepfilter op to the unlimited
 * output it keeps.
 */
export function mixback(x, y, { limit = DFN_LIMIT, floor = FLOOR, sampleRate = RATE } = {}) {
	let n = x.length, out = new Float32Array(n), cap = limit ? 10 ** (-Math.abs(limit) / 20) : 0
	if (!cap) { out.set(y.subarray(0, n)); return out }
	if (!floor) { for (let i = 0; i < n; i++) out[i] = y[i] + cap * (x[i] - y[i]); return out }
	// the noise's energy per frame, summed (c[k]: frames before k), and the frames' mean over the window around each
	let F = Math.max(1, Math.round(sampleRate / 100)), K = Math.ceil(n / F), c = new Float64Array(K + 1), g = new Float64Array(K), h = Math.round(50 * SMOOTH)
	for (let k = 0; k < K; k++) {
		let s = 0, e = Math.min(n, (k + 1) * F)
		for (let i = k * F; i < e; i++) s += (x[i] - y[i]) ** 2
		c[k + 1] = c[k] + s
	}
	let voice = 10 ** ((level(y, sampleRate) - Math.abs(floor)) / 10)
	for (let k = 0; k < K; k++) {
		let a = Math.max(0, k - h), b = Math.min(K, k + h + 1)
		g[k] = Math.min(cap, Math.sqrt(voice / Math.max((c[b] - c[a]) / (Math.min(n, b * F) - a * F), 1e-30)))
	}
	for (let i = 0; i < n; i++) {
		let t = i / F - .5, k = Math.min(K - 1, Math.max(0, Math.floor(t))), w = Math.min(1, Math.max(0, t - k))
		out[i] = y[i] + (k + 1 < K ? g[k] + w * (g[k + 1] - g[k]) : g[k]) * (x[i] - y[i])
	}
	return out
}

// the enhanced y where the gain g (at 48 kHz) is 1, the input x where it is 0: music passes untouched
function mix(y, x, g, rate) {
	if (rate !== RATE) g = fit(resample(g, { from: RATE, to: rate }), x.length)
	let out = new Float32Array(x.length)
	for (let i = 0; i < x.length; i++) out[i] = blend(g[i], y[i], x[i])
	return out
}

const fit = (y, n) => y.length === n ? y : (() => { let o = new Float32Array(n); o.set(y.subarray(0, n)); return o })()

// One channel through RNNoise at int16 scale, delay removed: out[i] lines up with x[i]. With the guard's network,
// also `pass` per 480 samples of the output, as the stream decides it: each output frame takes the decision made
// once the input frame two on (RNNoise's delay) is in, from the input so far
function rnnoiseOffline(x, net, limit, nn) {
	let st = create(net), delay = 2 * FRAME, L = x.length, n = Math.ceil((L + delay) / FRAME) * FRAME
	let pad = new Float32Array(n), y = new Float32Array(n), raw = new Float32Array(n), decide = nn && online(nn), on = []
	for (let i = 0; i < L; i++) pad[i] = x[i] * 32768
	raw.set(x)
	for (let i = 0; i < n; i += FRAME) {
		st.process(pad.subarray(i, i + FRAME), y.subarray(i, i + FRAME))
		if (decide) on.push(decide(raw.subarray(i, i + FRAME)))
	}
	let lim = limit ? 10 ** (-Math.abs(limit) / 20) : 0, out = new Float32Array(L)
	for (let i = 0; i < L; i++) out[i] = (y[i + delay] / 32768) * (1 - lim) + x[i] * lim
	return { y: out, pass: decide && Uint8Array.from(on.slice(2)) }
}
