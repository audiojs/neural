// Neural speech enhancement: RNNoise (pure JS, weights bundled) and DeepFilterNet3 (ONNX through
// @audio/neural-runtime, weights fetched from upstream and cached). Both run at 48 kHz on 10 ms hops;
// other rates are resampled in and out with @audio/resample-sinc. Offline results are aligned with the
// input: the models' delay is compensated.

import resample from '@audio/resample-sinc'
import { model as parseRNNoise, create, FRAME, LIMIT as RNNOISE_LIMIT } from './rnnoise.js'
import { loadDeepFilter, enhance, MODEL, LIMIT as DFN_LIMIT } from './deepfilter.js'

export { create as rnnoise, FRAME } from './rnnoise.js'
export { MODEL } from './deepfilter.js'

const RATE = 48000
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
	let handle = typeof opts.model === 'object' && opts.model ? opts.model : await load(opts.model, opts)
	try {
		let out = [], limit = opts.limit ?? LIMITS[handle.model]
		for (let ch of channels) {
			let x = rate === RATE ? Float32Array.from(ch) : resample(Float32Array.from(ch), { from: rate, to: RATE })
			let y = handle.model === 'rnnoise' ? rnnoiseOffline(x, handle.net, limit) : await enhance(x, handle.net, { ...opts, limit })
			out.push(rate === RATE ? y : fit(resample(y, { from: RATE, to: rate }), ch.length))
		}
		return audio instanceof Float32Array ? out[0] : Array.isArray(audio) ? out : { ...audio, channelData: out, sampleRate: rate }
	} finally {
		if (handle !== opts.model) handle.free()
	}
}

const fit = (y, n) => y.length === n ? y : (() => { let o = new Float32Array(n); o.set(y.subarray(0, n)); return o })()

// One channel through RNNoise at int16 scale, delay removed: out[i] lines up with x[i]
function rnnoiseOffline(x, net, limit) {
	let st = create(net), delay = 2 * FRAME, L = x.length, n = Math.ceil((L + delay) / FRAME) * FRAME
	let pad = new Float32Array(n), y = new Float32Array(n)
	for (let i = 0; i < L; i++) pad[i] = x[i] * 32768
	for (let i = 0; i < n; i += FRAME) st.process(pad.subarray(i, i + FRAME), y.subarray(i, i + FRAME))
	let lim = limit ? 10 ** (-Math.abs(limit) / 20) : 0, out = new Float32Array(L)
	for (let i = 0; i < L; i++) out[i] = (y[i + delay] / 32768) * (1 - lim) + x[i] * lim
	return out
}
