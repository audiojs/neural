// What a sound is, in AudioSet's words: Google's YAMNet (tensorflow/models, research/audioset/yamnet, Apache-2.0), a
// MobileNetV1 over log-mel patches of 0.96 s every 0.48 s, a sigmoid score for each of 521 classes per patch.
//
// The network runs through @audio/neural-runtime (ONNX, scripts/export.py: patches in, scores out); its features are
// computed here as upstream's features.py computes them (pad_waveform, tf.signal.stft, linear_to_mel_weight_matrix,
// tf.signal.frame), the same on every runtime.

import { fft } from 'fourier-transform'
import resample from '@audio/resample-sinc'
import { load, tensor, fetchModel } from '@audio/neural-runtime'
import CLASSES from './classes.js'

export { CLASSES }

/** YAMNet's rate, and a patch's frames and mel bands (params.py) */
export const SAMPLE_RATE = 16000, FRAMES = 96, BANDS = 64

// params.py: a 25 ms periodic Hann window every 10 ms, its FFT the next power of 2; 64 mel bands, 125 to 7500 Hz;
// log(mel + 0.001). A patch's 96 frames span 15,600 samples, 0.975 s: pad_waveform's least length
const WIN = 400, STEP = 160, NFFT = 512, BINS = NFFT / 2 + 1, LOW = 125, HIGH = 7500, OFFSET = 0.001
const SPAN = (FRAMES - 1) * STEP + WIN, K = CLASSES.length, BATCH = 32

// The hosted copy: audiojs/yamnet at REVISION, checked against its SHA-256; in Node kept in the neural cache's
// yamnet/yamnet.fp16.onnx, where scripts/export.py writes it too, read when of this SHA-256
export const REVISION = '32997c946aaca3ebca95148c657838b28fe0d794', SHA256 = '0a1f3b050ebfdfa651e78fb12714c3c1fe750ceebd58529818a1ddddec515688'
const REPO = 'audiojs/yamnet', FILE = 'yamnet.fp16.onnx'
export const MODEL = REVISION ? new URL(FILE, `https://huggingface.co/${REPO}/resolve/${REVISION}/`) : null

const isNode = globalThis.process?.release?.name === 'node'  // Node, Bun, Deno: not a process polyfill (esm.sh's, a bundler's), whose versions.node is set
const isUrl = s => /^[a-z][a-z0-9+.-]+:/i.test(s)

const hann = Float64Array.from({ length: WIN }, (_, i) => 0.5 - 0.5 * Math.cos(2 * Math.PI * i / WIN))  // tf.signal.hann_window: periodic

// tf.signal.linear_to_mel_weight_matrix(64, 257, 16000, 125, 7500): HTK mels, each band a triangle in mels over the
// bins above DC: its first bin and weights
const fbank = (() => {
	let mel = f => 1127 * Math.log(1 + f / 700), lo = mel(LOW), hi = mel(HIGH), edge = i => lo + (hi - lo) * i / (BANDS + 1)
	return Array.from({ length: BANDS }, (_, m) => {
		let a = edge(m), c = edge(m + 1), b = edge(m + 2), w = []
		for (let k = 1; k < BINS; k++) { let z = mel(k * SAMPLE_RATE / NFFT); w[k] = Math.max(0, Math.min((z - a) / (c - a), (b - z) / (b - c))) }
		let k0 = w.findIndex(v => v > 0), k1 = w.findLastIndex(v => v > 0) + 1
		return { k0, w: Float64Array.from(w.slice(k0, k1)) }
	})
})()

// hop in seconds → whole 10 ms frames (params.py's patch_hop_seconds; features.py frames by round(hop / 10 ms))
function hopFrames(hop = 0.48) {
	let h = Math.round(hop * SAMPLE_RATE / STEP)
	if (!(h >= 1)) throw new RangeError(`neural-tag: opts.hop must be 0.01 s or more, got ${hop}`)
	return h
}

// pad_waveform: a sound shorter than a patch is one patch, zeros after it; a longer one ends on a whole hop, zeros after
const count = (L, h) => 1 + Math.ceil(Math.max(0, L - SPAN) / (h * STEP))

// log-mel patches i0 to i1 of x, every h frames: Float32Array((i1 − i0) · 96 · 64), frame-major. A frame a patch shares
// with the one before it (h < 96) is copied from it
function patchesOf(x, h, i0, i1) {
	let out = new Float32Array((i1 - i0) * FRAMES * BANDS), frame = new Float64Array(NFFT), mag = new Float64Array(BINS)
	let spec = [new Float64Array(BINS), new Float64Array(BINS)], [re, im] = spec
	for (let i = i0; i < i1; i++) for (let t = 0; t < FRAMES; t++) {
		let o = ((i - i0) * FRAMES + t) * BANDS
		if (i > i0 && t < FRAMES - h) { out.copyWithin(o, o - (FRAMES - h) * BANDS, o - (FRAMES - h) * BANDS + BANDS); continue }
		let at = (i * h + t) * STEP
		for (let n = 0; n < WIN; n++) frame[n] = at + n < x.length ? x[at + n] * hann[n] : 0
		fft(frame, spec)
		for (let k = 0; k < BINS; k++) mag[k] = Math.sqrt(re[k] * re[k] + im[k] * im[k])
		for (let m = 0; m < BANDS; m++) {
			let { k0, w } = fbank[m], s = 0
			for (let k = 0; k < w.length; k++) s += w[k] * mag[k0 + k]
			out[o + m] = Math.log(s + OFFSET)
		}
	}
	return out
}

/**
 * YAMNet's input from `x` (mono, 16 kHz): the log-mel patches features.py makes of it, a patch every opts.hop seconds
 * (0.48): { data: Float32Array(patches × 96 × 64), frame-major, patches }.
 */
export function features(x, opts = {}) {
	let h = hopFrames(opts.hop), n = count(x.length, h)
	return { data: patchesOf(x, h, 0, n), patches: n }
}

// channels (an array of them) mixed down to their mean, as upstream's inference.py does, then resampled to 16 kHz
function mono(x, sampleRate) {
	if (!(sampleRate > 0)) throw new RangeError(`neural-tag: sampleRate must be a positive number, got ${sampleRate}`)
	if (Array.isArray(x) && x.length && typeof x[0] !== 'number') {
		let n = x[0].length, y = new Float32Array(n)
		if (x.some(c => c.length !== n)) throw new Error('neural-tag: channels of different lengths')
		for (let i = 0; i < n; i++) { let s = 0; for (let c of x) s += c[i]; y[i] = s / x.length }
		x = y
	}
	else if (!(x instanceof Float32Array)) x = Float32Array.from(x)
	return sampleRate === SAMPLE_RATE ? x : resample(x, { from: sampleRate, to: SAMPLE_RATE })
}

// The model's bytes: opts.weights (bytes, a URL or, in Node, a directory or file path holding yamnet.fp16.onnx), else in
// Node the neural cache's copy when of its SHA-256, else the hosted file, refused unless of its SHA-256
async function bytesOf(opts) {
	let w = opts.weights
	if (w instanceof Uint8Array || w instanceof ArrayBuffer) return new Uint8Array(w)
	if (w != null) {
		let href = String(w)
		if (!isUrl(href)) {
			if (!isNode) throw new Error(`neural-tag: opts.weights must be a URL in the browser, got '${href}'`)
			href = (await import('node:url')).pathToFileURL(href).href
		}
		if (!href.endsWith('.onnx')) href = (href.endsWith('/') ? href : href + '/') + FILE
		return fetchModel(href, { progress: opts.progress, fetch: opts.fetch })
	}
	let store = isNode ? await cached() : null, bytes = store?.read()
	if (bytes && await sha256(bytes) === SHA256) return bytes
	if (!MODEL) throw new Error(`neural-tag: ${FILE} is not hosted yet (REVISION unset)${store ? ` and ${store.dest} holds no copy of it` : ''}: pass opts.weights, a URL${isNode ? ', a path' : ''} or the bytes of ${FILE} (scripts/export.py makes it)`)
	try { bytes = await fetchModel(MODEL.href, { progress: opts.progress, fetch: opts.fetch, cache: !store }) }
	catch (e) { throw new Error(`neural-tag: ${FILE} not read (${e.message}): opts.weights takes a URL holding a copy`) }
	let hex = await sha256(bytes)
	if (hex !== SHA256) throw new Error(`neural-tag: ${MODEL} has SHA-256 ${hex}, expected ${SHA256}: a partial or altered download`)
	store?.write(bytes)
	return bytes
}

const subtle = async () => globalThis.crypto?.subtle ?? (await import('node:crypto')).webcrypto.subtle
const sha256 = async bytes => Array.from(new Uint8Array(await (await subtle()).digest('SHA-256', bytes)), b => b.toString(16).padStart(2, '0')).join('')

// <neural cache>/yamnet/yamnet.fp16.onnx: { dest, read() → bytes or null, write(bytes) whole or not at all }
async function cached() {
	let [fs, path, os] = await Promise.all([import('node:fs'), import('node:path'), import('node:os')])
	let dest = path.join(process.env.AUDIO_NEURAL_CACHE || path.join(os.homedir(), '.cache', 'audiojs', 'neural'), 'yamnet', FILE)
	return {
		dest,
		read() { try { let b = fs.readFileSync(dest); return new Uint8Array(b.buffer, b.byteOffset, b.length) } catch { return null } },
		write(bytes) {
			let part = `${dest}.${process.pid}.${Math.random().toString(36).slice(2)}.part`
			fs.mkdirSync(path.dirname(dest), { recursive: true })
			fs.writeFileSync(part, bytes)
			fs.renameSync(part, dest)
		},
	}
}

// One session per weights (a URL or path by its text, bytes by their object), backend and session options, made once
const named = new Map(), given = new WeakMap()
function session(opts) {
	let w = opts.weights, bytes = w instanceof Uint8Array || w instanceof ArrayBuffer
	let key = `${bytes ? '' : w ?? ''}:${opts.backend ?? 'auto'}:${JSON.stringify(opts.sessionOptions ?? {})}`
	let map = bytes ? given.get(w) ?? given.set(w, new Map()).get(w) : named
	if (!map.has(key)) map.set(key, bytesOf(opts).then(b => load(b, { backend: opts.backend, sessionOptions: opts.sessionOptions })).catch(e => { map.delete(key); throw e }))
	return map.get(key)
}

/**
 * Every patch's 521 scores of `x` (mono samples, or an array of channels, at `sampleRate`): { scores: a Float32Array(521)
 * per patch, hop: seconds between patches, times: each patch's start in seconds }. A patch hears 0.975 s from its start.
 * opts: { hop = 0.48, weights, backend, sessionOptions, progress, fetch }.
 */
export async function scores(x, sampleRate = SAMPLE_RATE, opts = {}) {
	let h = hopFrames(opts.hop), y = mono(x, sampleRate), n = count(y.length, h), s = await session(opts)
	let input = s.inputs[0]?.name ?? 'patches', output = s.outputs.some(o => o.name === 'scores') ? 'scores' : s.outputs[0]?.name
	let all = new Float32Array(n * K)
	for (let i = 0; i < n; i += BATCH) {
		let j = Math.min(n, i + BATCH), out = await s.run({ [input]: tensor(patchesOf(y, h, i, j), [j - i, FRAMES, BANDS]) })
		all.set(out[output].data, i * K)
	}
	let hop = h * STEP / SAMPLE_RATE
	return { scores: Array.from({ length: n }, (_, i) => all.subarray(i * K, (i + 1) * K)), hop, times: Array.from({ length: n }, (_, i) => i * hop) }
}

/**
 * The classes of `x` ranked by their mean score over its patches, as upstream's inference.py ranks them:
 * [{ name, id, index, score }], best first, opts.top of them (5; Infinity for all 521). Options as scores().
 */
export async function tag(x, sampleRate = SAMPLE_RATE, opts = {}) {
	let top = opts.top ?? 5
	if (!(top >= 0)) throw new RangeError(`neural-tag: opts.top must be 0 or more, got ${top}`)
	let { scores: s } = await scores(x, sampleRate, opts), mean = new Float64Array(K)
	for (let row of s) for (let k = 0; k < K; k++) mean[k] += row[k] / s.length
	return Array.from({ length: K }, (_, i) => i).sort((a, b) => mean[b] - mean[a] || a - b).slice(0, top)
		.map(index => ({ name: CLASSES[index].name, id: CLASSES[index].id, index, score: mean[index] }))
}
