// A sound made again from its mel spectrogram by Vocos (Siuzdak, ICLR 2024, arXiv:2306.00814; charactr's
// vocos-mel-24khz, MIT): each STFT frame's magnitude and phase predicted afresh, so a vocoder's artifacts in the input
// (a TTS's phase smear, its frames ringing on) give way to Vocos's own. At 24 kHz: what is above 12 kHz is gone.
//
// The network runs through @audio/neural-runtime (ONNX, scripts/export.py: the backbone and the head's linear layer,
// the complex frames out); its features and its iSTFT are computed here, as Vocos computes them (torchaudio's
// MelSpectrogram and torch.istft), the same on every runtime.

import { fft, ifft } from 'fourier-transform'
import resample from '@audio/resample-sinc'
import { load, tensor, fetchModel } from '@audio/neural-runtime'

/** Vocos's rate, frame and hop, mel bands (charactr/vocos-mel-24khz config.yaml) */
export const SAMPLE_RATE = 24000, N = 1024, HOP = 256, MELS = 100

// The hosted copy: audiojs/vocos-mel-24khz at REVISION, checked against its SHA-256
export const REVISION = '4f5fad0af046227c42655d4b518081d42dd09015', SHA256 = '54c5cb663571c9ace439e20fb43bce0b8948a992ee88eb8f7676d6ac19f376c0'
const REPO = 'audiojs/vocos-mel-24khz', FILE = 'vocos.fp16.onnx'
export const MODEL = new URL(FILE, `https://huggingface.co/${REPO}/resolve/${REVISION}/`)

// Frames each run takes (opts.chunk), and the context on each side: the backbone sees 27 frames each way (a convolution
// of 7, then 8 depthwise convolutions of 7; its norms are per frame), so a run's frames past the context are the whole
// graph's
const CHUNK = 1024, CONTEXT = 32

const isNode = globalThis.process?.release?.name === 'node'  // Node, Bun, Deno: not a process polyfill (esm.sh's, a bundler's), whose versions.node is set
const isUrl = s => /^[a-z][a-z0-9+.-]+:/i.test(s)

const hann = Float64Array.from({ length: N }, (_, i) => 0.5 - 0.5 * Math.cos(2 * Math.PI * i / N))  // torch.hann_window: periodic

// torch 'reflect' padding's index: the edge sample not repeated
const reflect = (i, n) => { if (n === 1) return 0; let p = 2 * (n - 1); i = ((i % p) + p) % p; return i < n ? i : p - i }

// torchaudio.functional.melscale_fbanks(513, 0, 12000, 100, 24000, norm=None, mel_scale='htk'): per band, its weight
// on each FFT bin
const fbank = (() => {
	let bins = N / 2 + 1, mel = f => 2595 * Math.log10(1 + f / 700), hz = m => 700 * (10 ** (m / 2595) - 1)
	let top = mel(SAMPLE_RATE / 2), f = Array.from({ length: MELS + 2 }, (_, i) => hz(top * i / (MELS + 1)))
	return Array.from({ length: MELS }, (_, m) => Float64Array.from({ length: bins }, (_, k) => {
		let fk = k * SAMPLE_RATE / 2 / (bins - 1)
		return Math.max(0, Math.min((fk - f[m]) / (f[m + 1] - f[m]), (f[m + 2] - fk) / (f[m + 2] - f[m + 1])))
	}))
})()

/**
 * Vocos's features of `x` (mono, 24 kHz): the log of a 100-band HTK mel spectrogram of magnitudes, frames of 1024 every
 * 256, centered with reflect padding: { data: Float32Array(MELS × frames), band-major, frames }.
 */
export function mel(x) {
	let L = x.length, T = 1 + Math.floor(L / HOP), bins = N / 2 + 1, frame = new Float64Array(N), mag = new Float64Array(bins)
	let data = new Float32Array(MELS * T)
	for (let t = 0; t < T; t++) {
		let at = t * HOP - N / 2
		for (let i = 0; i < N; i++) frame[i] = (at + i >= 0 && at + i < L ? x[at + i] : L ? x[reflect(at + i, L)] : 0) * hann[i]
		let [re, im] = fft(frame)
		for (let k = 0; k < bins; k++) mag[k] = Math.hypot(re[k], im[k])
		for (let m = 0; m < MELS; m++) {
			let s = 0, w = fbank[m]
			for (let k = 0; k < bins; k++) if (w[k]) s += w[k] * mag[k]
			data[m * T + t] = Math.log(Math.max(s, 1e-7))
		}
	}
	return { data, frames: T }
}

/**
 * torch.istft of Vocos's frames (re, im: Float32Array(513 × frames), bin-major), Hann of 1024, hop 256, centered:
 * weighted overlap-add over the window's squared envelope, `length` samples (default 256 · (frames − 1), torch's).
 */
export function istft(re, im, frames, length = HOP * (frames - 1)) {
	let bins = N / 2 + 1, out = new Float64Array((frames - 1) * HOP + N), env = new Float64Array(out.length)
	let fr = new Float64Array(bins), fi = new Float64Array(bins)
	for (let t = 0; t < frames; t++) {
		for (let k = 0; k < bins; k++) { fr[k] = re[k * frames + t]; fi[k] = im[k * frames + t] }
		let y = ifft(fr, fi)
		for (let i = 0, at = t * HOP; i < N; i++) { out[at + i] += y[i] * hann[i]; env[at + i] += hann[i] * hann[i] }
	}
	let y = new Float32Array(length)
	for (let i = 0, j = N / 2; i < length && j < out.length; i++, j++) if (env[j] > 1e-11) y[i] = out[j] / env[j]
	return y
}

// The model's bytes: opts.weights (bytes, a URL or, in Node, a directory or file path holding vocos.fp16.onnx), else
// the hosted file, read through the neural cache, refused unless of its SHA-256
async function bytesOf(opts) {
	let w = opts.weights
	if (w instanceof Uint8Array || w instanceof ArrayBuffer) return new Uint8Array(w)
	if (w != null) {
		let href = String(w)
		if (!isUrl(href)) {
			if (!isNode) throw new Error(`neural-vocode: opts.weights must be a URL in the browser, got '${href}'`)
			href = (await import('node:url')).pathToFileURL(href).href
		}
		if (!href.endsWith('.onnx')) href = (href.endsWith('/') ? href : href + '/') + FILE
		return fetchModel(href, { progress: opts.progress, fetch: opts.fetch })
	}
	let store = isNode ? await cached() : null, bytes = store?.read()
	if (bytes && await sha256(bytes) === SHA256) return bytes
	try { bytes = await fetchModel(MODEL.href, { progress: opts.progress, fetch: opts.fetch, cache: !store }) }
	catch (e) { throw new Error(`neural-vocode: ${FILE} not read (${e.message}): opts.weights takes a URL holding a copy`) }
	let hex = await sha256(bytes)
	if (hex !== SHA256) throw new Error(`neural-vocode: ${MODEL} has SHA-256 ${hex}, expected ${SHA256}: a partial or altered download`)
	store?.write(bytes)
	return bytes
}

const subtle = async () => globalThis.crypto?.subtle ?? (await import('node:crypto')).webcrypto.subtle
const sha256 = async bytes => Array.from(new Uint8Array(await (await subtle()).digest('SHA-256', bytes)), b => b.toString(16).padStart(2, '0')).join('')

// <neural cache>/vocos/vocos.fp16.onnx: { read() → bytes or null, write(bytes) whole or not at all }
async function cached() {
	let [fs, path, os] = await Promise.all([import('node:fs'), import('node:path'), import('node:os')])
	let dest = path.join(process.env.AUDIO_NEURAL_CACHE || path.join(os.homedir(), '.cache', 'audiojs', 'neural'), 'vocos', FILE)
	return {
		read() { try { let b = fs.readFileSync(dest); return new Uint8Array(b.buffer, b.byteOffset, b.length) } catch { return null } },
		write(bytes) {
			let part = `${dest}.${process.pid}.${Math.random().toString(36).slice(2)}.part`
			fs.mkdirSync(path.dirname(dest), { recursive: true })
			fs.writeFileSync(part, bytes)
			fs.renameSync(part, dest)
		},
	}
}

// One session per weights and backend, made once
const sessions = new Map()
function session(opts) {
	let key = `${opts.weights instanceof Uint8Array || opts.weights instanceof ArrayBuffer ? 'bytes' : opts.weights ?? ''}:${opts.backend ?? 'auto'}`
	if (!sessions.has(key)) sessions.set(key, bytesOf(opts).then(b => load(b, { backend: opts.backend })).catch(e => { sessions.delete(key); throw e }))
	return sessions.get(key)
}

/**
 * The complex frames Vocos predicts from `mel` (as mel() gives it): { re, im }, each Float32Array(513 × frames),
 * bin-major; opts.chunk frames a run (1024), each with 32 more each side for context: the same frames as one run.
 */
export async function frames({ data, frames: T }, opts = {}) {
	let s = await session(opts), bins = N / 2 + 1, re = new Float32Array(bins * T), im = new Float32Array(bins * T), chunk = opts.chunk ?? CHUNK
	for (let at = 0; at < T; at += chunk) {
		let a = Math.max(0, at - CONTEXT), b = Math.min(T, at + chunk + CONTEXT), n = b - a, x = new Float32Array(MELS * n)
		for (let m = 0; m < MELS; m++) x.set(data.subarray(m * T + a, m * T + b), m * n)
		let out = await s.run({ mel: tensor(x, [1, MELS, n]) }), k0 = at - a, keep = Math.min(chunk, T - at)
		for (let k = 0; k < bins; k++) for (let t = 0; t < keep; t++) {
			re[k * T + at + t] = out.re.data[k * n + k0 + t]
			im[k * T + at + t] = out.im.data[k * n + k0 + t]
		}
	}
	return { re, im }
}

/**
 * `x` (mono samples at `sampleRate`) made again by Vocos: its mel spectrogram, Vocos's frames, their iSTFT, at
 * `sampleRate` again and as long. opts: { sampleRate = 24000, weights, backend, chunk, progress, fetch }.
 */
export default async function vocode(x, opts = {}) {
	let fs = opts.sampleRate ?? SAMPLE_RATE
	if (!(x instanceof Float32Array)) x = Float32Array.from(x)
	if (!x.length) return new Float32Array(0)
	let y = fs === SAMPLE_RATE ? x : resample(x, { from: fs, to: SAMPLE_RATE }), m = mel(y), { re, im } = await frames(m, opts)
	let z = istft(re, im, m.frames, y.length)
	if (fs === SAMPLE_RATE) return z
	z = resample(z, { from: SAMPLE_RATE, to: fs })
	if (z.length === x.length) return z
	let out = new Float32Array(x.length)
	out.set(z.subarray(0, x.length))
	return out
}
