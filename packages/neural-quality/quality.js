// How listeners would rate a recording of speech, with no clean reference: DNSMOS P.835 (C. K. A. Reddy, V. Gopal,
// R. Cutler, "DNSMOS P.835: A Non-Intrusive Perceptual Objective Speech Quality Metric to Evaluate Noise Suppressors",
// ICASSP 2022), ITU-T P.835's three scales from 1 to 5: the speech (SIG), the background (BAK), the whole (OVRL).
//
// Microsoft's network (DNSMOS/DNSMOS/sig_bak_ovr.onnx, CC BY 4.0) runs through @audio/neural-runtime. Around it, the
// method of DNSMOS/dnsmos_local.py (MIT License, Copyright (c) Microsoft Corporation), both at microsoft/DNS-Challenge
// commit 82f1b17e7776a43eee395d0f45bae8abb700ad00, translated to JavaScript: ComputeScore.__call__ (lines 61-106) and
// get_polyfit_val's non-personalized branch (lines 50-59). The network takes the waveform (input_1, [N, 144160]
// float32) and computes its spectrogram itself, so nothing of it is computed here.
//
// Changes, 2026, audiojs: librosa.resample translated as the script ran it on the day of its commit (resampy.js:
// librosa 0.9.1, resampy 0.2.2's kaiser_best; librosa 0.10 and later refuse its positional call). Channels mixed to
// their mean (the script passes soundfile's 2-D array of a multichannel file on, and away from 16 kHz librosa resamples
// it along the channels: above 32 kHz resampy refuses it; at 32 kHz and below the network refuses one of 144,160
// frames or more, and a shorter one np.append flattens into its interleaved values, which the script scores without an
// error). An empty input refused (the script's doubling loop never ends on it), and a sample that is NaN or infinite.
// opts.windows: 'all' scores the windows the script's float arithmetic drops (below). The personalized variant
// (pDNSMOS) is left out.

import { load as loadModel, tensor, fetchModel } from '@audio/neural-runtime'
import resampy from './resampy.js'

/** The network's rate and window: SAMPLING_RATE and INPUT_LENGTH, dnsmos_local.py lines 19-20 */
export const SAMPLE_RATE = 16000, INPUT_LENGTH = 9.01
/** A window in samples: int(INPUT_LENGTH * fs), line 69 */
export const WINDOW = Math.trunc(INPUT_LENGTH * SAMPLE_RATE)

// get_polyfit_val, non-personalized, lines 51-53: np.poly1d coefficients, highest power first
const P_SIG = [-0.08397278, 1.22083953, 0.0052439]
const P_BAK = [-0.13166888, 1.60915514, -0.39604546]
const P_OVR = [-0.06766283, 1.11546468, 0.04602535]

// The pinned file: its commit, its SHA-256 and size as fetched from there (1,157,965 bytes); in Node kept at the neural
// cache's dnsmos/sig_bak_ovr.onnx, the path audio's bench/speech.py reads it from too
export const REVISION = '82f1b17e7776a43eee395d0f45bae8abb700ad00', SHA256 = '269fbebdb513aa23cddfbb593542ecc540284a91849ac50516870e1ac78f6edd'
const FILE = 'sig_bak_ovr.onnx'
export const MODEL = new URL(`https://raw.githubusercontent.com/microsoft/DNS-Challenge/${REVISION}/DNSMOS/DNSMOS/${FILE}`)

const isNode = globalThis.process?.release?.name === 'node'  // Node, Bun, Deno: not a process polyfill (esm.sh's, a bundler's)
const isUrl = s => /^[a-z][a-z0-9+.-]+:/i.test(s)
const horner = (p, x) => (p[0] * x + p[1]) * x + p[2]

/**
 * The polynomials that map the network's raw outputs to the P.835 scale (get_polyfit_val, non-personalized):
 * { sig, bak, ovrl } in, { sig, bak, ovrl } out. numpy evaluates them in float64 on the float32 outputs, as here.
 */
export function polyfit({ sig, bak, ovrl }) {
	return { sig: horner(P_SIG, sig), bak: horner(P_BAK, bak), ovrl: horner(P_OVR, ovrl) }
}

/**
 * The windows of a 16 kHz signal of `n` samples (n > 0), as ComputeScore lays them: the signal doubled until it holds a
 * window (np.append(audio, audio), lines 70-71); int(floor(length / fs) − 9.01) + 1 hops of 1 s (line 73); each window
 * audio[int(i · fs) : int((i + 9.01) · fs)], skipped when shorter than WINDOW (lines 83-85). That bound is a float
 * product: for i = 7 to 23, 119 to 122 and longer runs past 4.5 hours it truncates a sample short, so the script drops
 * those windows, and of a clip from 17 s to under 34 s it scores the first 15.01 s alone. opts.windows: 'upstream'
 * (default) drops them as the script does; 'all' keeps every hop.
 * → { length: the doubled length, starts: each window's first sample }
 */
export function windows(n, { windows: mode = 'upstream' } = {}) {
	if (!(n > 0)) throw new RangeError(`neural-quality: a signal of ${n} samples has no windows`)
	if (mode !== 'upstream' && mode !== 'all') throw new TypeError(`neural-quality: opts.windows is 'upstream' or 'all', not ${mode}`)
	let length = n
	while (length < WINDOW) length *= 2
	let hops = Math.trunc(Math.floor(length / SAMPLE_RATE) - INPUT_LENGTH) + 1, starts = []
	for (let i = 0; i < hops; i++) {
		let a = Math.trunc(i * SAMPLE_RATE), b = Math.min(length, Math.trunc((i + INPUT_LENGTH) * SAMPLE_RATE))
		if (mode === 'upstream' && b - a < WINDOW) continue
		starts.push(a)
	}
	return { length, starts }
}

/**
 * What the windows are cut from: `audio` (a Float32Array, an array of channels, or { channelData, sampleRate }) mixed
 * to the mean of its channels and resampled to 16 kHz as librosa 0.9.1 and resampy 0.2.2 resample it: Float32Array,
 * the float32 the script casts each window to.
 */
export function prepare(audio, opts = {}) {
	let channels, rate = opts.sampleRate
	if (ArrayBuffer.isView(audio)) channels = [audio]
	else if (Array.isArray(audio)) channels = typeof audio[0] === 'number' ? [audio] : audio
	else if (audio?.channelData) { channels = audio.channelData; rate = audio.sampleRate ?? rate }
	else throw new TypeError('neural-quality: audio must be a Float32Array, Float32Array[] or { channelData, sampleRate }')
	if (!(rate > 0 && rate < Infinity)) throw new TypeError(`neural-quality: sampleRate is required, a positive number (opts.sampleRate, or audio.sampleRate), got ${rate}`)
	if (!channels.length) throw new TypeError('neural-quality: audio has no channels')
	let n = channels[0].length, x = new Float32Array(n)
	if (channels.some(c => c.length !== n)) throw new RangeError('neural-quality: channels of different lengths')
	if (!n) throw new RangeError('neural-quality: no samples to score')
	for (let c of channels) for (let i = 0; i < n; i++) x[i] += channels.length > 1 ? c[i] / channels.length : c[i]
	for (let i = 0; i < n; i++) if (!Number.isFinite(x[i])) throw new RangeError(`neural-quality: sample ${i} is ${x[i]}: refused, the network would score it NaN`)
	return rate === SAMPLE_RATE ? x : Float32Array.from(resampy(x, rate, SAMPLE_RATE))
}

// The model's bytes: opts.weights (bytes, a URL or, in Node, a file or directory path holding sig_bak_ovr.onnx), else
// in Node the neural cache's copy when of SHA256, else the pinned file, refused unless of SHA256
async function bytesOf(opts) {
	let w = opts.weights
	if (w instanceof Uint8Array || w instanceof ArrayBuffer) return new Uint8Array(w)
	if (w != null) {
		let href = String(w)
		if (!isUrl(href)) {
			if (!isNode) throw new Error(`neural-quality: opts.weights must be a URL in the browser, got '${href}'`)
			href = (await import('node:url')).pathToFileURL(href).href
		}
		if (!href.endsWith('.onnx')) href = (href.endsWith('/') ? href : href + '/') + FILE
		return fetchModel(href, { progress: opts.progress, fetch: opts.fetch })
	}
	let store = isNode ? await cached() : null, bytes = store?.read()
	if (bytes && await sha256(bytes) === SHA256) return bytes
	try { bytes = await fetchModel(MODEL.href, { progress: opts.progress, fetch: opts.fetch, cache: !store }) }
	catch (e) { throw new Error(`neural-quality: ${FILE} not read (${e.message}): opts.weights takes a URL${isNode ? ', a path' : ''} or the bytes of a copy`) }
	let hex = await sha256(bytes)
	if (hex !== SHA256) throw new Error(`neural-quality: ${MODEL} has SHA-256 ${hex}, expected ${SHA256}: a partial or altered download`)
	store?.write(bytes)
	return bytes
}

const subtle = async () => globalThis.crypto?.subtle ?? (await import('node:crypto')).webcrypto.subtle
const sha256 = async bytes => Array.from(new Uint8Array(await (await subtle()).digest('SHA-256', bytes)), b => b.toString(16).padStart(2, '0')).join('')

// <neural cache>/dnsmos/sig_bak_ovr.onnx: { dest, read() → bytes or null, write(bytes) whole or not at all }
async function cached() {
	let [fs, path, os] = await Promise.all([import('node:fs'), import('node:path'), import('node:os')])
	let dest = path.join(process.env.AUDIO_NEURAL_CACHE || path.join(os.homedir(), '.cache', 'audiojs', 'neural'), 'dnsmos', FILE)
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

/**
 * The network, loaded once, for dnsmos() to reuse through opts.model: { session, free() }.
 * opts: { weights, backend, sessionOptions, progress, fetch }.
 */
export async function load(opts = {}) {
	let session = await loadModel(await bytesOf(opts), { backend: opts.backend, sessionOptions: opts.sessionOptions })
	return { session, free: () => session.free() }
}

/**
 * SIG, BAK and OVRL of `audio` (a Float32Array, an array of channels or { channelData, sampleRate }), each the mean over
 * its windows of the polynomial of the network's output, as dnsmos_local.py gives them: { sig, bak, ovrl, segments },
 * segments the windows scored, { time: its start in seconds (in the doubled signal, for a clip under 9.01 s), sig,
 * bak, ovrl, raw: the network's own three outputs }. opts: { sampleRate, model: a load() handle, windows, and load()'s
 * options when no handle is given }.
 */
export default async function dnsmos(audio, opts = {}) {
	let x = prepare(audio, opts), { starts } = windows(x.length, opts)
	let handle = opts.model ?? await load(opts)
	try {
		let s = handle.session, input = s.inputs[0]?.name ?? 'input_1', output = s.outputs[0]?.name, seg = new Float32Array(WINDOW), segments = []
		for (let a of starts) {
			// the doubled signal, sample a + j of it x[(a + j) mod n]
			for (let j = 0, k = a % x.length; j < WINDOW; j++, k = k + 1 === x.length ? 0 : k + 1) seg[j] = x[k]
			let [sig, bak, ovrl] = (await s.run({ [input]: tensor(seg, [1, WINDOW]) }))[output].data
			segments.push({ time: a / SAMPLE_RATE, ...polyfit({ sig, bak, ovrl }), raw: [sig, bak, ovrl] })
		}
		let mean = k => segments.reduce((m, r) => m + r[k], 0) / segments.length
		return { sig: mean('sig'), bak: mean('bak'), ovrl: mean('ovrl'), segments }
	} finally {
		if (handle !== opts.model) handle.free()
	}
}

export { dnsmos }
