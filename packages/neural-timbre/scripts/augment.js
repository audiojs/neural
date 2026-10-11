// Augmentation for training views and benchmark queries, all in audiojs: chorus, EQ, reverb,
// another voice or drums underneath, noise, compression, gain and a low-bitrate codec, applied in
// that order to a mono 44.1 kHz note. Every choice is drawn from a seeded generator, so a view or a
// benchmark query is reproducible from its seed.
import chorus from '@audio/effect-chorus'
import eq from '@audio/eq-parametric'
import freeverb from '@audio/reverb-freeverb'
import dattorro from '@audio/reverb-dattorro'
import compressor from '@audio/dynamics-compressor'
import encodeMp3 from '@audio/encode-mp3'
import decodeMp3 from '@audio/decode-mp3'
import encodeOpus from '@audio/encode-opus'
import decodeOpus from '@audio/decode-opus'
import { membrane, metal, noiseDrum } from '@audio/synth-drum'
import resample from '@audio/resample-sinc'

export const FS = 44100

// mulberry32 (Tommy Ettinger): 32-bit state, full period, adequate for choosing augmentations
export function rng(seed) {
	let a = seed >>> 0
	let next = () => {
		a = (a + 0x6d2b79f5) >>> 0
		let t = Math.imul(a ^ (a >>> 15), 1 | a)
		t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
		return ((t ^ (t >>> 14)) >>> 0) / 4294967296
	}
	next.uniform = (lo, hi) => lo + (hi - lo) * next()
	next.int = (lo, hi) => lo + Math.floor(next() * (hi - lo + 1)) // inclusive
	next.pick = list => list[Math.floor(next() * list.length)]
	next.log = (lo, hi) => lo * (hi / lo) ** next()
	return next
}

// A training chain: a third of views stay clean, the rest take each stage by its own chance.
// `other` draws the note under a mix: { voice, note, velocity, hold } of another voice.
export function chain(r, other) {
	if (r() < 1 / 3) return {}
	let c = {}
	if (r() < 0.25) c.chorus = { rate: r.uniform(0.2, 2), depth: r.uniform(0.1, 0.6), delay: r.uniform(0.008, 0.03), voices: r.int(2, 4) }
	if (r() < 0.45) c.eq = eqBands(r)
	if (r() < 0.35) c.reverb = r() < 0.75
		? { kind: 'freeverb', room: r.uniform(0.3, 0.95), damp: r.uniform(0.1, 0.7), mix: r.uniform(0.1, 0.5) }
		: { kind: 'dattorro', decay: r.uniform(0.3, 0.9), damping: r.uniform(0.1, 0.7), mix: r.uniform(0.1, 0.4) }
	if (r() < 0.25) c.mix = r() < 0.7 ? { ...other(r), snr: r.uniform(0, 20) } : { drums: r.int(0, 2 ** 31), snr: r.uniform(0, 20) }
	if (r() < 0.25) c.noise = { color: r.pick(['white', 'pink']), snr: r.uniform(5, 35), seed: r.int(0, 2 ** 31) }
	if (r() < 0.25) c.compressor = { threshold: r.uniform(-36, -10), ratio: r.uniform(2, 10), attack: r.uniform(1, 30), release: r.uniform(50, 300) }
	c.gain = r.uniform(-24, 0)
	if (r() < 0.15) c.codec = r() < 0.7 ? { kind: 'mp3', bitrate: r.pick([32, 48, 64]) } : { kind: 'opus', bitrate: r.pick([12, 16, 24]) }
	return c
}

const eqBands = r => [
	{ type: 'lowshelf', fc: r.uniform(80, 300), gain: r.uniform(-9, 9) },
	{ fc: r.log(300, 5000), Q: r.uniform(0.5, 2), gain: r.uniform(-9, 9) },
	{ type: 'highshelf', fc: r.uniform(3000, 8000), gain: r.uniform(-12, 6) },
]

// Benchmark conditions, one stage each at a fixed setting, plus the training chain drawn from the
// query's own seed. Frozen with the benchmark: changing one changes every figure that cites it.
export const CONDITIONS = {
	clean: () => ({}),
	reverb: () => ({ reverb: { kind: 'freeverb', room: 0.8, damp: 0.4, mix: 0.35 } }),
	eq: () => ({ eq: [{ type: 'lowshelf', fc: 150, gain: 6 }, { fc: 1500, Q: 1, gain: -6 }, { type: 'highshelf', fc: 5000, gain: -9 }] }),
	compressor: () => ({ compressor: { threshold: -30, ratio: 8, attack: 5, release: 100 } }),
	chorus: () => ({ chorus: { rate: 0.8, depth: 0.4, delay: 0.015, voices: 3 } }),
	gain: () => ({ gain: -30 }),
	mp3: () => ({ codec: { kind: 'mp3', bitrate: 32 } }),
	opus: () => ({ codec: { kind: 'opus', bitrate: 12 } }),
	noise20: r => ({ noise: { color: 'pink', snr: 20, seed: r.int(0, 2 ** 31) } }),
	noise10: r => ({ noise: { color: 'pink', snr: 10, seed: r.int(0, 2 ** 31) } }),
	mix10: (r, other) => ({ mix: { ...other(r), snr: 10 } }),
	mix0: (r, other) => ({ mix: { ...other(r), snr: 0 } }),
	drums10: r => ({ mix: { drums: r.int(0, 2 ** 31), snr: 10 } }),
	chain: (r, other) => { let c = {}; while (!Object.keys(c).length || (Object.keys(c).length === 1 && 'gain' in c)) c = chain(r, other); return c },
}

const rms = x => { let s = 0; for (let i = 0; i < x.length; i++) s += x[i] * x[i]; return Math.sqrt(s / Math.max(1, x.length)) }
const db2lin = db => 10 ** (db / 20)

// Mix `y` into `x` (in place) at `snr` dB below x's RMS over x's length.
function addAt(x, y, snr) {
	let gx = rms(x), gy = rms(y)
	if (!(gy > 0) || !(gx > 0)) return x
	let k = gx / gy * db2lin(-snr)
	for (let i = 0; i < x.length && i < y.length; i++) x[i] += k * y[i]
	return x
}

function noise({ color, seed }, n) {
	let r = rng(seed), out = new Float32Array(n)
	// pink: Paul Kellet's refined filter of white noise (musicdsp.org #76), -3 dB/octave within 0.05 dB above 9 Hz at 44.1 kHz
	let b0 = 0, b1 = 0, b2 = 0, b3 = 0, b4 = 0, b5 = 0, b6 = 0
	for (let i = 0; i < n; i++) {
		let w = r() * 2 - 1
		if (color === 'white') { out[i] = w; continue }
		b0 = 0.99886 * b0 + w * 0.0555179; b1 = 0.99332 * b1 + w * 0.0750759; b2 = 0.969 * b2 + w * 0.153852
		b3 = 0.8665 * b3 + w * 0.3104856; b4 = 0.55 * b4 + w * 0.5329522; b5 = -0.7616 * b5 - w * 0.016898
		out[i] = b0 + b1 + b2 + b3 + b4 + b5 + b6 + w * 0.5362; b6 = w * 0.115926
	}
	return out
}

// A bar of drums at 120 BPM from a seed: kick, snare or hat and cymbal hits on eighth notes.
function drums(seed, n) {
	let r = rng(seed), out = new Float32Array(n), step = Math.round(FS * 0.25)
	for (let at = Math.round(r() * step); at < n; at += step) {
		let hit = r.pick(['kick', 'kick', 'snare', 'hat', 'cymbal', 'rest'])
		if (hit === 'rest') continue
		let s = hit === 'kick' ? membrane(r.uniform(45, 70), { duration: 0.4, fs: FS })
			: hit === 'cymbal' ? metal(r.uniform(180, 260), { duration: 0.6, fs: FS, amp: 0.3 })
			: noiseDrum({ duration: hit === 'hat' ? 0.06 : 0.2, fs: FS, seed: r.int(1, 1e6), amp: hit === 'hat' ? 0.3 : 0.7 })
		for (let i = 0; i < s.length && at + i < n; i++) out[at + i] += s[i]
	}
	return out
}

const concat = parts => { let out = new Uint8Array(parts.reduce((n, p) => n + p.length, 0)), o = 0; for (let p of parts) { out.set(p, o); o += p.length } return out }

// Halfband lowpass for 2x rate changes: 63-tap Blackman-windowed sinc cut at a quarter of the
// rate (Oppenheim & Schafer, window method), every other tap but the centre zero.
const HALF = (() => {
	let n = 63, m = 31, h = new Float64Array(n)
	for (let i = 0; i < n; i++) {
		let k = i - m, w = 0.42 - 0.5 * Math.cos(2 * Math.PI * i / (n - 1)) + 0.08 * Math.cos(4 * Math.PI * i / (n - 1))
		h[i] = (k === 0 ? 0.5 : Math.sin(Math.PI * k / 2) / (Math.PI * k)) * w
	}
	return h
})()
function decimate2(x) {
	let out = new Float32Array(x.length >> 1), m = HALF.length >> 1
	for (let o = 0; o < out.length; o++) {
		let c = 2 * o, s = HALF[m] * x[c]
		for (let k = 1; k <= m; k += 2) s += HALF[m + k] * ((x[c - k] ?? 0) + (x[c + k] ?? 0))
		out[o] = s
	}
	return out
}
function interpolate2(y, n) {
	let out = new Float32Array(n), m = HALF.length >> 1
	for (let i = 0; i < n; i++) {
		if (i % 2 === 0) { out[i] = y[i >> 1] ?? 0; continue } // the centre tap is 0.5, doubled by the zero-stuffing gain
		let s = 0
		for (let k = 1; k <= m; k += 2) s += HALF[m + k] * ((y[(i - k) >> 1] ?? 0) + (y[(i + k) >> 1] ?? 0))
		out[i] = 2 * s
	}
	return out
}

// MP3 runs as MPEG-2 layer III at 22.05 kHz, the rate LAME picks itself for these bitrates at
// 44.1 kHz input; Opus codes at 48 kHz and comes back through the resampler.
async function codec(x, { kind, bitrate }) {
	let mp3 = kind === 'mp3', rate = mp3 ? FS / 2 : FS
	let enc = mp3 ? await encodeMp3({ sampleRate: rate, channels: 1, bitrate }) : await encodeOpus({ sampleRate: rate, channels: 1, bitrate })
	let bytes = concat([enc.encode([mp3 ? decimate2(x) : x]), enc.flush()]); enc.free?.()
	let { channelData, sampleRate } = mp3 ? await decodeMp3(bytes) : await decodeOpus(bytes)
	let y = channelData[0]
	if (sampleRate === FS / 2) return interpolate2(y, 2 * y.length)
	return sampleRate === FS ? y : resample(y, { from: sampleRate, to: FS })
}

// apply(x, spec, render) → Float32Array: x is the dry note (not modified); render(mix) returns
// the other voice's note for a mix stage.
export async function apply(x, spec, render) {
	let y = Float32Array.from(x)
	let peak = 0
	for (let i = 0; i < y.length; i++) peak = Math.max(peak, Math.abs(y[i]))
	if (peak > 0) for (let i = 0; i < y.length; i++) y[i] *= 0.5 / peak // a nominal level before level-dependent stages
	if (spec.chorus) y = chorus(y, { ...spec.chorus, fs: FS })
	if (spec.eq) y = Float32Array.from(eq(Float64Array.from(y), { bands: spec.eq.map(b => ({ ...b })), fs: FS }))
	if (spec.reverb) y = spec.reverb.kind === 'dattorro'
		? dattorro(y, { decay: spec.reverb.decay, damping: spec.reverb.damping, mix: spec.reverb.mix, fs: FS })
		: freeverb(y, { room: spec.reverb.room, damp: spec.reverb.damp, mix: spec.reverb.mix, fs: FS })
	if (spec.mix) addAt(y, spec.mix.drums != null ? drums(spec.mix.drums, y.length) : render(spec.mix), spec.mix.snr)
	if (spec.noise) addAt(y, noise(spec.noise, y.length), spec.noise.snr)
	if (spec.compressor) y = compressor(y, { ...spec.compressor })
	if (spec.gain) { let g = db2lin(spec.gain); for (let i = 0; i < y.length; i++) y[i] *= g }
	if (spec.codec) y = await codec(y, spec.codec)
	return y
}
