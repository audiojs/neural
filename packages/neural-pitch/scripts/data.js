// Training data: phrases rendered by audiojs synths with their exact f0, then degraded.
//
// Every note is rendered at a fixed pitch by its synth, then read at a varying rate (varispeed, a
// Hann-windowed sinc read, anti-aliased when faster), so f0(t) = f0 of the render · rate(t) exactly:
// vibrato, scoops, drift and glides need no synth support. The singing voice (LF or Rosenberg pulse
// of @audio/voice-glottis through @audio/voice-tract) accumulates phase at f0(t) directly. Each
// render's f0 is the synth's true one, checked against YIN on steady notes (scripts/labels.js):
// @audio/synth-pluck's loop averages each sample with the next one still in the loop, so it sounds at
// fs / (N − ½); a DX7 patch keeps OP1, a carrier in every algorithm, at ratio 1 and the other operators
// at integer ratios, so the key is the f0; a tonewheel registration draws 8' but not 16' or 5⅓'.
// A frame is labelled with the note loudest there, if that note is within 40 dB of the clip's loudest
// frame and 6 dB above any other; frames where two notes are within 6 dB get no pitch label (weight 0);
// frames without a note are unvoiced.

import osc from '@audio/synth-osc'
import voice from '@audio/synth-voice'
import pluck from '@audio/synth-pluck'
import fm from '@audio/synth-fm'
import modal from '@audio/synth-modal'
import dx7, { INIT } from '@audio/synth-dx7'
import tonewheel, { wheel, wheelHz } from '@audio/synth-tonewheel'
import { lfCycle, rosenbergCycle } from '@audio/voice-glottis'
import tract, { VOWELS } from '@audio/voice-tract'
import { white, pink, brown, blue } from '@audio/synth-noise'
import resample, { sincRead } from '@audio/resample-sinc'
import eq from '@audio/eq-parametric'
import { highpass, lowpass, filter } from '@audio/biquad'
import freeverb from '@audio/reverb-freeverb'
import dattorro from '@audio/reverb-dattorro'
import fdn from '@audio/reverb-fdn'
import encodeMp3 from '@audio/encode-mp3'
import decodeMp3 from '@audio/decode-mp3'
import { fft, ifft } from 'fourier-transform'

const hz = m => 440 * 2 ** ((m - 69) / 12)

// ---------------------------------------------------------------- randomness

/** splitmix32-seeded xorshift: r() in [0, 1), with helpers */
export function rng(seed) {
	let s = (seed >>> 0) || 1
	let r = () => { s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 5; s >>>= 0; return s / 4294967296 }
	for (let i = 0; i < 4; i++) r()
	r.uniform = (a, b) => a + (b - a) * r()
	r.int = (a, b) => a + Math.floor((b - a + 1) * r())
	r.pick = a => a[Math.floor(r() * a.length)]
	r.weighted = w => { let t = r() * Object.values(w).reduce((a, b) => a + b), k; for (k in w) if ((t -= w[k]) < 0) return k; return k }
	r.normal = () => Math.sqrt(-2 * Math.log(1 - r())) * Math.cos(2 * Math.PI * r())
	r.log = (a, b) => a * (b / a) ** r()
	return r
}

// ---------------------------------------------------------------- phrases

// Pitch ranges (MIDI) and articulation per source
const SOURCES = {
	voice: { lo: 38, hi: 84, sustain: true },       // glottis + tract
	osc: { lo: 28, hi: 96, sustain: true },
	synth: { lo: 28, hi: 91, sustain: true },       // @audio/synth-voice
	pluck: { lo: 33, hi: 88, sustain: false },
	fm: { lo: 28, hi: 91, sustain: true },
	modal: { lo: 33, hi: 93, sustain: false },
	dx7: { lo: 28, hi: 93, sustain: true },
	tonewheel: { lo: 36, hi: 90, sustain: true }
}
export const SOURCE_NAMES = Object.keys(SOURCES)

/** A phrase: notes with onset, duration, nominal MIDI pitch (fractional) and expression. */
function phrase(r, src, length, steady = false) {
	let { lo, hi, sustain } = SOURCES[src]
	let span = r.int(5, 14), centre = r.uniform(lo + span / 2, hi - span / 2)
	let tuning = r.uniform(-50, 50) / 100
	let style = r.weighted(sustain ? { legato: 3, normal: 3, staccato: 1 } : { normal: 3, staccato: 1 })
	let vib = r.weighted({ none: 4, light: 3, strong: 3 })
	let vibDepth = vib === 'none' ? 0 : vib === 'light' ? r.uniform(8, 40) : r.uniform(40, 120), vibRate = r.uniform(4, 8)
	let notes = [], t = r.uniform(0, 0.4), m = centre + r.uniform(-span / 2, span / 2)
	while (t < length - 0.1) {
		let d = style === 'staccato' ? r.uniform(0.06, 0.25) : r.log(0.1, 1.5)
		let gap = style === 'legato' ? 0 : style === 'staccato' ? r.uniform(0.05, 0.5) : r.log(0.02, 0.4)
		if (r() < 0.08) gap += r.uniform(0.3, 1.2)                         // a rest
		let step = r() < 0.15 ? r.int(-12, 12) : r.int(-3, 3)
		m = Math.max(centre - span / 2, Math.min(centre + span / 2, m + step))
		d = Math.min(d, length - t)
		notes.push({
			t, d, midi: Math.round(m) + tuning + r.normal() * 0.12,
			gain: r.uniform(-8, 0),
			scoop: r() < 0.25 ? { depth: r.uniform(30, 200), tau: r.uniform(0.02, 0.1) } : null,
			fall: r() < 0.08 ? { depth: r.uniform(50, 300), time: r.uniform(0.05, 0.2) } : null,
			vib: vibDepth ? { depth: vibDepth * r.uniform(0.7, 1.3), rate: vibRate * r.uniform(0.9, 1.1), delay: r.uniform(0, 0.3), phase: r() * 2 * Math.PI } : null,
			glide: style === 'legato' && r() < 0.5 ? r.uniform(0.02, 0.2) : 0,
			drift: [0, 1, 2].map(() => ({ a: r.uniform(0, 8), f: r.uniform(0.2, 1.5), p: r() * 2 * Math.PI }))
		})
		t += d + gap
	}
	for (let i = 0; i < notes.length; i++) notes[i].next = style === 'legato' && notes[i + 1] && notes[i + 1].t - notes[i].t - notes[i].d < 1e-9 ? notes[i + 1] : null
	if (steady) for (let nt of notes) { nt.scoop = nt.fall = nt.vib = null; nt.glide = 0; nt.drift = [] }
	return { notes, style }
}

/** Pitch deviation of a note from its nominal pitch at time u into it (cents); the glide goes to the next note. */
function cents(n, u) {
	let c = 0
	if (n.scoop) c -= n.scoop.depth * Math.exp(-u / n.scoop.tau)
	if (n.fall && u > n.d - n.fall.time) c -= n.fall.depth * ((Math.min(u, n.d) - n.d + n.fall.time) / n.fall.time) ** 2
	if (n.vib && u > n.vib.delay) c += n.vib.depth * Math.min(1, (u - n.vib.delay) / 0.15) * Math.sin(2 * Math.PI * n.vib.rate * u + n.vib.phase)
	for (let d of n.drift) c += d.a * Math.sin(2 * Math.PI * d.f * u + d.p)
	if (n.next && n.glide && u > n.d - n.glide) {
		let x = Math.min(1, (u - n.d + n.glide) / n.glide)
		c += (n.next.midi - n.midi) * 100 * x * x * (3 - 2 * x)   // smoothstep
	}
	return c
}

/** cents(n, u) for u = i / fs, i < m: evaluated every 16 samples, linear between (the curve rendered is the curve labelled). */
function curve(n, fs, m) {
	let c = new Float64Array(m), K = 16
	for (let i = 0; i < m; i += K) {
		let a = cents(n, i / fs), b = cents(n, Math.min(i + K, m) / fs)
		for (let j = 0; j < K && i + j < m; j++) c[i + j] = a + (b - a) * j / K
	}
	return c
}

// ---------------------------------------------------------------- sources

/** Varispeed read of `src` (its f0 constant) at rates 2^(c[i]/1200) for output sample i. */
function varispeed(src, c) {
	let out = new Float32Array(c.length), pos = 0
	for (let i = 0; i < c.length && pos < src.length - 1; i++) {
		let rate = 2 ** (c[i] / 1200)
		out[i] = sincRead(src, pos, 8, Math.min(1, 1 / rate))
		pos += rate
	}
	return out
}

// 2:1 decimation: windowed-sinc lowpass at 0.45 of the output rate, 95 taps (Blackman)
const HB = (() => {
	let M = 47, h = new Float64Array(2 * M + 1), s = 0
	for (let k = -M; k <= M; k++) {
		let x = 0.45 * k, w = 0.42 + 0.5 * Math.cos(Math.PI * k / (M + 1)) + 0.08 * Math.cos(2 * Math.PI * k / (M + 1))
		s += h[k + M] = (k ? Math.sin(Math.PI * x) / (Math.PI * x) : 1) * w
	}
	return h.map(v => v / s)
})()
function halve(x) {
	let n = x.length >> 1, y = new Float32Array(n), M = HB.length >> 1
	for (let i = 0; i < n; i++) {
		let s = 0, c = 2 * i
		for (let k = -M; k <= M; k++) { let j = c + k; if (j >= 0 && j < x.length) s += HB[k + M] * x[j] }
		y[i] = s
	}
	return y
}

function envelope(fs, n, d, attack, release) {
	let e = new Float32Array(n), a = Math.max(1, attack * fs), rl = Math.max(1, release * fs), off = d * fs
	for (let i = 0; i < n; i++) e[i] = Math.min(1, i / a) * (i < off ? 1 : Math.max(0, 1 - (i - off) / rl))
	return e
}

// DX7 voice: a random patch whose six operators run at integer ratios (coarse 1…8, fine 0, detune 0) with
// OP1 at ratio 1, so its partials are harmonics of the key; pitch EG flat, LFO pitch depth 0.
function dx7Patch(r) {
	let p = Uint8Array.from(INIT), ratios = []
	for (let op = 0; op < 6; op++) {
		// OP1 (stored last) is a carrier in every algorithm: at ratio 1 and near full level, the key is the f0
		let o = op * 21, coarse = op === 5 || r() < 0.6 ? 1 : r.int(1, 8)
		let level = op === 5 ? r.int(90, 99) : r() < 0.5 ? r.int(60, 95) : r.int(0, 60)
		p.set([r.int(50, 99), r.int(20, 90), r.int(20, 90), r.int(30, 80), 99, r.int(70, 99), r.int(50, 95), 0], o)
		p[o + 16] = level; p[o + 17] = 0; p[o + 18] = coarse; p[o + 19] = 0; p[o + 20] = 7
		p[o + 15] = r.int(0, 3)   // velocity sensitivity
		ratios.push(coarse)
	}
	p[134] = r.int(0, 31); p[135] = r.int(0, 7)                          // algorithm, feedback
	p.set([50, 50, 50, 50], 130)                                         // pitch EG levels
	p[139] = 0; p[143] = 0                                               // LFO pitch depth, pitch sensitivity
	p[144] = 24                                                          // transpose C3: no shift
	return p
}

// Tonewheel registration: 8' drawn, 16' and 5⅓' not, so the 8' wheel is the f0 and every other wheel a
// (tempered) harmonic of it. With 16' drawn the period doubles while the 8' dominates: an ambiguous label.
function registration(r) {
	let d = Array.from({ length: 9 }, () => r() < 0.5 ? 0 : r.int(1, 8))
	d[0] = d[1] = 0; d[2] = r.int(4, 8)
	return d
}

// One note at nominal MIDI pitch `m`, fixed pitch, `sec` long including its tail: → { x, f0 }
function note(r, src, m, sec, fs, cfg) {
	let f = hz(m)
	switch (src) {
		case 'osc': return { x: osc(f, { duration: sec, fs, type: cfg.type, amp: 1, phase: r() }), f0: f }
		case 'synth': return { x: voice(f, { duration: Math.max(0.02, sec - cfg.release), fs, type: cfg.type, attack: cfg.attack, decay: cfg.decay, sustain: cfg.sustain, release: cfg.release, fc: cfg.fc, envAmount: cfg.envAmount, amp: 1 }), f0: f }
		case 'pluck': { let N = Math.max(2, Math.round(fs / f)); return { x: pluck(f, { duration: sec, fs, damp: cfg.damp, amp: 1, seed: r.int(1, 1e6) }), f0: fs / (N - 0.5) } }
		case 'fm': return { x: fm(f, { duration: sec, fs, ops: cfg.ops, attack: cfg.attack, release: Math.min(cfg.release, sec / 2), amp: 1 }), f0: f }
		case 'modal': return { x: modal(f, { ...cfg.modal, duration: sec, fs, amp: 1, seed: r.int(1, 1e6) }), f0: f }
		case 'dx7': {
			let key = Math.round(m), fk = hz(key)
			let up = fs > 16384 ? 1 : 2, x = dx7(fk, { patch: cfg.patch, duration: Math.max(0.02, sec - cfg.release), release: cfg.release, velocity: r.uniform(0.5, 1), fs: fs * up })
			return { x: up > 1 ? halve(x) : x, f0: fk }   // MSFA's tables start above 16384 Hz
		}
		case 'tonewheel': {
			let key = Math.max(36, Math.min(96, Math.round(m)))
			// f0: the lowest drawn wheel, or its half or third if some drawn wheel is off its harmonics (foldback)
			let f = cfg.drawbars.flatMap((v, i) => v > 0 ? [wheelHz(wheel(key, i))] : []), lo = Math.min(...f)
			let f0 = [1, 2, 3, 4].map(k => lo / k).find(F => f.every(v => Math.abs(v / F / Math.round(v / F) - 1) < 0.01))
			return { x: tonewheel([{ midi: key, time: 0, duration: Math.max(0.02, sec - 0.03) }], { drawbars: cfg.drawbars, fs, percussion: cfg.percussion, click: cfg.click }), f0 }
		}
	}
}

// Singing voice: LF (or Rosenberg) pulses from @audio/voice-glottis read at f0(t) with continuous phase,
// rendered at 2·fs and halved, through the vocal tract with a vowel per note.
function sing(r, ph, fs, n) {
	let model = r() < 0.8 ? 'lf' : 'rosenberg', Rd = r.uniform(0.4, 2.6)
	let table = model === 'lf' ? lfCycle(4096, { Rd }) : rosenbergCycle(4096, { open: r.uniform(0.3, 0.6), close: r.uniform(0.08, 0.2) })
	let shimmer = r.uniform(0, 0.06), jitter = r.uniform(0, 0.008), breath = r.uniform(0, 0.08)
	let F = 2 * fs, N = 2 * n, src = new Float32Array(N), gain = new Float32Array(N), f0 = new Float64Array(N)
	ph.notes.forEach((nt, k) => {
		// a legato note starts without an attack and ends without a release: the voice runs on
		let into = ph.notes[k - 1]?.next === nt, a = Math.round(nt.t * F), b = Math.min(N, Math.round((nt.t + nt.d + (nt.next ? 0 : 0.05)) * F))
		let fn = hz(nt.midi), g = 10 ** (nt.gain / 20), att = into ? 1e-3 : r.uniform(0.01, 0.08), rel = nt.next ? 1e-3 : 0.03
		let c = curve(nt, F, Math.max(0, b - a))
		for (let i = a; i < b; i++) {
			f0[i] = fn * 2 ** (c[i - a] / 1200)
			gain[i] = g * Math.min(1, (i - a) / F / att) * Math.min(1, (b - i) / (rel * F))
		}
	})
	let phase = 0, amp = 1, jit = 1, noise = rng(r.int(1, 1e9))
	for (let i = 0; i < N; i++) {
		if (!f0[i]) { phase = 0; continue }
		let p = phase * 4096, k = Math.floor(p), fr = p - k
		src[i] = gain[i] * (amp * (table[k] * (1 - fr) + table[(k + 1) & 4095] * fr) + breath * (2 * noise() - 1))
		phase += f0[i] * jit / F
		if (phase >= 1) { phase -= 1; amp = 1 + shimmer * (2 * noise() - 1); jit = 1 + jitter * (2 * noise() - 1) }
		f0[i] *= jit
	}
	let x = halve(src), y = new Float32Array(n)
	// tract per note, vowel and length changing between notes; state carried
	let sections = r.int(18, 30), opts = { sections, lipReflection: r.uniform(-0.9, -0.75), glottalReflection: r.uniform(0.6, 0.85), damping: r.uniform(0.99, 0.998), fs }
	let edges = [0, ...ph.notes.map(nt => Math.round(nt.t * fs)), n]
	for (let k = 0; k + 1 < edges.length; k++) {
		let a = edges[k], b = edges[k + 1]
		if (b <= a) continue
		opts.shape = r() < 0.8 ? r.pick(Object.keys(VOWELS)) : VOWELS.a.map(v => v * r.uniform(0.4, 1.8))
		y.set(tract(x.subarray(a, b), opts), a)
	}
	// f0 at fs, as the source sample nearest each output sample
	let f = new Float64Array(n)
	for (let i = 0; i < n; i++) f[i] = f0[Math.min(N - 1, 2 * i)]
	return { a: 0, x: y, f }
}

/** Source configuration for a phrase. */
function configure(r, src) {
	switch (src) {
		case 'osc': return { type: r.weighted({ sine: 1, triangle: 2, square: 2, sawtooth: 3 }), attack: r.log(0.003, 0.1), release: r.log(0.01, 0.3) }
		case 'synth': return { type: r.weighted({ sine: 1, triangle: 2, square: 3, sawtooth: 3 }), attack: r.log(0.003, 0.1), decay: r.log(0.05, 0.5), sustain: r.uniform(0.3, 1), release: r.log(0.02, 0.3), fc: r.log(400, 8000), envAmount: r.uniform(0, 0.9) }
		case 'pluck': return { damp: r.uniform(0.985, 0.9995) }
		case 'fm': {
			let ops = Array.from({ length: r.int(1, 3) }, () => ({ ratio: r.pick([1, 1, 2, 2, 3, 4, 5, 7]), index: r.log(0.2, 6), indexDecay: r() < 0.5 ? r.log(0.05, 1) : 0, indexFloor: r.uniform(0, 1), feedback: r() < 0.3 ? r.uniform(0, 1.2) : 0 }))
			return { ops, attack: r.log(0.002, 0.08), release: r.log(0.02, 0.3) }
		}
		case 'modal': {
			let model = r.weighted({ string: 4, 'tube-open': 1, 'tube-closed': 1, bar: 1 })
			return { modal: { model, nmodes: r.int(4, 16), t60: r.log(0.3, 4), damping: r.uniform(0.3, 1.2), inharmonicity: model === 'string' && r() < 0.5 ? r.log(1e-5, 3e-4) : 0, strike: r.uniform(0.05, 0.5), exciter: r() < 0.7 ? 'impulse' : 'noise' } }
		}
		case 'dx7': return { patch: dx7Patch(r), release: r.log(0.05, 0.8) }
		case 'tonewheel': return { drawbars: registration(r), percussion: r() < 0.25 ? r.pick(['second', 'third']) : false, click: r() }
	}
}

// ---------------------------------------------------------------- clips

/**
 * A labelled clip: { x (Float32Array at fs), fs, frames (centre samples), f0 (Hz, 0 unvoiced),
 * weight (0: two notes compete, no pitch label) }.
 */
export async function clip(seed, { fs, length = 6, hop = 0.025, source, clean = false, steady = false } = {}) {
	let r = rng(seed)
	fs ??= +r.weighted({ 16000: 2, 22050: 1.5, 44100: 4.5, 48000: 2 })
	let n = Math.round(length * fs)
	let kind = source ?? (r() < 0.08 ? 'none' : r.weighted({ voice: 5, osc: 1.5, synth: 1.5, pluck: 1, fm: 1.5, modal: 1, dx7: 1.5, tonewheel: 1 }))

	// clean notes: each with its samples placed on the clip and its f0 per sample (0 outside)
	let parts = []
	if (kind === 'voice') parts.push(sing(r, phrase(r, 'voice', length, steady), fs, n))
	else if (kind !== 'none') {
		let ph = phrase(r, kind, length, steady), cfg = configure(r, kind), sustain = SOURCES[kind].sustain
		for (let nt of ph.notes) {
			let tail = sustain ? (cfg.release ?? 0.1) : r.uniform(0.2, 1.5)
			let sec = Math.min(length - nt.t, nt.d + tail) + 0.05
			let { x, f0 } = note(r, kind, nt.midi, sec * 1.6 + 0.1, fs, cfg)   // headroom for slower reads
			// the render's own pitch vs the nominal one: dx7 and tonewheel play keys; the rest follow nt.midi
			let a = Math.round(nt.t * fs), m = Math.min(n - a, Math.round(sec * fs))
			if (m <= 0) continue
			// dx7 and tonewheel play the nearest key: the read rate adds the rest of the nominal pitch
			let key = kind === 'dx7' || kind === 'tonewheel', c = curve(nt, fs, m), off = key ? 100 * (nt.midi - Math.round(nt.midi)) : 0
			let y = varispeed(x, off ? c.map(v => v + off) : c)
			let env = sustain ? envelope(fs, m, nt.d, cfg.attack ?? 0.01, cfg.release ?? 0.05) : null
			let g = 10 ** (nt.gain / 20), f = new Float64Array(m)
			let base = f0 * 2 ** (off / 1200)
			for (let i = 0; i < m; i++) {
				y[i] *= g * (env ? env[i] : 1)
				f[i] = base * 2 ** (c[i] / 1200)
			}
			parts.push({ a, x: y, f })
		}
	}

	// labels from the clean parts, at frame centres every `hop` s from a random start
	let H = Math.round(hop * fs), start = r.int(0, H - 1), frames = []
	for (let c = start; c < n; c += H) frames.push(c)
	let W = Math.round(0.02 * fs), fAt = (p, c) => c >= p.a && c < p.a + p.f.length ? p.f[c - p.a] : 0
	let rms = parts.map(p => frames.map(c => {
		let s = 0, a = Math.max(p.a, c - (W >> 1)), b = Math.min(p.a + p.x.length, c + (W >> 1))
		for (let i = a; i < b; i++) s += p.x[i - p.a] ** 2
		return Math.sqrt(s / W)
	}))
	let peak = Math.max(1e-12, ...rms.map(e => Math.max(...e)))
	let f0 = new Float32Array(frames.length), weight = new Float32Array(frames.length).fill(1)
	frames.forEach((c, j) => {
		let best = -1, second = 0
		rms.forEach((e, k) => { if (fAt(parts[k], c) > 0 && (best < 0 || e[j] > rms[best][j])) best = k })
		if (best < 0 || rms[best][j] < peak * 0.01) return                 // −40 dB
		rms.forEach((e, k) => { if (k !== best && e[j] > second) second = e[j] })
		f0[j] = fAt(parts[best], c)
		if (second * 2 > rms[best][j]) weight[j] = 0                          // within 6 dB
	})

	let x = new Float32Array(n)
	for (let p of parts) for (let i = 0; i < p.x.length && p.a + i < n; i++) x[p.a + i] += p.x[i]
	if (kind === 'none') x = negative(r, n, fs)
	if (!clean) x = await degrade(r, x, fs)
	return { x, fs, frames: Int32Array.from(frames), f0, weight, kind }
}

// Unpitched content: coloured noise bursts, tract-filtered noise (fricatives, breath), clicks, silence.
function negative(r, n, fs) {
	let x = new Float32Array(n), t = 0
	while (t < n) {
		let d = Math.round(r.log(0.05, 1.5) * fs), a = t + Math.round(r.log(0.02, 0.8) * fs)
		if (a >= n) break
		d = Math.min(d, n - a)
		let kind = r.weighted({ noise: 3, fricative: 3, click: 1, silence: 1 }), g = r.log(0.02, 1), y
		if (kind === 'noise') y = r.pick([white, pink, brown, blue])(d / fs, { fs, seed: r.int(1, 65535) })
		else if (kind === 'fricative') y = tract(white(d / fs, { fs, seed: r.int(1, 65535) }), { shape: r.pick(Object.keys(VOWELS)), sections: r.int(12, 30), fs })
		else if (kind === 'click') { y = new Float32Array(d); for (let i = 0; i < d; i += Math.round(r.log(0.005, 0.2) * fs)) y[i] = r.uniform(-1, 1) }
		else y = new Float32Array(d)
		let e = envelope(fs, d, d / fs - 0.01, r.log(0.001, 0.05), 0.01)
		for (let i = 0; i < d && a + i < n; i++) x[a + i] += g * y[i] * e[i]
		t = a + d
	}
	return x
}

// ---------------------------------------------------------------- degradations

const rmsOf = x => { let s = 0; for (let i = 0; i < x.length; i++) s += x[i] * x[i]; return Math.sqrt(s / Math.max(1, x.length)) }

/** Room: a synthetic impulse response (exponential noise tail, decay faster at high frequencies), or an algorithmic reverb. */
function room(r, x, fs) {
	let kind = r.weighted({ ir: 3, freeverb: 1, dattorro: 1, fdn: 1 })
	if (kind === 'freeverb') return freeverb(Float32Array.from(x), { room: r.uniform(0.3, 0.95), damp: r.uniform(0.1, 0.8), mix: r.uniform(0.15, 0.6), fs })
	if (kind === 'dattorro') return dattorro(Float32Array.from(x), { decay: r.uniform(0.3, 0.9), damping: r.uniform(0.05, 0.7), mix: r.uniform(0.15, 0.6), fs })
	if (kind === 'fdn') return fdn(Float32Array.from(x), { t60: r.log(0.3, 3), damping: r.uniform(0.1, 0.8), mix: r.uniform(0.15, 0.6), fs })
	let t60 = r.log(0.15, 2.5), len = Math.round(Math.min(3, 1.2 * t60) * fs), pre = Math.round(r.uniform(0.002, 0.02) * fs)
	let lo = rng(r.int(1, 1e9)), ir = new Float64Array(len)
	let t60hi = t60 * r.uniform(0.3, 0.8), split = r.log(1000, 5000), a = Math.exp(-2 * Math.PI * split / fs)
	let lp = 0
	for (let i = pre; i < len; i++) {
		let w = lo.normal(), t = (i - pre) / fs
		lp = (1 - a) * w + a * lp
		ir[i] = lp * 10 ** (-3 * t / t60) + (w - lp) * 10 ** (-3 * t / t60hi)
	}
	let e = 0
	for (let i = 0; i < len; i++) e += ir[i] * ir[i]
	let drr = r.uniform(-4, 12), g = Math.sqrt(10 ** (-drr / 10) / e)
	for (let i = 0; i < len; i++) ir[i] *= g
	ir[0] = 1
	return convolve(x, ir)
}

/** Linear convolution by FFT, same length as x. */
export function convolve(x, h) {
	let L = 2 ** Math.ceil(Math.log2(x.length + h.length)), a = new Float64Array(L), b = new Float64Array(L)
	a.set(x); b.set(h)
	let [ar, ai] = fft(a).map(v => Float64Array.from(v)), [br, bi] = fft(b)
	for (let k = 0; k < ar.length; k++) { let re = ar[k] * br[k] - ai[k] * bi[k]; ai[k] = ar[k] * bi[k] + ai[k] * br[k]; ar[k] = re }
	return Float32Array.from(ifft(ar, ai).subarray(0, x.length))
}

async function mp3(r, x, fs) {
	let bitrate = r.pick([32, 48, 64, 96, 128])
	let enc = await encodeMp3({ sampleRate: fs, channels: 1, bitrate })
	let parts = [enc.encode([x]), enc.flush()]; enc.free?.()
	let bytes = new Uint8Array(parts.reduce((s, p) => s + p.length, 0)), o = 0
	for (let p of parts) { bytes.set(p, o); o += p.length }
	let { channelData, sampleRate } = await decodeMp3(bytes)
	let y = sampleRate === fs ? channelData[0] : resample(channelData[0], { from: sampleRate, to: fs })
	// the codec's delay, by cross-correlation over the first 4096 lags
	let best = 0, arg = 0, m = Math.min(x.length, y.length - 4096)
	for (let lag = 0; lag < 4096; lag += 1) {
		let s = 0
		for (let i = 0; i < m; i += 7) s += x[i] * y[i + lag]
		if (s > best) { best = s; arg = lag }
	}
	let out = new Float32Array(x.length)
	out.set(y.subarray(arg, arg + x.length))
	return out
}

/** The degradation chain: EQ, room, noise at 0–20 dB SNR, codec, level. */
export async function degrade(r, x, fs) {
	let y = Float64Array.from(x)
	if (r() < 0.5) {
		let bands = Array.from({ length: r.int(1, 3) }, () => ({ fc: r.log(80, Math.min(9000, fs * 0.4)), Q: r.log(0.4, 4), gain: r.uniform(-12, 12), type: r.weighted({ peak: 3, lowshelf: 1, highshelf: 1 }) }))
		eq(y, { bands, fs })
	}
	if (r() < 0.12) filter(y, { coefs: highpass(r.log(100, 400), Math.SQRT1_2, fs) })     // thin: fundamental gone
	if (r() < 0.15) filter(y, { coefs: lowpass(r.log(1500, Math.min(8000, fs * 0.45)), Math.SQRT1_2, fs) })
	y = Float32Array.from(y)
	if (r() < 0.35) y = room(r, y, fs)
	let s = rmsOf(y)
	if (r() < 0.6 && s > 0) {
		let snr = r.uniform(0, 20), nz = r.pick([white, pink, pink, brown, blue])(y.length / fs, { fs, seed: r.int(1, 65535) })
		let g = s / (rmsOf(nz) * 10 ** (snr / 20))
		for (let i = 0; i < y.length; i++) y[i] += g * nz[i]
	} else if (r() < 0.3) {
		let nz = white(y.length / fs, { fs, seed: r.int(1, 65535) }), g = r.log(1e-5, 1e-3) / rmsOf(nz)
		for (let i = 0; i < y.length; i++) y[i] += g * nz[i]                 // hiss floor only
	}
	if (r() < 0.04) {
		let f = r.pick([50, 60]), g = s * r.log(0.01, 0.2)
		for (let i = 0; i < y.length; i++) y[i] += g * (Math.sin(2 * Math.PI * f * i / fs) + 0.5 * Math.sin(4 * Math.PI * f * i / fs + 1) + 0.3 * Math.sin(6 * Math.PI * f * i / fs + 2))
	}
	if (r() < 0.08 && y.length > 8192) y = await mp3(r, y, fs)
	let peak = 0
	for (let i = 0; i < y.length; i++) peak = Math.max(peak, Math.abs(y[i]))
	if (peak > 0) {
		let clip = r() < 0.03, g = 10 ** (r.uniform(clip ? 0 : -45, clip ? 8 : -1) / 20) / peak
		for (let i = 0; i < y.length; i++) y[i] = Math.max(-1, Math.min(1, y[i] * g))
	}
	return y
}
