// Features against their definition, the network against PyTorch (fixtures/parity.json, from
// scripts/export.py), pitch and voicing on rendered tones, and the pYIN stage 1 through
// @audio/pitch-pyin's track and notes, whole and streamed.
import test, { ok, is } from 'tst'
import { readFileSync } from 'node:fs'
import { createHash } from 'node:crypto'
import { track, notes } from '@audio/pitch-pyin'
import pitch, { frame, candidates, analyzer, network, CONFIG } from './pitch.js'
import { normalize, hz, BINS, LOW } from './features.js'
import { WEIGHTS } from './weights.js'

const fs = 44100
const cents = (a, b) => 1200 * Math.log2(a / b)
const half = h => { let e = (h >> 10) & 31, m = h & 1023, s = h & 32768 ? -1 : 1; return e ? s * (1 + m / 1024) * 2 ** (e - 15) : s * m * 2 ** -24 }

// harmonic tone following f(t) Hz, phase integrated per sample
function tone(f, dur, rate = fs, amp = 0.3) {
	let n = Math.round(dur * rate), d = new Float32Array(n), ph = 0
	for (let i = 0; i < n; i++) {
		d[i] = amp * (Math.sin(ph) + 0.6 * Math.sin(2 * ph) + 0.4 * Math.sin(3 * ph) + 0.25 * Math.sin(4 * ph))
		ph += 2 * Math.PI * f(i / rate) / rate
	}
	return d
}
function noise(n, seed = 1, amp = 0.1) {
	let d = new Float32Array(n), s = seed
	for (let i = 0; i < n; i++) { s = (s * 1103515245 + 12345) & 0x7fffffff; d[i] = amp * (s / 0x3fffffff - 1) }
	return d
}

// ------------------------------------------------ features

test('features: a sine peaks at its bin, full scale reads −6.02 dB, at every rate', () => {
	for (let rate of [16000, 22050, 44100, 48000]) {
		let x = Float32Array.from({ length: rate }, (_, i) => Math.sin(2 * Math.PI * 440 * i / rate))
		let db = analyzer(rate).db(x, rate >> 1), k = db.indexOf(Math.max(...db))
		is(LOW + k / 3, 69, `${rate} Hz: peak at MIDI 69`)
		ok(Math.abs(db[k] + 6.0206) < 0.01, `${rate} Hz: level ${db[k].toFixed(3)} dB`)
	}
})

test('features: the same tone gives the same input at 16 and 48 kHz', () => {
	let a = new Float32Array(BINS), b = new Float32Array(BINS)
	normalize(analyzer(16000).db(tone(() => 196, 1, 16000), 8000), a)
	normalize(analyzer(48000).db(tone(() => 196, 1, 48000), 24000), b)
	let worst = 0
	for (let k = 0; k < BINS; k++) if (hz(LOW + k / 3) < 7000 && (a[k] > 0.5 || b[k] > 0.5)) worst = Math.max(worst, Math.abs(a[k] - b[k]))
	ok(worst < 0.02, `bins within 60 dB of the peak differ by ${(worst * 80).toFixed(2)} dB at most`)
})

// ------------------------------------------------ network

test('network: equals PyTorch on the fixtures', () => {
	let fx = JSON.parse(readFileSync(new URL('./fixtures/parity.json', import.meta.url), 'utf8'))
	let bytes = Buffer.from(WEIGHTS.data, 'base64')
	is(createHash('sha256').update(bytes).digest('hex'), fx.weights, 'weights are the exported ones')
	let raw = Buffer.from(fx.input, 'base64'), input = Float32Array.from({ length: raw.length / 2 }, (_, i) => half(raw.readUInt16LE(2 * i)))
	let lb = Buffer.from(fx.logits, 'base64'), logits = new Float32Array(lb.buffer, lb.byteOffset, lb.length / 4)
	let net = network(), dl = 0, dv = 0
	for (let f = 0; f < fx.frames; f++) {
		let r = net(Float32Array.from(input.subarray(f * BINS, (f + 1) * BINS)), fx.level[f])
		for (let k = 0; k < BINS; k++) dl = Math.max(dl, Math.abs(r.logits[k] - logits[f * BINS + k]))
		dv = Math.max(dv, Math.abs(r.voicing - fx.voicing[f]))
	}
	ok(dl < 1e-4, `logits within ${dl.toExponential(2)} of PyTorch`)
	ok(dv < 1e-5, `voicing within ${dv.toExponential(2)}`)
	ok(CONFIG.K === BINS, 'config matches the features')
})

// ------------------------------------------------ frame-wise

test('pitch: steady tones from 55 to 1760 Hz within 20 cents, at 16 and 44.1 kHz', () => {
	for (let rate of [16000, fs]) for (let f of [55, 110, 220, 440, 880, 1760]) {
		let r = pitch(tone(() => f, 0.5, rate), { fs: rate }), c = [], v = []
		for (let i = 20; i < r.f0.length - 20; i++) { c.push(Math.abs(cents(r.f0[i], f))); v.push(r.voicing[i]) }
		c.sort((a, b) => a - b)
		ok(c[c.length >> 1] < 20, `${rate} Hz, ${f} Hz: median ${c[c.length >> 1].toFixed(1)} cents`)
		ok(Math.min(...v) > 0.5, `${rate} Hz, ${f} Hz: voiced`)
	}
})

test('pitch: silence and white noise are unvoiced', () => {
	let s = pitch(new Float32Array(fs), { fs }), n = pitch(noise(fs), { fs })
	ok(Math.max(...s.voicing) < 0.5, `silence: voicing ≤ ${Math.max(...s.voicing).toFixed(3)}`)
	let loud = n.voicing.filter(v => v >= 0.5).length
	ok(loud <= n.voicing.length * 0.05, `noise: ${loud} of ${n.voicing.length} frames voiced`)
})

test('pitch: frame i at i·hop, one per hop', () => {
	let r = pitch(new Float32Array(10000), { fs, hopSize: 100 })
	is(r.times.length, 101, 'frames 0…⌊n/hop⌋')
	is(r.times[7], 7 * 100 / fs, 'time of frame 7')
	let one = frame(fs)(tone(() => 330, 0.2), 4410)
	is(one.posterior.length, BINS, 'posterior over every bin')
	ok(Math.abs(one.posterior.reduce((a, b) => a + b) - 1) < 1e-5, 'posterior sums to 1')
})

// ------------------------------------------------ pYIN stage 1

test('candidates: pYIN track follows ±50 cent vibrato at 5.5 Hz', () => {
	let f = t => 220 * 2 ** (0.5 * Math.sin(2 * Math.PI * 5.5 * t) / 12)
	let r = track(tone(f, 2), { fs, candidates }), worst = 0, voiced = 0
	for (let i = 30; i < r.f0.length - 30; i++) if (r.f0[i] > 0) { voiced++; worst = Math.max(worst, Math.abs(cents(r.f0[i], f(r.times[i])))) }
	ok(voiced > (r.f0.length - 60) * 0.98, `voiced ${voiced} of ${r.f0.length - 60}`)
	ok(worst < 30, `within ${worst.toFixed(1)} cents`)
})

test('candidates: pYIN notes of a played line, in noise', () => {
	let line = [60, 62, 64, 65, 67], x = new Float32Array(Math.round(3.5 * fs))
	line.forEach((m, k) => x.set(tone(() => hz(m), 0.45), Math.round((0.3 + 0.6 * k) * fs)))
	let nz = noise(x.length, 3, 0.02)
	for (let i = 0; i < x.length; i++) x[i] += nz[i]
	let n = notes(x, { fs, candidates })
	is(n.map(e => e.midi), line, 'the five notes')
	ok(n.every((e, k) => Math.abs(e.time - (0.3 + 0.6 * k)) < 0.05), 'onsets within 50 ms')
})

test('candidates: streaming in odd blocks equals one call', () => {
	let x = tone(t => 196 * 2 ** (t / 2), 1.5), whole = track(x, { fs, candidates })
	let w = track({ fs, candidates }), parts = []
	for (let a = 0, b = 997; a < x.length; a += b, b = (b * 5) % 4000 + 300) parts.push(w(x.subarray(a, a + b)))
	parts.push(w())
	let f0 = parts.flatMap(p => Array.from(p.f0))
	is(f0, Array.from(whole.f0), 'same f0, frame for frame')
})
