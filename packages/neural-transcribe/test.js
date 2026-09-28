// Without a model: the note-creation port against basic-pitch's own on fixtures/notes.json
// (fixtures/make-notes.py), windowing and unwrapping through a mock model, input handling.
// With the model in the neural cache (the first transcribe() fetches it) the ONNX path runs;
// with scripts/reference.py's outputs there too, results are compared against Python basic-pitch.
import test, { ok, is, rejects, throws } from 'tst'
import { existsSync, readFileSync, readdirSync } from 'node:fs'
import { createHash } from 'node:crypto'
import os from 'node:os'
import path from 'node:path'
import { fetchModel } from '@audio/neural-runtime'
import transcribe, { posteriors, toNotes, MODEL } from './transcribe.js'

const CACHE = process.env.AUDIO_NEURAL_CACHE || path.join(os.homedir(), '.cache', 'audiojs', 'neural')
const REF = path.join(CACHE, 'basic-pitch')
const HAS_MODEL = existsSync(path.join(CACHE, createHash('sha256').update(MODEL).digest('hex')))

const rows = (flat, w) => Array.from({ length: flat.length / w }, (_, t) => flat.subarray(t * w, (t + 1) * w))
const f32 = file => { let b = readFileSync(file); return new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4) }
const HOP_S = 256 / 22050
const MODEL_RUN = { timeout: 120000 } // a real model run on a busy machine

// Ours, with `upstream: true`, against basic-pitch's [start_s, end_s, midi, amplitude, bins]: the
// same notes, times to 1e-9, bends exact, velocity to `tol` (numpy averages the float32 posterior
// in float32).
function sameNotes(js, py, label, tol = 1e-6) {
	let ref = py.map(([s, e, m, a, b]) => ({ s, e, m, a, b })).sort((x, y) => x.s - y.s || x.m - y.m)
	is(js.length, ref.length, `${label}: note count`)
	let bad = js.filter((n, i) => {
		let r = ref[i]
		return !r || n.midi !== r.m || Math.abs(n.time - r.s) > 1e-9 || Math.abs(n.time + n.duration - r.e) > 1e-9 ||
			Math.abs(n.velocity - r.a) > tol || n.bends.length !== r.b.length || n.bends.some((c, k) => c !== r.b[k] * 100 / 3)
	})
	is(bad.length, 0, `${label}: notes differing from basic-pitch`)
}

// ------------------------------------------------ 1. note creation vs basic-pitch

// fixtures/make-notes.py's posteriors(), bit for bit: 24-bit LCG noise under 0.25 plus events
// [key, first frame, end frame, note posterior, onset peak, gap at 0.125, bend pattern in bins]
const EVENTS = [
	[39, 10, 40, 0.75, 0.875, null, [0]],
	[40, 20, 60, 0.625, 0.75, null, [0, 1, 1, 0]],
	[0, 50, 80, 0.75, 0.875, null, [-1, 0, 1, 2]],
	[87, 50, 90, 0.75, 0.875, null, [0, 1, 2, 1, -2]],
	[60, 100, 150, 0.75, 0.875, [115, 121], [0, 0, 1]],
	[62, 100, 170, 0.75, 0.875, [120, 135], [-1, 0]],
	[20, 180, 191, 0.75, 0.875, null, [0]],
	[25, 180, 192, 0.75, 0.875, null, [0]],
	[70, 200, 240, 0.9375, 0, null, [1, 0, -1]],
	[50, 250, 280, 0.625, 0, null, [0]],
	[55, 250, 280, 0.625, 0, null, [0]],
	[30, 330, 400, 0.75, 0.875, null, [0, 3, -3]],
]
function synthetic(T = 400) {
	let s = 1, u = () => (s = (s * 1664525 + 1013904223) % 4294967296, (s >>> 8) / 16777216)
	let note = new Float32Array(T * 88), onset = new Float32Array(T * 88), contour = new Float32Array(T * 264)
	for (let a of [note, onset, contour]) for (let i = 0; i < a.length; i++) a[i] = u() * 0.25
	for (let [key, t0, t1, amp, peak, gap, bend] of EVENTS) {
		for (let t = t0; t < Math.min(t1, T); t++) {
			note[t * 88 + key] = gap && gap[0] <= t && t < gap[1] ? 0.125 : amp
			let b = 3 * key + bend[(t - t0) % bend.length]
			for (let [d, v] of [[-1, 0.5], [1, 0.5], [0, 0.875]]) if (b + d >= 0 && b + d < 264) contour[t * 264 + b + d] = v
		}
		if (peak && t0 < T) onset[t0 * 88 + key] = peak
	}
	return { note: rows(note, 88), onset: rows(onset, 88), contour: rows(contour, 264) }
}

test('toNotes: basic-pitch note creation on synthetic posteriors, 8 option sets (fixtures/notes.json)', () => {
	let p = synthetic()
	for (let { opts, notes } of JSON.parse(readFileSync(new URL('./fixtures/notes.json', import.meta.url)))) sameNotes(toNotes(p, { ...opts, upstream: true }), notes, JSON.stringify(opts))
})

// One note per key at the keyboard's ends and middle, its contour peak stepping through -1, 0, +1
// bins around 3(m - 21) + 1, where the model puts MIDI m; zeros elsewhere, so the peak wins
test('toNotes: bends are cents from the bin of the note\'s pitch; upstream: true reads them one bin (+33.3 cents) sharp, as basic-pitch', () => {
	let T = 80, keys = [0, 40, 87], steps = [-1, 0, 1]
	let note = new Float32Array(T * 88), onset = new Float32Array(T * 88), contour = new Float32Array(T * 264)
	for (let key of keys) {
		onset[10 * 88 + key] = 0.9
		for (let t = 10; t < 60; t++) { note[t * 88 + key] = 0.9; contour[t * 264 + 3 * key + 1 + steps[t % 3]] = 1 }
	}
	let p = { note: rows(note, 88), onset: rows(onset, 88), contour: rows(contour, 264) }
	for (let [upstream, shift] of [[false, 0], [true, 1]]) {
		let notes = toNotes(p, { upstream })
		is(notes.map(n => n.midi), keys.map(k => k + 21), 'one note per key')
		for (let n of notes) is(n.bends, Array.from({ length: 50 }, (_, k) => (steps[(10 + k) % 3] + shift) * 100 / 3), `MIDI ${n.midi}${upstream ? ', upstream' : ''}`)
	}
})

test('toNotes: 60 s of dense posteriors (109k melodia candidates) in < 1.5 s of CPU (basic-pitch rescans all cells per note)', () => {
	let p = synthetic(5168), c0 = process.cpuUsage()
	let n = toNotes(p, { frameThreshold: 0.19 })
	let { user, system } = process.cpuUsage(c0), cpu = (user + system) / 1e6
	console.log(`  (${n.length} notes, ${cpu.toFixed(2)} s CPU)`)
	ok(cpu < 1.5, `${cpu.toFixed(2)} s CPU`)
})

test('toNotes: frame threshold below 0 throws (basic-pitch loops forever); empty posteriors give no notes', () => {
	throws(() => toNotes(synthetic(10), { frameThreshold: -0.1 }), /frameThreshold/)
	is(toNotes({ note: [], onset: [], contour: [] }), [])
})

// ------------------------------------------------ 2. windowing and unwrapping

// Mock model: frame j of each window reports the window's sample at j·256 as its note posterior,
// the negation as onset, twice it as contour, so the unwrapped frames name their source exactly.
function probe() {
	let calls = 0, inputs = []
	let session = async () => ({
		inputs: [{ name: 'serving_default_input_2:0' }],
		async run(feeds) {
			let { data, dims: [B, N, C] } = feeds['serving_default_input_2:0']
			calls++; inputs.push({ data, dims: [B, N, C] })
			let note = new Float32Array(B * 172 * 88), onset = new Float32Array(B * 172 * 88), contour = new Float32Array(B * 172 * 264)
			for (let b = 0; b < B; b++) for (let j = 0; j < 172; j++) {
				let v = data[b * N + j * 256], r = b * 172 + j
				note.fill(v, r * 88, (r + 1) * 88); onset.fill(-v, r * 88, (r + 1) * 88); contour.fill(2 * v, r * 264, (r + 1) * 264)
			}
			return { 'StatefulPartitionedCall:1': { data: note }, 'StatefulPartitionedCall:2': { data: onset }, 'StatefulPartitionedCall:0': { data: contour } }
		},
		free() {},
	})
	return { session, calls: () => calls, inputs }
}

test('posteriors: 43844-sample windows every 36164 after 3840 zeros, middle 142 of 172 frames, trunc(L/36164·142) frames (run_inference)', async () => {
	for (let L of [1000, 36164 * 3, 200607]) {
		let x = Float32Array.from({ length: L }, (_, i) => (i + 1) / 1048576) // distinct, exact in float32
		let T = Math.trunc(L / 36164 * 142)
		for (let batch of [1, 4]) {
			let m = probe(), p = await posteriors(x, { sampleRate: 22050, batch, session: m.session })
			is(p.note.length, T, `L ${L}: ${T} frames`)
			is(m.calls(), Math.ceil(Math.ceil(T / 142) / batch), `L ${L}, batch ${batch}: model runs`)
			ok(m.inputs.every(({ dims }) => dims[1] === 43844 && dims[2] === 1), 'input [B, 43844, 1]')
			let wrong = 0
			for (let t = 0; t < T; t++) {
				let at = Math.floor(t / 142) * 36164 + (15 + t % 142) * 256 - 3840, v = at >= 0 && at < L ? x[at] : 0
				if (p.note[t][87] !== v || p.onset[t][0] !== -v || p.contour[t][263] !== 2 * v) wrong++
			}
			is(wrong, 0, `L ${L}, batch ${batch}: frames from the right window and offset`)
		}
	}
})

test('posteriors: frame times as model_frames_to_time; the last frame lands within 2 frames of the end (test_inference.py:59-64)', async () => {
	let L = 200607, p = await posteriors(new Float32Array(L), { sampleRate: 22050, session: probe().session })
	let offset = (256 / 22050) * (172 - 43844 / 256) + 0.0018
	is(p.times[171], 171 * 256 / 22050)
	is(p.times[172], 172 * 256 / 22050 - offset)
	is(p.times[786], 786 * 256 / 22050 - 4 * offset)
	ok(Math.abs(p.times[p.times.length - 1] - L / 22050) < 2 * HOP_S, `last frame ${p.times[p.times.length - 1].toFixed(3)} s, audio ${(L / 22050).toFixed(3)} s`)
})

test('input: mono, channels averaged, { channelData, sampleRate }, resampled to 22050 Hz', async () => {
	let a = Float32Array.from({ length: 50000 }, (_, i) => Math.sin(i / 7)), b = Float32Array.from({ length: 50000 }, (_, i) => Math.cos(i / 5))
	let m = probe()
	await posteriors([a, b], { sampleRate: 22050, session: m.session })
	let mean = m.inputs[0].data.subarray(3840, 3850), want = a.subarray(0, 10).map((v, i) => (v + b[i]) / 2)
	ok(mean.every((v, i) => Math.abs(v - want[i]) < 1e-7), 'channels averaged')
	let p = await posteriors({ channelData: [a], sampleRate: 44100 }, { session: probe().session })
	is(p.note.length, Math.trunc(25000 / 36164 * 142), '44.1 kHz: frames of the 25000 samples at 22.05 kHz')
})

test('input errors and empty audio', async () => {
	await rejects(() => transcribe(new Float32Array(10), {}), /sampleRate/)
	await rejects(() => transcribe('x', { sampleRate: 22050 }), /audio must be/)
	await rejects(() => posteriors(new Float32Array(50000), { sampleRate: 22050, session: async () => ({ run: async () => ({}) }) }), /StatefulPartitionedCall:1/)
	let used = false
	is(await transcribe(new Float32Array(0), { sampleRate: 22050, session: () => { used = true } }), [])
	ok(!used, 'no model run for empty audio')
})

// ------------------------------------------------ 3. the real model

// onnxruntime-node with 4 intra-op threads, in neural-runtime's Session shape (neural-runtime has
// no thread option in Node; this suite shares the machine)
async function capped(model) {
	let { default: ort } = await import('onnxruntime-node')
	let s = await ort.InferenceSession.create(typeof model === 'string' ? await fetchModel(model) : model, { intraOpNumThreads: 4 })
	return {
		inputs: s.inputNames.map(name => ({ name })),
		async run(feeds) {
			let out = await s.run(Object.fromEntries(Object.entries(feeds).map(([k, t]) => [k, new ort.Tensor(t.type, t.data, t.dims)])))
			return Object.fromEntries(Object.entries(out).map(([k, t]) => [k, { data: t.data, dims: t.dims }]))
		},
		free() { s.release() },
	}
}

// C major triad, six harmonics at 1/h, 44.1 kHz
function triad() {
	let fs = 44100, x = new Float32Array(fs * 2)
	for (let midi of [60, 64, 67]) {
		let f = 440 * 2 ** ((midi - 69) / 12)
		for (let i = Math.round(0.25 * fs); i < Math.round(1.75 * fs); i++) for (let h = 1; h <= 6; h++) x[i] += 0.1 * Math.sin(2 * Math.PI * f * h * i / fs) / h
	}
	return x
}

;(HAS_MODEL ? test : test.skip)('transcribe: C major triad through @audio/neural-runtime; in-tune notes bend 0 cents, +33.3 with upstream: true (basic-pitch#87)', MODEL_RUN, async () => {
	for (let [upstream, want] of [[false, 0], [true, 100 / 3]]) {
		let notes = await transcribe(triad(), { sampleRate: 44100, upstream })
		for (let midi of [60, 64, 67]) {
			let n = notes.find(n => n.midi === midi && Math.abs(n.time - 0.25) < 0.05)
			ok(n, `${midi} at 0.25 s`)
			if (!n) continue
			ok(Math.abs(n.time + n.duration - 1.75) < 0.1, `${midi} ends ${(n.time + n.duration).toFixed(3)} s`)
			let mid = n.bends.slice(5, -5)
			ok(mid.every(c => c === want), `${midi}${upstream ? ', upstream' : ''}: bends ${[...new Set(mid.map(c => c.toFixed(1)))]}`)
		}
	}
})

// tones at 44.1 kHz, each `dur` s from `at(j)`, phase accumulated along cents(τ); `amp(h)` weighs
// harmonic h, up to 10 kHz
function tones(midis, { dur = 0.6, gap = 0.2, amp = h => +(h === 1), cents = () => 0 } = {}) {
	let fs = 44100, at = j => 0.25 + j * (dur + gap), x = new Float32Array(Math.round(fs * (at(midis.length) + 0.25)))
	midis.forEach((m, j) => {
		let f0 = 440 * 2 ** ((m - 69) / 12), ph = 0
		for (let i = Math.round(at(j) * fs); i < Math.round((at(j) + dur) * fs); i++) {
			let tau = i / fs - at(j), f = f0 * 2 ** (cents(tau) / 1200), env = Math.min(1, tau / 0.01, (dur - tau) / 0.01)
			ph += 2 * Math.PI * f / fs
			for (let h = 1; h * f < 10000; h++) x[i] += 0.3 * env * amp(h) * Math.sin(h * ph)
		}
	})
	return { x, at }
}

// A frame may land one bin off (3 of 2,103 mid-note frames of sine, triangle and six-harmonic tones
// at MIDI 36-96 did, README, Verification): 95% of each note's frames must read the pitch
;(HAS_MODEL ? test : test.skip)('bends: in-tune sines and triangles at MIDI 36-96 read 0 cents mid-note, +33.3 with upstream: true', MODEL_RUN, async () => {
	let midis = [36, 48, 60, 72, 84, 96]
	for (let [shape, amp] of [['sine', h => +(h === 1)], ['triangle', h => h % 2 ? (-1) ** ((h - 1) / 2) / (h * h) : 0]]) {
		let { x, at } = tones(midis, { amp }), p = await posteriors(x, { sampleRate: 44100, session: capped })
		for (let [upstream, want] of [[false, 0], [true, 100 / 3]]) {
			let notes = toNotes(p, { upstream })
			midis.forEach((m, j) => {
				let n = notes.find(n => n.midi === m && Math.abs(n.time - at(j)) < 0.05)
				ok(n, `${shape} ${m} at ${at(j).toFixed(2)} s`)
				let mid = n?.bends.slice(5, -5) ?? [], right = mid.filter(c => c === want).length
				ok(mid.length && right >= 0.95 * mid.length, `${shape} ${m}${upstream ? ', upstream' : ''}: ${right} of ${mid.length} frames at ${want.toFixed(1)} (${[...new Set(mid.map(c => c.toFixed(1)))]})`)
			})
		}
	}
})

// The bends of each frame, from 50 ms after the onset to 50 ms before the end, against the vibrato
// at the frame's time: bias and RMS of the error, correlation with the vibrato
;(HAS_MODEL ? test : test.skip)('bends: a ±30-cent vibrato at 5.5 Hz (triangle, MIDI 45-81) is tracked without bias; upstream: true reads it one bin sharp', MODEL_RUN, async () => {
	let midis = [45, 57, 69, 81], dur = 1.2, cents = tau => 30 * Math.sin(2 * Math.PI * 5.5 * tau)
	let { x, at } = tones(midis, { dur, amp: h => h % 2 ? (-1) ** ((h - 1) / 2) / (h * h) : 0, cents })
	let p = await posteriors(x, { sampleRate: 44100, session: capped })
	for (let upstream of [false, true]) {
		let notes = toNotes(p, { upstream }), err = [], sb = 0, st = 0, sbt = 0
		midis.forEach((m, j) => {
			let n = notes.find(n => n.midi === m && Math.abs(n.time - at(j)) < 0.06)
			ok(n, `${m} at ${at(j).toFixed(2)} s`)
			let s = p.times.indexOf(n?.time)
			n?.bends.forEach((b, k) => {
				let tau = p.times[s + k] - at(j), c = cents(tau)
				if (tau < 0.05 || tau > dur - 0.05) return
				err.push(b - c); sb += b * b; st += c * c; sbt += b * c
			})
		})
		let bias = err.reduce((a, e) => a + e, 0) / err.length, rms = Math.sqrt(err.reduce((a, e) => a + e * e, 0) / err.length), corr = sbt / Math.sqrt(sb * st)
		console.log(`  (${upstream ? 'upstream' : 'default'}: ${err.length} frames, bias ${bias.toFixed(1)} cents, RMS ${rms.toFixed(1)}, correlation ${corr.toFixed(2)})`)
		if (upstream) ok(bias > 25 && bias < 42, `upstream bias ${bias.toFixed(1)} cents, one 33.3-cent bin`)
		else {
			ok(Math.abs(bias) < 8, `bias ${bias.toFixed(1)} cents`)
			ok(rms < 16, `RMS ${rms.toFixed(1)} cents (the vibrato itself: 21.2; 33.3-cent steps allow 9.6)`)
		}
	}
})

// scripts/reference.py outputs: <name>.json with the variants, posteriors and the audio as basic-pitch read it
let refs = existsSync(REF) ? readdirSync(REF).filter(f => f.endsWith('.json')).map(f => f.slice(0, -5)) : []
if (!refs.length) test.skip('basic-pitch references (scripts/reference.py) not found in ' + REF, () => {})
for (let name of refs) {
	let file = ext => path.join(REF, `${name}.${ext}`)
	let { rate, variants } = JSON.parse(readFileSync(file('json')))
	let py = { note: rows(f32(file('note.f32')), 88), onset: rows(f32(file('onset.f32')), 88), contour: rows(f32(file('contour.f32')), 264) }

	test(`${name}: toNotes on basic-pitch's posteriors reproduces its notes, ${variants.length} option sets`, () => {
		for (let { opts, notes } of variants) sameNotes(toNotes(py, { ...opts, upstream: true }), notes, JSON.stringify(opts))
	})

	// batch 8 runs other ORT kernels: posteriors move by up to ~6e-5, velocities with them
	;(HAS_MODEL ? test : test.skip)(`${name}: same 22.05 kHz input → posteriors within 1e-6 of Python onnxruntime (batch 1), identical notes (batch 8)`, MODEL_RUN, async () => {
		let x = f32(file('22k.f32'))
		let p = await posteriors(x, { sampleRate: 22050, batch: 1, session: capped })
		is(p.note.length, py.note.length, 'frames')
		let max = 0
		for (let k of ['note', 'onset', 'contour']) p[k].forEach((r, t) => r.forEach((v, f) => { max = Math.max(max, Math.abs(v - py[k][t][f])) }))
		ok(max < 1e-6, `max |Δ| ${max.toExponential(1)}`)
		sameNotes(await transcribe(x, { sampleRate: 22050, session: capped, upstream: true }), variants[0].notes, 'batch 8', 1e-4)
	})

	// basic-pitch resamples with librosa (soxr_hq), this package with @audio/resample-sinc (88.6 dB
	// apart on vocadito_10 at 1.2): a note on a threshold may move by a frame (README, Verification)
	;(HAS_MODEL ? test : test.skip)(`${name}: from ${rate} Hz, every basic-pitch note with the same pitch and onset and offset within one frame, and no others`, MODEL_RUN, async () => {
		let notes = await transcribe(f32(file('src.f32')), { sampleRate: rate, session: capped })
		let ref = variants[0].notes
		let matched = within => { let used = new Set(); return ref.filter(([s, e, m]) => {
			let i = notes.findIndex((n, j) => !used.has(j) && n.midi === m && Math.abs(n.time - s) <= within && Math.abs(n.time + n.duration - e) <= within)
			if (i >= 0) used.add(i)
			return i >= 0
		}).length }
		let hit = matched(HOP_S * 1.01)
		console.log(`  (${hit} of ${ref.length} basic-pitch notes within one frame, ${matched(1e-9)} identical; ${notes.length} notes here)`)
		is(hit, ref.length, `${hit} of ${ref.length} within one frame`)
		is(notes.length, ref.length, 'no other notes')
	})
}
