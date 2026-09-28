// Polyphonic note transcription with pitch bends: Basic Pitch (Bittner, Bosch, Rubinstein,
// Meseguer-Brocal, Ewert, "A Lightweight Instrument-Agnostic Model for Polyphonic Note
// Transcription and Multipitch Estimation", ICASSP 2022), its ONNX model run through
// @audio/neural-runtime.
//
// Copyright 2022 Spotify AB
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file
// except in compliance with the License. You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the
// License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
// either express or implied. See the License for the specific language governing permissions
// and limitations under the License.
//
// Changes, 2026, audiojs: translated to JavaScript from basic_pitch/{constants,inference,
// note_creation}.py at github.com/spotify/basic-pitch commit fa5997a (windowing, unwrapping,
// note creation, onset inference, pitch bends, frame times). Resampling to 22050 Hz by
// @audio/resample-sinc instead of librosa's soxr; several windows per model run; frequency
// limits clamped to the 88 keys (Python's negative slice indices wrap around below 26.7 Hz);
// the melodia loop visits cells in sorted order instead of rescanning for the maximum (same
// order: values only ever drop to 0); note amplitude averaged in float64 (numpy: float32);
// bends returned per note in cents, centered on the contour bin the model gives the note's
// pitch, one above the bin get_pitch_bends centers on (opts.upstream keeps Basic Pitch's);
// MIDI export left out.

import resample from '@audio/resample-sinc'
import { load, tensor } from '@audio/neural-runtime'

// The ONNX export in the Python package, pinned to the commit this port follows.
export const MODEL = 'https://raw.githubusercontent.com/spotify/basic-pitch/fa5997af0a8210982619003269994a1be25eddf3/basic_pitch/saved_models/icassp_2022/nmp.onnx'

// Model I/O names: inference.py Model.predict, ONNX branch (lines 168-181).
const INPUT = 'serving_default_input_2:0'
const OUT = { note: 'StatefulPartitionedCall:1', onset: 'StatefulPartitionedCall:2', contour: 'StatefulPartitionedCall:0' }

const RATE = 22050 // AUDIO_SAMPLE_RATE, constants.py:33
const HOP = 256 // FFT_HOP, constants.py:25
const FPS = Math.floor(RATE / HOP) // ANNOTATIONS_FPS = 86, constants.py:40
const WIN_FRAMES = FPS * 2 // ANNOT_N_FRAMES = 172 (AUDIO_WINDOW_LENGTH 2 s), constants.py:38,44
const WIN = RATE * 2 - HOP // AUDIO_N_SAMPLES = 43844, constants.py:47
const OVERLAP_FRAMES = 30 // DEFAULT_OVERLAPPING_FRAMES, inference.py:190
const OVERLAP = OVERLAP_FRAMES * HOP // overlap_len = 7680, inference.py:304
const STEP = WIN - OVERLAP // hop_size = 36164, inference.py:305
const KEEP = WIN_FRAMES - OVERLAP_FRAMES // frames kept per window = 142, inference.py:278
const TRIM = OVERLAP_FRAMES / 2 // n_olap = 15 frames dropped at each window edge, inference.py:267
const KEYS = 88 // ANNOTATIONS_N_SEMITONES, constants.py:32
const BINS = KEYS * 3 // N_FREQ_BINS_CONTOURS (CONTOURS_BINS_PER_SEMITONE 3), constants.py:28,36
const MIDI_OFFSET = 21 // note_creation.py:40
const MAX_FREQ_IDX = KEYS - 1 // note_creation.py:43
const ENERGY_TOL = 11 // energy_tol, output_to_notes_polyphonic, note_creation.py:370
const MAGIC_ALIGNMENT_OFFSET = 0.0018 // note_creation.py:47
const BEND_TOL = 25 // n_bins_tolerance, get_pitch_bends, note_creation.py:183

// Defaults of inference.predict: DEFAULT_ONSET_THRESHOLD, DEFAULT_FRAME_THRESHOLD,
// DEFAULT_MINIMUM_NOTE_LENGTH_MS (inference.py:185-187).
const ONSET_THRESHOLD = 0.5, FRAME_THRESHOLD = 0.3, MIN_DURATION = 0.1277

// scipy.signal.windows.gaussian(51, std=5): exp(-n²/(2·5²)), n = -25..25 (note_creation.py:199)
const GAUSS = Float64Array.from({ length: 2 * BEND_TOL + 1 }, (_, i) => Math.exp(-((i - BEND_TOL) ** 2) / 50))

// np.round: half to even
const roundEven = x => { let r = Math.round(x); return r - x === 0.5 && r % 2 ? r - 1 : r }
// librosa.hz_to_midi, librosa.midi_to_hz
const hzToMidi = f => 12 * (Math.log2(f) - Math.log2(440)) + 69
const midiToHz = m => 440 * 2 ** ((m - 69) / 12)

const rows = (flat, width) => Array.from({ length: flat.length / width }, (_, t) => flat.subarray(t * width, (t + 1) * width))

// ------------------------------------------------------------------- input

// Mono at 22050 Hz, as librosa.load(sr=22050, mono=True) reads it: channels averaged, resampled.
function prepare(audio, opts) {
	let channels, rate = opts.sampleRate
	if (ArrayBuffer.isView(audio)) channels = [audio]
	else if (Array.isArray(audio)) channels = audio
	else if (audio?.channelData) { channels = audio.channelData; rate = audio.sampleRate ?? rate }
	else throw new TypeError('neural-transcribe: audio must be a Float32Array, Float32Array[] or { channelData, sampleRate }')
	if (!(rate > 0)) throw new TypeError('neural-transcribe: sampleRate is required (opts.sampleRate, or audio.sampleRate)')
	if (!channels.length) throw new TypeError('neural-transcribe: audio has no channels')
	let x = channels[0]
	if (channels.length > 1) {
		x = new Float32Array(x.length)
		for (let ch of channels) for (let i = 0; i < x.length; i++) x[i] += ch[i] / channels.length
	}
	return rate === RATE ? x : resample(x, { from: rate, to: RATE })
}

// Frame n's time in seconds (model_frames_to_time, note_creation.py:346-357): hop-256 frame
// times, pulled back once per 172 frames because each 2 s window holds 43844 samples, not 172·256.
function frameTimes(T) {
	let offset = (HOP / RATE) * (WIN_FRAMES - WIN / HOP) + MAGIC_ALIGNMENT_OFFSET
	return Float64Array.from({ length: T }, (_, n) => (n * HOP) / RATE - offset * Math.floor(n / WIN_FRAMES))
}

// --------------------------------------------------------------- posteriors

// posteriors(audio, opts) → { note, onset, contour, times }: the model's frame posteriors,
// unwrapped as run_inference does (inference.py:282-330). The signal is prefixed with half an
// overlap of zeros and cut into 43844-sample windows every 36164 samples; each window gives
// 172 frames, of which the middle 142 are kept.
export async function posteriors(audio, opts = {}) {
	let x = prepare(audio, opts)
	let T = Math.trunc(x.length / STEP * KEEP) // inference.py:277-279
	let flat = { note: new Float32Array(T * KEYS), onset: new Float32Array(T * KEYS), contour: new Float32Array(T * BINS) }
	if (T) {
		let session = await (opts.session ? opts.session(opts.model ?? MODEL, opts) : load(opts.model ?? MODEL, { backend: opts.device }))
		try {
			let windows = Math.ceil(T / KEEP), batch = Math.max(1, Math.floor(opts.batch ?? 8))
			let name = session.inputs?.[0]?.name ?? INPUT
			for (let w0 = 0; w0 < windows; w0 += batch) {
				let b = Math.min(batch, windows - w0), input = new Float32Array(b * WIN)
				for (let j = 0; j < b; j++) {
					let from = (w0 + j) * STEP - OVERLAP / 2, lo = Math.max(0, from), hi = Math.min(x.length, from + WIN)
					if (hi > lo) input.set(x.subarray(lo, hi), j * WIN + lo - from)
				}
				let out = await session.run({ [name]: tensor(input, [b, WIN, 1], 'float32') })
				for (let key in OUT) {
					let y = out[OUT[key]]?.data
					if (!y) throw new Error(`neural-transcribe: model output '${OUT[key]}' (${key}) missing; got ${Object.keys(out).join(', ')}`)
					let width = key === 'contour' ? BINS : KEYS, dst = flat[key]
					for (let j = 0; j < b; j++) for (let f = 0; f < KEEP; f++) {
						let t = (w0 + j) * KEEP + f
						if (t >= T) break
						dst.set(y.subarray((j * WIN_FRAMES + TRIM + f) * width, (j * WIN_FRAMES + TRIM + f + 1) * width), t * width)
					}
				}
			}
		} finally {
			session.free?.()
		}
	}
	return { note: rows(flat.note, KEYS), onset: rows(flat.onset, KEYS), contour: rows(flat.contour, BINS), times: frameTimes(T) }
}

// ------------------------------------------------------------------- notes

// toNotes(posteriors, opts) → notes: model_output_to_notes (note_creation.py:52-116) with
// output_to_notes_polyphonic (360-511), get_infered_onsets (289-311), constrain_frequency
// (314-343) and get_pitch_bends (182-219).
export function toNotes({ note, onset, contour, times }, opts = {}) {
	let T = note.length, N = T * KEYS
	let onsetThr = opts.onsetThreshold ?? ONSET_THRESHOLD, frameThr = opts.frameThreshold ?? FRAME_THRESHOLD
	if (!(frameThr >= 0)) throw new RangeError(`neural-transcribe: frameThreshold must be ≥ 0, got ${frameThr}`) // below 0 upstream's melodia loop never ends
	let minLen = roundEven((opts.minDuration ?? MIN_DURATION) * (RATE / HOP)) // predict, inference.py:469
	times ??= frameTimes(T)

	// constrain_frequency: zero keys outside [minFreq, maxFreq]
	let lo = opts.minFreq != null ? roundEven(hzToMidi(opts.minFreq) - MIDI_OFFSET) : 0
	let hi = opts.maxFreq != null ? roundEven(hzToMidi(opts.maxFreq) - MIDI_OFFSET) : KEYS
	lo = Math.min(KEYS, Math.max(0, lo)); hi = Math.min(KEYS, Math.max(0, hi))
	let frames = new Float32Array(N), on = new Float64Array(N)
	for (let t = 0; t < T; t++) for (let f = lo; f < hi; f++) { frames[t * KEYS + f] = note[t][f]; on[t * KEYS + f] = onset[t][f] }

	if (opts.inferOnsets ?? true) inferOnsets(on, frames, T)
	let energy = Float64Array.from(frames)
	let events = []
	let clear = (t, f) => { let i = t * KEYS + f; energy[i] = 0; if (f < MAX_FREQ_IDX) energy[i + 1] = 0; if (f > 0) energy[i - 1] = 0 }

	// onsets: local maxima in time (scipy.signal.argrelmax, axis 0) at or above the threshold,
	// visited backwards in time, higher keys first, as np.where(...)[::-1] orders them
	for (let t = T - 1; t >= 0; t--) for (let f = KEYS - 1; f >= 0; f--) {
		let i = t * KEYS + f, v = t > 0 && t < T - 1 && on[i] > on[i - KEYS] && on[i] > on[i + KEYS] ? on[i] : 0
		if (!(v >= onsetThr) || t >= T - 1) continue
		let e = t + 1, k = 0 // extend while the key's energy stays above the frame threshold
		for (; e < T - 1 && k < ENERGY_TOL; e++) k = energy[e * KEYS + f] < frameThr ? k + 1 : 0
		e -= k
		if (e - t <= minLen) continue
		for (let j = t; j < e; j++) clear(j, f)
		events.push([t, e, f, mean(frames, t, e, f)])
	}

	// melodia trick: grow notes from the strongest remaining energy, forwards then backwards.
	// np.argmax picks the first maximum in row-major order; cells only ever drop to 0, so the
	// candidates sorted once by (value desc, index asc) replay that order.
	if (opts.melodiaTrick ?? true) {
		let cand = []
		for (let i = 0; i < N; i++) if (energy[i] > frameThr) cand.push(i)
		cand.sort((a, b) => energy[b] - energy[a] || a - b)
		for (let c of cand) {
			if (!(energy[c] > frameThr)) continue
			let mid = Math.floor(c / KEYS), f = c % KEYS
			energy[c] = 0
			let i = mid + 1, k = 0
			for (; i < T - 1 && k < ENERGY_TOL; i++) { k = energy[i * KEYS + f] < frameThr ? k + 1 : 0; clear(i, f) }
			let end = i - 1 - k
			i = mid - 1; k = 0
			for (; i > 0 && k < ENERGY_TOL; i--) { k = energy[i * KEYS + f] < frameThr ? k + 1 : 0; clear(i, f) }
			let start = i + 1 + k
			if (end - start <= minLen) continue
			events.push([start, end, f, mean(frames, start, end, f)])
		}
	}

	return events.map(([s, e, f, amp]) => {
		let midi = f + MIDI_OFFSET
		return { time: times[s], duration: times[e] - times[s], midi, freq: midiToHz(midi), velocity: amp, bends: bends(contour, s, e, midi, opts.upstream) }
	}).sort((a, b) => a.time - b.time || a.midi - b.midi)
}

// get_infered_onsets: the smaller of the 1- and 2-frame rises of the note posterior, rescaled
// to the onsets' maximum, merged by max. A posterior that never rises divides 0 by 0: NaN
// onsets, no peaks, as in numpy.
function inferOnsets(on, frames, T) {
	let diff = new Float64Array(T * KEYS), maxOn = -Infinity, maxDiff = -Infinity
	for (let i = 0; i < T * KEYS; i++) if (on[i] > maxOn) maxOn = on[i]
	for (let t = 2; t < T; t++) for (let f = 0; f < KEYS; f++) {
		let i = t * KEYS + f, d = Math.min(frames[i] - frames[i - KEYS], frames[i] - frames[i - 2 * KEYS])
		diff[i] = d < 0 ? 0 : d
	}
	for (let i = 0; i < T * KEYS; i++) if (diff[i] > maxDiff) maxDiff = diff[i]
	for (let i = 0; i < T * KEYS; i++) on[i] = Math.max(on[i], maxOn * diff[i] / maxDiff)
}

function mean(frames, s, e, f) {
	let sum = 0
	for (let t = s; t < e; t++) sum += frames[t * KEYS + f]
	return sum / (e - s)
}

// get_pitch_bends: per frame, the contour bin within ±25 bins (±8⅓ semitones) of the note's
// pitch that peaks under a Gaussian (σ = 5 bins) weighting, in cents from the pitch. The model
// puts MIDI m at contour bin 3(m − 21) + 1 (constants.py:29-30: 27.5 Hz is "the second bin");
// get_pitch_bends centers on 3(m − 21) (midi_pitch_to_contour_bin, exact after rounding), so
// Basic Pitch reads an in-tune note as +33.3 cents (spotify/basic-pitch#87). `upstream` keeps
// its center.
function bends(contour, s, e, midi, upstream) {
	let c = 3 * (midi - MIDI_OFFSET) + (upstream ? 0 : 1)
	let from = Math.max(c - BEND_TOL, 0), to = Math.min(BINS, c + BEND_TOL + 1)
	let g = Math.max(0, BEND_TOL - c), shift = BEND_TOL - g, out = new Array(e - s)
	for (let t = s; t < e; t++) {
		let row = contour[t], best = -Infinity, arg = 0
		for (let k = 0; k < to - from; k++) { let v = row[from + k] * GAUSS[g + k]; if (v > best) { best = v; arg = k } }
		out[t - s] = (arg - shift) * 100 / 3
	}
	return out
}

// ------------------------------------------------------------------ default

// transcribe(audio, opts) → [{ time, duration, midi, freq, velocity, bends }]
export default async function transcribe(audio, opts = {}) {
	return toNotes(await posteriors(audio, opts), opts)
}
