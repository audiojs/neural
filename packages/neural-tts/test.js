// Pure helpers (splitSentences/joinWithSilence): deterministic, no network, no model.
// Model path (tts.js's default/loadModel): network-gated — needs @huggingface/transformers
// to fetch Xenova/speecht5_tts + Xenova/speecht5_hifigan from the HF hub (cached after
// first run, shared with the rest of the @audio/neural lane under $AUDIO_NEURAL_CACHE/hf
// or ~/.cache/audiojs/neural/hf). Skips with t.skip when offline.

import { execSync } from 'node:child_process'
import test, { almost, ok, is, throws, rejects } from 'tst'
import { SpeechT5HifiGan, Tensor } from '@huggingface/transformers'
import tts, { loadModel, splitSentences, joinWithSilence, models } from './tts.js'

// ── Pure helpers — no network ────────────────────────────────────────────────────────────

test('splitSentences — splits on sentence punctuation, keeps punctuation with its sentence', () => {
	let out = splitSentences('Hello world. This is a test! Are you sure? Yes.')
	is(out, ['Hello world.', 'This is a test!', 'Are you sure?', 'Yes.'])
})

test('splitSentences — a run with no sentence punctuation over maxChars splits at word boundaries, never mid-word', () => {
	let word = 'the quick brown fox jumps over the lazy dog '
	let text = word.repeat(Math.ceil(700 / word.length)) // > maxChars(600), no . ! ? anywhere
	ok(text.length > 600, 'fixture is actually over the default maxChars')
	let out = splitSentences(text, 100)
	ok(out.length >= 2, `split into ${out.length} chunks`)
	for (let chunk of out) {
		ok(chunk.length <= 100, `chunk length ${chunk.length} <= maxChars`)
		ok(!/^\s|\s$/.test(chunk), 'no leading/trailing whitespace')
	}
	is(out.join(' ').replace(/\s+/g, ' '), text.trim().replace(/\s+/g, ' '), 'rejoining the chunks recovers the original words, in order')
})

test('splitSentences — empty/whitespace-only text → []', () => {
	is(splitSentences(''), [])
	is(splitSentences('   '), [])
})

test('joinWithSilence — inserts a silent gap between chunks, none before the first or after the last', () => {
	let sr = 1000
	let a = new Float32Array(10).fill(1)
	let b = new Float32Array(20).fill(2)
	let out = joinWithSilence([a, b], sr, 100) // 100ms @ 1000Hz = 100 samples
	is(out.length, 10 + 100 + 20)
	for (let i = 0; i < 10; i++) is(out[i], 1)
	for (let i = 10; i < 110; i++) is(out[i], 0, `gap sample ${i} is silent`)
	for (let i = 110; i < 130; i++) is(out[i], 2)
})

test('joinWithSilence — single chunk passes through unchanged; empty list → empty array', () => {
	let a = new Float32Array([1, 2, 3])
	is(Array.from(joinWithSilence([a], 1000)), [1, 2, 3])
	is(joinWithSilence([], 1000).length, 0)
})

test('models — curated table shape', () => {
	ok(models.length >= 2)
	for (let m of models) {
		ok(typeof m.id === 'string' && m.id.includes('/'), m.id)
		ok(['speecht5', 'vits'].includes(m.family))
		ok(typeof m.license === 'string' && m.license.length > 0)
		ok(Number.isFinite(m.sampleRate) && m.sampleRate > 0)
	}
	ok(models.find(m => m.id === 'Xenova/speecht5_tts')?.license === 'MIT')
	ok(models.find(m => m.id === 'Xenova/mms-tts-eng')?.license === 'CC-BY-NC-4.0')
})

// ── Model tests (network-gated) ──────────────────────────────────────────────────────────

// Synchronous connectivity probe — must land before any top-level await (tst auto-runs
// once test() registration goes quiet). Same pattern as @audio/neural-align and
// @audio/neural-diarize's test.js files.
let online = true
try { execSync('curl -sS -m 3 -o /dev/null https://huggingface.co', { stdio: 'ignore' }) }
catch { online = false }
let netTest = online ? test : test.skip
if (!online) console.log('neural-tts: offline — skipping @huggingface/transformers-gated tests')

let net // warm SpeechT5 model, set by the first (cold-download-timeout) test
let TEXT = 'Hare Krishna. The quick brown fox jumps over the lazy dog.'

function measure(data) {
	let peak = 0, sum = 0
	for (let v of data) { peak = Math.max(peak, Math.abs(v)); sum += v * v }
	return { peak, rms: Math.sqrt(sum / data.length) }
}

netTest('loadModel — warm up (first run downloads Xenova/speecht5_tts + vocoder, ~630 MB)', async () => {
	net = await loadModel()
}, { timeout: 600000 })

netTest('speak — native 16kHz: non-silent, bounded, plausible duration', async () => {
	let t0 = Date.now()
	let r = await net.speak(TEXT)
	let ms = Date.now() - t0
	console.log(`neural-tts: speak() on ${TEXT.length}-char text took ${ms}ms`)

	is(r.sampleRate, 16000, "SpeechT5's native rate")
	ok(r.duration >= 1 && r.duration <= 8, `duration ${r.duration.toFixed(2)}s in [1,8]`)
	almost(r.duration, r.channelData[0].length / r.sampleRate, 1e-9, 'duration matches channelData length/sampleRate')
	let { peak, rms } = measure(r.channelData[0])
	ok(peak <= 1, `peak ${peak.toFixed(3)} <= 1`)
	ok(rms > 0.01, `rms ${rms.toFixed(4)} > 0.01 (non-silent)`)
}, { timeout: 60000 })

netTest('speak — sampleRate: 44100 resamples to the requested rate, internally consistent duration', async () => {
	// Not compared against a separate native-rate call's duration: SpeechT5's autoregressive
	// stop decision (predicted-stop-probability ≥ threshold) is sensitive enough to
	// ONNX-Runtime-CPU floating-point non-determinism that two independent generate_speech()
	// runs on identical input can stop a few mel frames apart (measured below, in the
	// "deterministic" test) — so two separate speak() calls are two separate generations, not
	// the same audio at two rates. What this checks is what actually depends on `sampleRate`:
	// the returned rate and the internal length/duration/sampleRate arithmetic.
	let r = await net.speak(TEXT, { sampleRate: 44100 })
	is(r.sampleRate, 44100)
	ok(r.duration >= 1 && r.duration <= 8, `duration ${r.duration.toFixed(2)}s in [1,8]`)
	almost(r.duration, r.channelData[0].length / 44100, 1e-9, 'duration matches channelData length/sampleRate at the requested rate')
}, { timeout: 60000 })

netTest('vocoder — deterministic given a fixed spectrogram: bit-identical across repeated calls', async () => {
	// The brief's determinism claim is about the vocoder specifically. Verified directly and
	// in isolation here: HiFi-GAN is a single forward pass (no autoregression, no internal
	// state), so feeding the *same* synthetic mel-spectrogram tensor through it repeatedly
	// must — and does — reproduce the exact same waveform, byte for byte.
	let vocoder = await SpeechT5HifiGan.from_pretrained('Xenova/speecht5_hifigan', { dtype: 'fp32' })
	let T = 50, numMel = 80
	let data = new Float32Array(T * numMel)
	for (let t = 0; t < T; t++) for (let m = 0; m < numMel; m++) data[t * numMel + m] = Math.sin(t * 0.3 + m * 0.05) * 0.5
	let spectrogram = new Tensor('float32', data, [T, numMel])

	let a = await vocoder({ spectrogram })
	let b = await vocoder({ spectrogram })
	let da = Float32Array.from(a.waveform.data), db = Float32Array.from(b.waveform.data)
	is(da.length, db.length)
	let identical = true
	for (let i = 0; i < da.length; i++) if (da[i] !== db[i]) { identical = false; break }
	ok(identical, 'vocoder output is byte-identical across two calls on the same spectrogram')
	await vocoder.dispose?.()
}, { timeout: 60000 })

netTest('speak — repeated calls on identical text stay close in duration (autoregressive stop is not bit-deterministic)', async () => {
	// The full pipeline is NOT bit-deterministic end to end: SpeechT5's decoder feeds its own
	// previous output back in at every step, so the tiny floating-point differences ONNX
	// Runtime's multi-threaded CPU execution provider can introduce between two runs of the
	// *same* MatMuls compound across iterations — confirmed by direct comparison during
	// development (two runs on this exact TEXT differed by 1024–2048 samples, ~65–130ms, in a
	// ~4.1s utterance, with near-zero sample-aligned correlation past the divergence point).
	// The vocoder itself (tested above) is not the source. This checks the pipeline stays in
	// the same ballpark — same rough length, still well-formed audio — not sample equality;
	// see README § Limitations for the measured numbers.
	let a = await net.speak(TEXT)
	let b = await net.speak(TEXT)
	let da = a.channelData[0], db = b.channelData[0]
	let deltaSamples = Math.abs(da.length - db.length)
	let deltaFrac = deltaSamples / Math.max(da.length, db.length)
	console.log(`neural-tts: two identical-input calls — lengths ${da.length} vs ${db.length} (Δ${deltaSamples} samples, ${(deltaFrac * 100).toFixed(1)}%)`)
	ok(deltaFrac < 0.1, `length delta ${(deltaFrac * 100).toFixed(1)}% < 10% of the longer run`)
	let { rms: rmsA } = measure(da), { rms: rmsB } = measure(db)
	ok(rmsA > 0.01 && rmsB > 0.01, 'both runs produced non-silent audio')
}, { timeout: 60000 })

netTest('speak — empty string throws', async () => {
	await rejects(() => net.speak(''), /non-empty/)
	await rejects(() => net.speak('   '), /non-empty/)
}, { timeout: 60000 })

netTest('speak — 2000-char text yields ≥2 chunks worth of audio, longer than a 100-char text', async () => {
	let long = 'The quick brown fox jumps over the lazy dog. '.repeat(45) // > 2000 chars, many sentences
	ok(long.length > 2000, `fixture is ${long.length} chars`)
	let short = long.slice(0, 100)
	let progressCalls = []
	let rLong = await net.speak(long, { progress: p => progressCalls.push(p) })
	let rShort = await net.speak(short)
	ok(progressCalls.filter(p => p.status === 'speak').length >= 2, `≥2 sentence-progress calls, got ${progressCalls.length}`)
	ok(rLong.duration > rShort.duration, `long (${rLong.duration.toFixed(2)}s) > short (${rShort.duration.toFixed(2)}s)`)
}, { timeout: 120000 })

netTest('free — releases the ONNX Runtime sessions; speak() after free() throws', async () => {
	await net.free()
	await throws(() => net.speak(TEXT), /free\(\)/)
}, { timeout: 60000 })

netTest('speak — Xenova/mms-tts-eng (VITS): non-silent, native 16kHz, no speaker embedding needed', async () => {
	let vits = await loadModel('Xenova/mms-tts-eng')
	let r = await vits.speak(TEXT)
	is(r.sampleRate, 16000)
	ok(r.duration >= 1 && r.duration <= 8, `duration ${r.duration.toFixed(2)}s in [1,8]`)
	let { peak, rms } = measure(r.channelData[0])
	ok(peak <= 1, `peak ${peak.toFixed(3)} <= 1`)
	ok(rms > 0.01, `rms ${rms.toFixed(4)} > 0.01 (non-silent)`)
	await vits.free()
}, { timeout: 300000 })
