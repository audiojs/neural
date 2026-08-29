// tst suite for @audio/neural-asr.
//
// Network-gated tests download real Whisper weights and run real inference — set
// AUDIO_NEURAL_OFFLINE=1 to force-skip them, or they self-skip when huggingface.co isn't
// reachable. Offline-safe tests (mono/to16k/toSegments unit tests) always run.
//
// tst's default auto-run scheduler polls "has the registered test count stopped changing"
// and force-runs whatever's registered once it looks stable — which fires immediately
// during any `await` between two `test()` calls (an in-progress model download looks
// identical to "no more tests coming"). Manual mode sidesteps that: register everything
// (interleaving whatever async setup we want), then call test.run() ourselves.
import test, { ok, is, almost, rejects } from 'tst'
test.manual = true

import transcribe, { loadModel, mono, to16k, toSegments, models } from './asr.js'
import rawBuf from 'audio-lena/raw'

const lena = new Float32Array(rawBuf)   // 44.1 kHz mono, 12.27 s (audio-lena/raw is an ArrayBuffer of f32le samples)
const LENA_SR = 44100

// ─────────────────────────────────────────────────────────────────────────────
// Offline-safe: synthetic unit tests for the chunk→segment mapping and audio helpers
// ─────────────────────────────────────────────────────────────────────────────

test('mono — equal-weight average across channels; passthrough when already mono', () => {
	let a = Float32Array.from([1, 0, -1, 0.5])
	let solo = mono([a])
	is(solo, a, 'single channel returned as-is, not copied element-wise')

	let l = Float32Array.from([1, 1, 1])
	let r = Float32Array.from([-1, 0, 1])
	let m = mono([l, r])
	almost(m[0], 0, 1e-9); almost(m[1], 0.5, 1e-9); almost(m[2], 1, 1e-9)
})

test('to16k — resamples to 16 kHz; no-op (copy) at 16 kHz', () => {
	let at16k = Float32Array.from({ length: 1600 }, (_, i) => Math.sin(i / 10))
	let out = to16k(at16k, 16000)
	is(out.length, at16k.length)
	is(out === at16k, false, 'returns a copy, not the same buffer')
	for (let i = 0; i < at16k.length; i++) is(out[i], at16k[i])

	let at44k = Float32Array.from({ length: 4410 }, (_, i) => Math.sin(2 * Math.PI * 440 * i / 44100))
	let down = to16k(at44k, 44100)
	almost(down.length, 1600, 2, 'resampled length tracks the rate ratio (44100→16000, 1s)')
})

test('toSegments — maps {text,timestamp} chunks to {start,end,text}; clamps a null end to duration', () => {
	let chunks = [
		{ text: ' Hello', timestamp: [0, 1.2] },
		{ text: ' world.', timestamp: [1.2, null] },
	]
	let segs = toSegments(chunks, 3.5)
	is(segs.length, 2)
	is(segs[0], { start: 0, end: 1.2, text: 'Hello' })
	is(segs[1], { start: 1.2, end: 3.5, text: 'world.' }, 'null end clamps to the given duration')
})

test('toSegments — rounds to 3 decimals; empty/undefined chunks → []', () => {
	let segs = toSegments([{ text: ' x', timestamp: [0.123456, 0.789999] }], 1)
	almost(segs[0].start, 0.123, 1e-9)
	almost(segs[0].end, 0.79, 1e-9)
	is(toSegments(undefined, 1), [])
	is(toSegments([], 1), [])
})

test('models — curated table shape', () => {
	ok(models.length > 0)
	for (let m of models) {
		ok(typeof m.id === 'string' && m.id.startsWith('onnx-community/'), m.id)
		ok(typeof m.params === 'string')
		ok(m.languages === 'en' || m.languages === 'multi')
		ok(Number.isFinite(m.size) && m.size > 0)
	}
})

test('transcribe — rejects loudly on ambiguous audio input (before touching the network)', async () => {
	// checkAudio() runs synchronously before any model load, so these reject immediately —
	// no download, no timeout needed — regardless of network availability.
	await rejects(() => transcribe(new Float32Array(10), {}), /sampleRate/, 'missing sampleRate for raw Float32Array')
	await rejects(() => transcribe('not audio', { sampleRate: 16000 }), /must be Float32Array/)
	await rejects(() => transcribe([], { sampleRate: 16000 }), /Float32Array\[\]/, 'empty channel array')
})

// ─────────────────────────────────────────────────────────────────────────────
// Network-gated: real Whisper weights, real inference.
//
// Model/content note: audio-lena/raw turns out to be a short German film-audio clip (wav
// metadata: artist "Lena Stolze", album "Das schreckliche Mädchen" — a real 1990 German
// film) — not the continuous English "reading a passage" a package name might suggest.
// `onnx-community/whisper-tiny.en` (English-only, the size class the brief for this suite
// suggested) produces a garbage repeated-token loop on it: wrong language entirely for
// an English-only model. `language: 'german'` on a multilingual model is required for a
// meaningful transcript; `onnx-community/whisper-tiny`/`whisper-base` still loop-hallucinate
// on this clip even with the language forced (verified) — root cause: transformers.js's ASR
// pipeline has no anti-repetition generation defaults (see README — Limitations), which this
// package works around internally via `no_repeat_ngram_size: 3`. With that fix,
// `Xenova/whisper-tiny` (same architecture/size as onnx-community's, 42 MB, but exported
// WITH cross-attention outputs — required for `timestamps: 'word'`, see README) gives a
// short but stable, real transcript: "Halle, Halle!" (verified: whisper-base and
// whisper-small on the SAME resampled audio produce longer but strictly worse gibberish —
// bigger did not mean better here, so tiny is not just the fast choice, it's the correct one).
// This is genuinely a short, hard clip (~2s of clear speech in 12.27s) — not a bug in this
// package; the assertions below match the observed, verified content rather than assume it.
// ─────────────────────────────────────────────────────────────────────────────

const MODEL = 'Xenova/whisper-tiny'
const MODEL_OPTS = { dtype: 'q8', device: 'cpu', language: 'german', task: 'transcribe' }
const COLD_TIMEOUT = 600_000   // covers a cold ~42 MB download + ONNX session init from scratch
const WARM_TIMEOUT = 60_000    // generous margin over the ~0.3-2s measured on a warm pipeline

let offline = !!process.env.AUDIO_NEURAL_OFFLINE
if (!offline) {
	try {
		let res = await fetch('https://huggingface.co', { method: 'HEAD', signal: AbortSignal.timeout(8000) })
		offline = !res.ok
	} catch { offline = true }
}
let netTest = offline ? test.skip : test
if (offline) console.log('  (network tests skipped: ' + (process.env.AUDIO_NEURAL_OFFLINE ? 'AUDIO_NEURAL_OFFLINE set' : 'huggingface.co unreachable') + ')')

let warm = null   // set by the first network test; reused (and freed) by the rest

netTest('asr — loadModel: cold load + real transcript on audio-lena (44.1 kHz → resampled)', async () => {
	let t0 = Date.now()
	warm = await loadModel(MODEL, MODEL_OPTS)
	let result = await warm.transcribe(lena, { sampleRate: LENA_SR, timestamps: 'segment' })
	console.log(`  loadModel+transcribe: ${((Date.now() - t0) / 1000).toFixed(1)}s — text: ${JSON.stringify(result.text)}`)

	ok(result.text.trim().length > 0, 'non-empty transcript')
	ok(/halle/i.test(result.text), `contains the observed word "Halle" — got ${JSON.stringify(result.text)}`)
	is(result.language, 'german')
	is(result.segments.length, 1)
	is(result.segments[0].start, 0)
	ok(result.segments[0].end > 0 && result.segments[0].end <= 12.5, `segment end within [0, 12.5]: ${result.segments[0].end}`)
	is(result.text, result.segments.map(s => s.text).join(' '), 'text is the join of segment texts')
}, { timeout: COLD_TIMEOUT })

netTest('asr — segments monotonic and within [0, 12.5] on the full 12.27s clip', async () => {
	let { segments } = await warm.transcribe(lena, { sampleRate: LENA_SR, timestamps: 'segment' })
	let prevEnd = 0
	for (let s of segments) {
		ok(s.start >= 0 && s.end <= 12.5, `segment within bounds: ${JSON.stringify(s)}`)
		ok(s.start >= prevEnd - 1e-9, `segment start monotonic: ${s.start} >= ${prevEnd}`)
		ok(s.end >= s.start, 'segment end >= start')
		prevEnd = s.end
	}
}, { timeout: WARM_TIMEOUT })

netTest('asr — word timestamps: monotonic, count matches text, matches segment-level content', async () => {
	let { text, words, segments } = await warm.transcribe(lena, { sampleRate: LENA_SR, timestamps: 'word' })
	ok(words.length > 0, 'at least one word')
	let wordCountFromText = text.trim().split(/\s+/).filter(Boolean).length
	// ±20% per the suite convention for short transcripts; with only a handful of words the
	// absolute slack is tiny anyway (round() below matches the brief's "≈ ±20%" for this count)
	almost(words.length, wordCountFromText, Math.max(1, Math.round(wordCountFromText * 0.2)), 'words.length ≈ word count of text')
	let prevEnd = -1
	for (let w of words) {
		ok(w.end >= w.start, 'word end >= start')
		ok(w.start >= prevEnd - 1e-9, `word start monotonic: ${w.start} >= ${prevEnd}`)
		prevEnd = w.start
	}
	ok(segments.length > 0, 'segments reconstructed from words')
	is(segments[0].start, words[0].start)
}, { timeout: WARM_TIMEOUT })

netTest('asr — loadModel reuse gives identical output twice (deterministic greedy decoding)', async () => {
	let a = await warm.transcribe(lena, { sampleRate: LENA_SR, timestamps: 'segment' })
	let b = await warm.transcribe(lena, { sampleRate: LENA_SR, timestamps: 'segment' })
	is(a.text, b.text)
	is(a.segments, b.segments)
}, { timeout: WARM_TIMEOUT })

netTest('asr — {channelData, sampleRate} input form matches the Float32Array form', async () => {
	let a = await warm.transcribe(lena, { sampleRate: LENA_SR })
	let b = await warm.transcribe({ channelData: [lena], sampleRate: LENA_SR }, {})
	is(a.text, b.text)
}, { timeout: WARM_TIMEOUT })

netTest('asr — stereo input mixes down to the same result as mono', async () => {
	let a = await warm.transcribe(lena, { sampleRate: LENA_SR })
	let b = await warm.transcribe([lena, lena], { sampleRate: LENA_SR })
	is(a.text, b.text, 'two identical channels average back to the original signal')
}, { timeout: WARM_TIMEOUT })

netTest('asr — 1s digital silence: Whisper hallucinates non-empty text (documented, not a bug here)', async () => {
	let silence = new Float32Array(16000)   // already 16 kHz — exercises the sampleRate===SR no-resample path too
	let { text, segments } = await warm.transcribe(silence, { sampleRate: 16000 })
	console.log(`  silence transcript: ${JSON.stringify(text)}`)
	// Bounded, not empty: this documents the hallucination rather than asserting it away.
	// A real regression (e.g. the no_repeat_ngram_size fix breaking) would blow well past this.
	ok(text.trim().length < 200, `hallucination stays bounded (no infinite loop): ${text.length} chars`)
	is(segments[0].start, 0)
}, { timeout: WARM_TIMEOUT })

netTest('asr — vad:true on the same silence skips inference entirely: empty text, zero segments', async () => {
	let { text, segments } = await warm.transcribe(new Float32Array(16000), { sampleRate: 16000, vad: true })
	is(text, '')
	is(segments, [])
}, { timeout: WARM_TIMEOUT })

netTest('asr — vad:true on real audio remaps timestamps back to the original timeline', async () => {
	let { text, segments } = await warm.transcribe(lena, { sampleRate: LENA_SR, vad: true, timestamps: 'segment' })
	ok(/halle/i.test(text), `VAD-trimmed transcript still contains "Halle": ${JSON.stringify(text)}`)
	for (let s of segments) ok(s.start >= 0 && s.end <= 12.5, `remapped segment within original bounds: ${JSON.stringify(s)}`)
}, { timeout: WARM_TIMEOUT })

netTest('asr — opts.cues without @audio/subtitle installed throws a clear, actionable error', async () => {
	await rejects(() => warm.transcribe(lena, { sampleRate: LENA_SR, cues: true }), /@audio\/subtitle/)
}, { timeout: WARM_TIMEOUT })

netTest('asr — free() is idempotent; transcribe() after free() throws', async () => {
	await warm.free()
	await warm.free()   // must not throw a second time
	await rejects(() => warm.transcribe(lena, { sampleRate: LENA_SR }), /after free/)
}, { timeout: WARM_TIMEOUT })

test.run().then(async state => {
	if (warm) await warm.free().catch(() => {})
	// Manual mode (required above to avoid the auto-run scheduler's registration race) skips
	// tst's own process.exit() at the end of run() — and onnxruntime-node's native thread pool
	// keeps the event loop alive indefinitely after a session is disposed, so without an
	// explicit exit here the process never terminates on its own once the model was loaded.
	process.exit(state.failed.length ? 1 : 0)
})
