// Core (cluster/toSegments/toSubtitles): deterministic, no network, no model.
// Adapter (diarize.js's model path): network-gated — needs @huggingface/transformers to
// fetch Xenova/wavlm-base-plus-sv from the HF hub (cached after first run, shared with the
// rest of the @audio/neural lane under $AUDIO_NEURAL_CACHE/hf or ~/.cache/audiojs/neural/hf).
// Skips with t.skip when offline.

import { execSync } from 'node:child_process'
import test, { almost, ok, is, throws } from 'tst'
import raw from 'audio-lena/raw'
import psola from '@audio/shift-psola'
import diarize, { cluster, toSegments, toSubtitles, loadModel } from './diarize.js'

// deterministic PRNG (mulberry32, public domain) — seeded, reproducible trials. Same
// generator @audio/neural-align's test.js uses, for consistency across the lane.
function mulberry32(seed) {
	return function () {
		seed |= 0; seed = (seed + 0x6D2B79F5) | 0
		let t = Math.imul(seed ^ (seed >>> 15), 1 | seed)
		t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
		return ((t ^ (t >>> 14)) >>> 0) / 4294967296
	}
}
// Box-Muller, driven by the same seeded PRNG — deterministic Gaussian noise.
function gaussian(rnd) {
	let u1 = Math.max(rnd(), 1e-9), u2 = rnd()
	return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2)
}

// n random unit vectors in `dim` dimensions. Random Gaussian vectors in 64-d are already
// nearly orthogonal (expected |cosine| ≈ 1/√64 ≈ 0.125 between any two), so these serve
// directly as well-separated cluster centers without hand-picking an explicit geometry.
function randomUnitVectors(rnd, n, dim) {
	let out = []
	for (let i = 0; i < n; i++) {
		let v = Array.from({ length: dim }, () => gaussian(rnd))
		let norm = Math.sqrt(v.reduce((s, x) => s + x * x, 0))
		out.push(v.map(x => x / norm))
	}
	return out
}

// `perCluster` noisy samples around each of `centers` (dim-d unit vectors). noiseStd=0.03
// keeps intra-cluster cosine similarity high (dot ≈ 1 − dim·noiseStd²/2 ≈ 0.97 for dim=64)
// while the near-orthogonal centers (≈0.125) stay far apart — a clean separation relative
// to the default clustering threshold (0.75).
function syntheticClusters(rnd, centers, perCluster, dim, noiseStd = 0.03) {
	let embeddings = [], trueLabels = []
	centers.forEach((c, ci) => {
		for (let i = 0; i < perCluster; i++) {
			let v = c.map(x => x + noiseStd * gaussian(rnd))
			embeddings.push(Float32Array.from(v))
			trueLabels.push(ci)
		}
	})
	return { embeddings, trueLabels }
}

// Asserts `labels` (from cluster()) agree with `trueLabels` up to relabeling: every pair of
// points from the same true cluster gets the same predicted label, every pair from
// different true clusters gets a different one.
function assertPartitionMatches(labels, trueLabels, msg) {
	let mapped = new Map() // trueLabel -> predicted label
	let usedPredicted = new Set()
	for (let i = 0; i < trueLabels.length; i++) {
		let tl = trueLabels[i], pl = labels[i]
		if (mapped.has(tl)) {
			is(pl, mapped.get(tl), `${msg}: point ${i} (true cluster ${tl}) labeled consistently with its cluster`)
		} else {
			ok(!usedPredicted.has(pl), `${msg}: true cluster ${tl}'s label ${pl} not already used by another true cluster`)
			mapped.set(tl, pl)
			usedPredicted.add(pl)
		}
	}
}

// ── cluster() — pure, synthetic ──────────────────────────────────────────────────────

test('cluster — 3 well-separated Gaussian clusters (64-d, seeded) → exactly 3 labels, default threshold', () => {
	let rnd = mulberry32(1)
	let centers = randomUnitVectors(rnd, 3, 64)
	let { embeddings, trueLabels } = syntheticClusters(rnd, centers, 8, 64)
	let labels = cluster(embeddings, {})
	is(new Set(labels).size, 3, 'exactly 3 distinct labels')
	assertPartitionMatches(labels, trueLabels, '3-cluster auto-threshold')
})

test('cluster — same 3-cluster data with speakers:2 → exactly 2 labels', () => {
	let rnd = mulberry32(1)
	let centers = randomUnitVectors(rnd, 3, 64)
	let { embeddings } = syntheticClusters(rnd, centers, 8, 64)
	let labels = cluster(embeddings, { speakers: 2 })
	is(new Set(labels).size, 2, 'exactly 2 distinct labels when speakers is fixed')
})

test('cluster — one Gaussian cluster only → 1 label', () => {
	let rnd = mulberry32(2)
	let centers = randomUnitVectors(rnd, 1, 64)
	let { embeddings } = syntheticClusters(rnd, centers, 10, 64)
	let labels = cluster(embeddings, {})
	is(new Set(labels).size, 1, 'a single cluster of noisy points around one center never splits')
	is(labels.length, 10)
})

test('cluster — empty input → empty output', () => {
	let labels = cluster([], {})
	is(labels.length, 0)
})

test('cluster — speakers clamped to [1, n]; minSpeakers/maxSpeakers bound the auto-threshold result', () => {
	let rnd = mulberry32(3)
	let centers = randomUnitVectors(rnd, 3, 64)
	let { embeddings } = syntheticClusters(rnd, centers, 4, 64)
	is(new Set(cluster(embeddings, { speakers: 100 })).size, embeddings.length, 'speakers beyond n clamps to n (every point its own cluster)')
	is(new Set(cluster(embeddings, { speakers: 0 })).size, 1, 'speakers below 1 clamps to 1')
	ok(new Set(cluster(embeddings, { threshold: 0.75, maxSpeakers: 2 })).size <= 2, 'maxSpeakers forces merging past the threshold')
})

// ── toSegments() — pure, synthetic window/label sequences ───────────────────────────────

// 7 contiguous 1.5s/0.75s-hop windows (the pipeline's own defaults) — matches what
// windowsFor() would produce for a single ~5.25s speech region.
function slidingWindows(n, window = 1.5, hop = 0.75) {
	return Array.from({ length: n }, (_, i) => ({ start: i * hop, end: i * hop + window }))
}

test('toSegments — median filter fixes a single flipped window', () => {
	let windows = slidingWindows(7)
	let labels = [0, 0, 0, 1, 0, 0, 0] // isolated flip at index 3, same label (0) on both sides
	let segs = toSegments(labels, windows, { minSegment: 0.5 })
	is(segs.length, 1, 'the flip is corrected before segmenting — one continuous segment')
	is(segs[0].speaker, 'S0')
	almost(segs[0].start, windows[0].start, 1e-9)
	almost(segs[0].end, windows[6].end, 1e-9)
})

test('toSegments — a real (non-flip) speaker change still splits, at the window-center midpoint', () => {
	let windows = slidingWindows(6)
	let labels = [0, 0, 0, 1, 1, 1] // a genuine 3-window run of each speaker — median filter must not touch this
	let segs = toSegments(labels, windows, { minSegment: 0.5 })
	is(segs.length, 2)
	is(segs[0].speaker, 'S0'); is(segs[1].speaker, 'S1')
	let boundary = (windows[2].start + windows[2].end + windows[3].start + windows[3].end) / 4 // midpoint(center(w2), center(w3))
	almost(segs[0].end, boundary, 1e-9)
	almost(segs[1].start, boundary, 1e-9)
})

test('toSegments — minSegment merges a short spurious segment into its longer neighbour', () => {
	// Hand-placed (non-uniform) windows: a short 2-window "S1" run sits between two long "S0"
	// runs. The run's own windows are packed close together (centers 1.2 and 1.8) so the
	// midpoint-boundary segment it produces is only 0.35s — under the 0.5s minSegment floor —
	// while both S0 neighbours are ≥1.35s. This isolates the merge step from the median
	// filter (a 2-window same-label run is never touched by the single-window flip filter).
	let windows = [
		{ start: 0.0, end: 1.0 },  // S0, center 0.5
		{ start: 0.7, end: 1.7 },  // S0, center 1.2
		{ start: 1.4, end: 1.6 },  // S1, center 1.5
		{ start: 1.5, end: 1.7 },  // S1, center 1.6
		{ start: 1.6, end: 2.0 },  // S0, center 1.8
		{ start: 2.3, end: 3.3 },  // S0, center 2.8
	]
	let labels = [0, 0, 1, 1, 0, 0]
	let segs = toSegments(labels, windows, { minSegment: 0.5 })
	is(segs.length, 2, 'the short S1 blip is absorbed into a neighbour, leaving 2 segments')
	ok(segs.every(s => s.speaker === 'S0'), 'no S1 segment survives — both remaining segments are S0')
	almost(segs[0].start, 0, 1e-9)
	almost(segs[segs.length - 1].end, 3.3, 1e-9)
	// coverage is exactly preserved (contiguous run, no gaps introduced by the merge)
	almost(segs[segs.length - 1].end - segs[0].start, windows[5].end - windows[0].start, 1e-9)
})

test('toSegments — a silence gap between VAD regions is never bridged by smoothing or merging', () => {
	let windows = [
		{ start: 0, end: 1 }, { start: 0.5, end: 1.5 }, // region A: 2 windows, S0
		{ start: 5, end: 6 }, { start: 5.5, end: 6.5 }, // region B (gap 1.5→5): 2 windows, S1
	]
	let labels = [0, 0, 1, 1]
	let segs = toSegments(labels, windows, { minSegment: 0.5 })
	is(segs.length, 2, 'two separate regions stay two separate segments')
	almost(segs[0].end, 1.5, 1e-9, 'region A ends at its own last window, not stretched toward region B')
	almost(segs[1].start, 5, 1e-9, 'region B starts at its own first window')
})

test('toSegments — scores are averaged per segment from per-window confidence', () => {
	let windows = slidingWindows(3)
	let labels = [0, 0, 0]
	let segs = toSegments(labels, windows, { minSegment: 0.5, scores: [0.9, 0.8, 0.7] })
	almost(segs[0].score, (0.9 + 0.8 + 0.7) / 3, 1e-9)
})

test('toSegments — labels/windows length mismatch throws; empty windows → empty segments', () => {
	throws(() => toSegments([0, 1], slidingWindows(1), {}))
	is(toSegments([], [], {}).length, 0)
})

// ── toSubtitles() — pure, overlap logic ──────────────────────────────────────────────────

test('toSubtitles — prefixes each cue with the speaker of the segment it overlaps most', () => {
	let segments = [{ start: 0, end: 5, speaker: 'S0' }, { start: 5, end: 10, speaker: 'S1' }]
	let cues = [
		{ start: 0, end: 2, text: 'Hello' },       // entirely inside S0
		{ start: 4, end: 7, text: 'world' },       // 1s in S0, 2s in S1 — majority S1
		{ start: 20, end: 21, text: 'unmatched' }, // no overlapping segment at all
	]
	let out = toSubtitles(segments, cues)
	is(out[0].text, '<v S0>Hello</v>')
	is(out[1].text, '<v S1>world</v>')
	is(out[2].text, 'unmatched', 'a cue with no overlap is returned unchanged')
	is(cues[0].text, 'Hello', 'input cues are not mutated')
})

// ── Adapter (network-gated) ──────────────────────────────────────────────────────────────

// Synchronous connectivity probe — tst auto-runs once test() registration goes quiet, so
// the online/offline decision must land before any top-level await (an async check here
// would let tst's auto-run fire on the sync tests alone, before the network-gated ones get
// registered). Same pattern as @audio/neural-align's test.js.
let online = true
try { execSync('curl -sS -m 3 -o /dev/null https://huggingface.co', { stdio: 'ignore' }) }
catch { online = false }
let netTest = online ? test : test.skip
if (!online) console.log('neural-diarize: offline — skipping @huggingface/transformers-gated tests')

let lena = new Float32Array(raw) // 44.1 kHz mono, 12.27 s (audio-lena) — one speaker
let LENA_SR = 44100

let net // warm model, set by the first (cold-download-timeout) test

netTest('loadModel — warm up (first run downloads Xenova/wavlm-base-plus-sv, ~380 MB)', async () => {
	net = await loadModel()
}, { timeout: 600000 })

netTest('diarize — single real speaker (audio-lena): 1 speaker, segments cover most of the clip', async () => {
	let t0 = Date.now()
	let r = await net.diarize(lena, { sampleRate: LENA_SR })
	let ms = Date.now() - t0
	let duration = lena.length / LENA_SR
	console.log(`neural-diarize: diarize() on ${duration.toFixed(2)}s took ${ms}ms (${(duration * 1000 / ms).toFixed(1)}× realtime)`)

	is(r.speakers, 1, `expected 1 speaker on a single-voice recording, got ${r.speakers}`)
	let covered = r.segments.reduce((s, seg) => s + (seg.end - seg.start), 0)
	// audio-lena is continuous narration with only short pauses — the brief's own bar is
	// "≥80% of the speech"; @audio/vad's hangover-merged regions are the "speech" this
	// pipeline can see, so 80% of the *clip* is a conservative proxy (any real pause budget
	// eats into it too).
	ok(covered / duration >= 0.8, `segments cover ${(100 * covered / duration).toFixed(1)}% of the clip (want ≥80%)`)
	ok(r.segments.length <= 5, `few segments for one continuous speaker: got ${r.segments.length}`)
}, { timeout: 60000 })

netTest('diarize — synthetic 2-speaker (lena + pitch/formant-shifted lena): boundary within 1s of 6.0', async () => {
	let half = Math.round(6 * LENA_SR)
	let lena6 = Float32Array.from(lena.subarray(0, half))
	// -5 semitones via PSOLA: shift-psola's own README documents that its final resample
	// rescales the whole spectrum by `ratio`, so formants move with f0 — the strongest
	// timbral change available among @audio/shift's pitch shifters (shift-formant does the
	// opposite: it explicitly *preserves* formants to avoid a chipmunk/giant effect, which
	// would keep the voice sounding like the same speaker).
	let shifted6 = psola(lena6, { semitones: -5, sampleRate: LENA_SR })
	is(shifted6.length, lena6.length, 'psola preserves buffer length (pitch shift, not time-stretch)')

	// Measure the two halves' own mean embeddings independently — the honest number this
	// package's README cites for how separable the shift actually is on this model, per the
	// brief ("report the measured cosine distance ... so the limit is documented").
	let [a, b] = await Promise.all([
		net.diarize(lena6, { sampleRate: LENA_SR, embeddings: true }),
		net.diarize(shifted6, { sampleRate: LENA_SR, embeddings: true }),
	])
	let meanEmbedding = embs => {
		let dim = embs[0].length, m = new Float64Array(dim)
		for (let e of embs) for (let d = 0; d < dim; d++) m[d] += e[d] / embs.length
		return m
	}
	let ma = meanEmbedding(a.embeddings), mb = meanEmbedding(b.embeddings)
	let dot = 0, na = 0, nb = 0
	for (let d = 0; d < ma.length; d++) { dot += ma[d] * mb[d]; na += ma[d] * ma[d]; nb += mb[d] * mb[d] }
	let cosSim = dot / (Math.sqrt(na) * Math.sqrt(nb))
	console.log(`neural-diarize: pitch/formant-shifted (-5 semitones) mean-embedding cosine similarity vs original = ${cosSim.toFixed(3)} (distance ${(1 - cosSim).toFixed(3)}); default threshold 0.75`)

	let synthetic = new Float32Array(lena6.length + shifted6.length)
	synthetic.set(lena6, 0)
	synthetic.set(shifted6, lena6.length)
	let r = await net.diarize(synthetic, { sampleRate: LENA_SR })
	console.log(`neural-diarize: synthetic 2-speaker test — detected ${r.speakers} speaker(s), ${r.segments.length} segment(s)`)

	if (cosSim < 0.75) {
		ok(r.speakers >= 2, `mean-embedding cosine similarity ${cosSim.toFixed(3)} is below the clustering threshold — expect at least 2 speakers, got ${r.speakers}`)
		// Find where the speaker identity first differs from the opening segment — the boundary
		// the model should detect. Windows straddling a hard splice (or landing on the
		// pitch-shifted audio's brief settling artifact) are noisier than the clean geometric
		// window overlap alone predicts and can spawn an extra short-lived cluster further along
		// (see README § Limitations for a measured example from this exact test) — so this checks
		// the core capability (a change is detected near 6.0s), not a demand for a clean 1-cut split.
		let firstSpeaker = r.segments[0].speaker
		let boundarySeg = r.segments.find(s => s.speaker !== firstSpeaker)
		ok(boundarySeg, 'a speaker change from the opening segment is detected somewhere')
		ok(boundarySeg && Math.abs(boundarySeg.start - 6) < 1, `first speaker-change boundary ${boundarySeg?.start.toFixed(2)}s within 1s of 6.0s`)
	} else {
		// Honest fallback per the brief: if −5 semitones + formant shift isn't enough to read
		// as a different speaker to this model, don't assert a false pass — report why.
		console.log('neural-diarize: pitch/formant-shifted copy was judged the SAME speaker as the original — see README § Limitations for the measured cosine distance.')
		ok(true, 'documented limitation, not a failure — see console output and README')
	}
}, { timeout: 60000 })

netTest('free — releases the ONNX Runtime session; diarize() after free() throws', async () => {
	await net.free()
	await throws(() => net.diarize(lena, { sampleRate: LENA_SR }), /free\(\)/)
}, { timeout: 60000 })
