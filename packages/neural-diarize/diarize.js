// @audio/neural-diarize — speaker diarization: who spoke when.
//
// Pipeline: (1) @audio/vad finds speech regions, hangover-merged so short
// pauses (breaths, plosives) don't fragment a turn; (2) fixed sliding windows
// inside each region are embedded with a transformers.js speaker-verification
// model; (3) agglomerative (average-linkage) clustering on cosine distance
// groups windows by speaker; (4) window labels are smoothed and collapsed
// into segments.
//
// Model: WavLM (Chen et al. 2022, "WavLM: Large-Scale Self-Supervised
// Pre-Training for Full Stack Speech Processing", https://arxiv.org/abs/2110.13900),
// fine-tuned for speaker verification on VoxCeleb1 with an X-Vector head
// (Snyder et al. 2018, "X-Vectors: Robust DNN Embeddings for Speaker
// Recognition", https://www.danielpovey.com/files/2018_icassp_xvectors.pdf).
// Base WavLM (microsoft/unilm, MIT) — see README for the exact license chain.
// Run via @huggingface/transformers (Apache-2.0) / ONNX Runtime, the same
// transformers.js pattern as @audio/neural-asr and @audio/neural-align.
//
// WavLM's feature extractor expects 16 kHz mono input — like Whisper in
// neural-asr, the pipeline does not resample a raw Float32Array for you, so
// callers' audio is mixed to mono and resampled here first.

import sinc from '@audio/resample-sinc'
import { vad } from '@audio/vad'
import { AutoProcessor, WavLMForXVector, env } from '@huggingface/transformers'
import os from 'node:os'
import path from 'node:path'

const SR = 16000 // WavLM-base-plus-sv's fixed input rate, baked into the pretrained encoder
const IS_NODE = typeof process !== 'undefined' && !!process.versions?.node
const DEFAULT_MODEL = 'Xenova/wavlm-base-plus-sv'

// Lane-wide cache location (matches @audio/neural-runtime, which caches raw ONNX bytes
// under the same root): $AUDIO_NEURAL_CACHE/hf, else ~/.cache/audiojs/neural/hf. Set at
// module load so a bare `import` already points transformers.js away from its own
// default (buried inside node_modules, wiped on reinstall) before any model loads.
// `opts.cache` overrides per call. Browser: untouched, transformers.js's Cache API default.
function defaultCacheDir() {
	return path.join(process.env.AUDIO_NEURAL_CACHE || path.join(os.homedir(), '.cache', 'audiojs', 'neural'), 'hf')
}
if (IS_NODE) env.cacheDir = defaultCacheDir()

// Default clustering threshold, chosen from the model's own published numbers
// (Xenova/wavlm-base-plus-sv model card, https://huggingface.co/Xenova/wavlm-base-plus-sv):
// same-speaker pairs score cos_sim ≈ 0.96 / 0.96, a different-speaker pair scores ≈ 0.62.
// 0.75 sits with wide margin on both sides of that gap on real speech; see README for the
// synthetic 2-speaker measurement this package's own test adds on top.
const DEFAULT_THRESHOLD = 0.75

// Speech runs shorter than this many seconds after hangover-merge are dropped as VAD blips
// (breath, click) rather than treated as a speaker turn — also the default minimum output
// segment length (README documents both uses of the one knob).
const DEFAULT_MIN_SEGMENT = 0.5

// Seconds of padding added on each side of a detected speech run before merging —
// bridges the short silence VAD leaves around plosives/breaths without a config knob,
// same value and rationale as @audio/neural-asr's vadTrim.
const HANGOVER = 0.3

const EPS = 1e-6

// ── Input normalization (same shapes as @audio/neural-asr) ─────────────────────────────

/** Mix any channel layout down to mono (equal-weight average). Returns the input unchanged if already mono. */
export function mono(channelData) {
	if (channelData.length === 1) return channelData[0]
	let n = channelData[0].length
	let out = new Float32Array(n)
	for (let c = 0; c < channelData.length; c++) {
		let ch = channelData[c]
		for (let i = 0; i < n; i++) out[i] += ch[i] / channelData.length
	}
	return out
}

/** Resample mono data to WavLM's fixed 16 kHz input rate (no-op if already 16 kHz). */
export function to16k(data, sampleRate) {
	if (sampleRate === SR) return Float32Array.from(data)
	return sinc(data, { from: sampleRate, to: SR })
}

function resolveAudio(audio, opts) {
	if (audio instanceof Float32Array) {
		if (!(opts.sampleRate > 0)) throw new Error('neural-diarize: opts.sampleRate is required for a raw Float32Array input')
		return { data: mono([audio]), sampleRate: opts.sampleRate }
	}
	if (Array.isArray(audio)) {
		if (!audio.length || !(audio[0] instanceof Float32Array)) throw new Error('neural-diarize: array input must be Float32Array[] (one per channel)')
		if (!(opts.sampleRate > 0)) throw new Error('neural-diarize: opts.sampleRate is required for a Float32Array[] input')
		return { data: mono(audio), sampleRate: opts.sampleRate }
	}
	if (audio && audio.channelData) {
		if (!(audio.sampleRate > 0)) throw new Error('neural-diarize: {channelData, sampleRate} input requires a positive sampleRate')
		return { data: mono(audio.channelData), sampleRate: audio.sampleRate }
	}
	throw new Error('neural-diarize: audio must be Float32Array (mono), Float32Array[] (per-channel), or {channelData, sampleRate}')
}

// ── Step 1: VAD → hangover-merged speech regions ────────────────────────────────────────

function speechRegions(data16k, minSegment) {
	let { active, times, hop } = vad(data16k, { fs: SR })
	let total = data16k.length / SR
	let regions = []
	for (let i = 0; i < active.length; ) {
		if (!active[i]) { i++; continue }
		let j = i
		while (j < active.length && active[j]) j++
		let start = Math.max(0, times[i] - HANGOVER)
		let end = Math.min(total, times[j - 1] + hop / SR + HANGOVER)
		let last = regions[regions.length - 1]
		if (last && start <= last.end) last.end = Math.max(last.end, end)
		else regions.push({ start, end })
		i = j
	}
	return regions.filter(r => r.end - r.start >= minSegment)
}

// ── Step 2: fixed sliding windows inside each speech region ─────────────────────────────

function windowsFor(regions, window, hop) {
	let wins = []
	for (let r of regions) {
		let len = r.end - r.start
		if (len <= window) { wins.push({ start: r.start, end: r.end }); continue }
		let t = r.start
		while (t + window < r.end) { wins.push({ start: t, end: t + window }); t += hop }
		let last = wins[wins.length - 1]
		let tail = { start: r.end - window, end: r.end } // final window always full-length, flush with the region end
		if (!last || tail.start - last.start > EPS) wins.push(tail)
	}
	return wins
}

// ── Step 3: embeddings + clustering ──────────────────────────────────────────────────────

function cosineSim(a, b) {
	let dot = 0, na = 0, nb = 0
	for (let i = 0; i < a.length; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i] }
	let denom = Math.sqrt(na) * Math.sqrt(nb)
	return denom ? dot / denom : 0
}

/**
 * Agglomerative hierarchical clustering, average linkage (UPGMA) on cosine distance.
 * Pure JS, no model — takes embeddings straight from `diarize()`'s `embeddings` output or
 * any other source. Stops merging at `speakers` (exact count, clamped to `[1, n]`) if given,
 * else at the first merge whose distance exceeds `1 - threshold`, subject to
 * `minSpeakers`/`maxSpeakers` bounds (merges past the threshold to respect `maxSpeakers`,
 * never merges below `minSpeakers`).
 */
export function cluster(embeddings, { threshold = DEFAULT_THRESHOLD, speakers, minSpeakers = 1, maxSpeakers = 8 } = {}) {
	let n = embeddings.length
	if (n === 0) return new Int32Array(0)
	if (n === 1) return new Int32Array(1)

	let parent = Int32Array.from({ length: n }, (_, i) => i)
	let find = i => { while (parent[i] !== i) { parent[i] = parent[parent[i]]; i = parent[i] } return i }

	let size = new Float64Array(n).fill(1)
	let alive = new Uint8Array(n).fill(1)
	let D = Array.from({ length: n }, () => new Float64Array(n))
	for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
		let d = 1 - cosineSim(embeddings[i], embeddings[j])
		D[i][j] = D[j][i] = d
	}

	let targetK = speakers != null ? Math.max(1, Math.min(speakers | 0, n)) : null
	let distThresh = 1 - threshold

	while (true) {
		let ids = []
		for (let i = 0; i < n; i++) if (alive[i]) ids.push(i)
		if (ids.length <= 1) break
		if (targetK != null) { if (ids.length <= targetK) break }
		else if (ids.length <= minSpeakers) break

		let bi = -1, bj = -1, bd = Infinity
		for (let x = 0; x < ids.length; x++) for (let y = x + 1; y < ids.length; y++) {
			let i = ids[x], j = ids[y], d = D[i][j]
			if (d < bd) { bd = d; bi = i; bj = j }
		}

		if (targetK == null && ids.length <= maxSpeakers && bd > distThresh) break

		// Lance-Williams average-linkage (UPGMA) update: merge bj into bi.
		let si = size[bi], sj = size[bj]
		for (let k of ids) {
			if (k === bi || k === bj) continue
			let nd = (si * D[bi][k] + sj * D[bj][k]) / (si + sj)
			D[bi][k] = D[k][bi] = nd
		}
		size[bi] = si + sj
		alive[bj] = 0
		parent[bj] = bi
	}

	let roots = []
	for (let i = 0; i < n; i++) if (alive[i]) roots.push(i)
	let labelOf = new Map(roots.map((r, k) => [r, k]))
	let labels = new Int32Array(n)
	for (let i = 0; i < n; i++) labels[i] = labelOf.get(find(i))
	return labels
}

function computeCentroids(embeddings, labels, k) {
	let dim = embeddings[0].length
	let sums = Array.from({ length: k }, () => new Float64Array(dim))
	let counts = new Float64Array(k)
	for (let i = 0; i < embeddings.length; i++) {
		let l = labels[i]
		counts[l]++
		let e = embeddings[i]
		for (let d = 0; d < dim; d++) sums[l][d] += e[d]
	}
	return sums.map((s, l) => Float32Array.from(s, v => v / (counts[l] || 1)))
}

// ── Step 4: window labels → segments ─────────────────────────────────────────────────────

// Mode/median filter over a categorical label sequence: an isolated single-window flip
// surrounded by the same label on both sides is corrected to match its neighbours.
function medianFilter(labels) {
	let n = labels.length
	let out = labels.slice()
	for (let i = 1; i < n - 1; i++) {
		if (labels[i - 1] === labels[i + 1] && labels[i] !== labels[i - 1]) out[i] = labels[i - 1]
	}
	return out
}

let centerOf = w => (w.start + w.end) / 2
let round3 = t => Math.round(t * 1000) / 1000

// Merge any segment shorter than minSegment into its longer same-run neighbour (a run is a
// maximal contiguous span of windows with no VAD silence gap between them — a merge never
// bridges real silence into a different speech region). A tiny segment with no same-run
// neighbour (a whole region shorter than minSegment) is left as-is — see README.
function mergeShort(segments, minSegment) {
	let out = segments.slice()
	let changed = true
	while (changed) {
		changed = false
		for (let i = 0; i < out.length; i++) {
			if (out[i].end - out[i].start >= minSegment) continue
			let prev = i > 0 && out[i - 1].run === out[i].run ? out[i - 1] : null
			let next = i < out.length - 1 && out[i + 1].run === out[i].run ? out[i + 1] : null
			if (!prev && !next) continue
			let target = prev && (!next || prev.end - prev.start >= next.end - next.start) ? prev : next
			target.start = Math.min(target.start, out[i].start)
			target.end = Math.max(target.end, out[i].end)
			target.members = target.members.concat(out[i].members)
			out.splice(i, 1)
			changed = true
			break
		}
	}
	return out
}

/**
 * Collapse per-window speaker labels into segments: median-filters isolated flips, merges
 * consecutive same-speaker windows, and splits speaker changes at the midpoint between
 * window centers. `scores[i]` (optional, e.g. cosine similarity to the assigned cluster's
 * centroid) is averaged over each segment's member windows into `segment.score`. Exported
 * so this pure post-processing step is testable on synthetic label sequences, without a
 * model (see test.js).
 */
export function toSegments(labels, windows, { minSegment = DEFAULT_MIN_SEGMENT, scores } = {}) {
	let n = windows.length
	if (n === 0) return []
	if (labels.length !== n)
		throw new Error(`neural-diarize: toSegments — labels.length (${labels.length}) must equal windows.length (${n})`)

	// Split into contiguous runs first — smoothing and merging never cross a real silence
	// gap between separate VAD-detected speech regions.
	let runStarts = [0]
	for (let i = 1; i < n; i++) if (windows[i].start - windows[i - 1].end > EPS) runStarts.push(i)
	runStarts.push(n)

	let segments = []
	for (let r = 0; r < runStarts.length - 1; r++) {
		let lo = runStarts[r], hi = runStarts[r + 1]
		let runLabels = medianFilter(Array.from(labels.slice(lo, hi)))
		let groupStart = lo
		for (let i = lo + 1; i <= hi; i++) {
			if (i === hi || runLabels[i - lo] !== runLabels[groupStart - lo]) {
				let start = groupStart === lo ? windows[groupStart].start : (centerOf(windows[groupStart - 1]) + centerOf(windows[groupStart])) / 2
				let end = i === hi ? windows[i - 1].end : (centerOf(windows[i - 1]) + centerOf(windows[i])) / 2
				let members = []
				for (let k = groupStart; k < i; k++) members.push(k)
				segments.push({ start, end, speaker: runLabels[groupStart - lo], run: r, members })
				groupStart = i
			}
		}
	}

	segments = mergeShort(segments, minSegment)

	return segments.map(seg => {
		let out = { start: round3(seg.start), end: round3(seg.end), speaker: `S${seg.speaker}` }
		if (scores) out.score = round3(seg.members.reduce((s, i) => s + scores[i], 0) / seg.members.length)
		return out
	})
}

// ── toSubtitles: overlay speaker labels onto ASR/subtitle cues ──────────────────────────

/**
 * Prefix each cue's text with a WebVTT voice span (`<v S0>…</v>`, https://www.w3.org/TR/webvtt1/#webvtt-cue-voice-span)
 * naming the diarization segment it overlaps most, by time overlap. `cues` is any
 * `{start, end, text}[]` — @audio/neural-asr's `segments`/`words`, @audio/subtitle's `Cue[]`,
 * or your own. A cue with no overlapping segment (silence-only ASR artifact, or diarization
 * that ran on a shorter clip) is returned unchanged.
 */
export function toSubtitles(segments, cues) {
	return cues.map(cue => {
		let best = null, bestOverlap = 0
		for (let seg of segments) {
			let overlap = Math.min(cue.end, seg.end) - Math.max(cue.start, seg.start)
			if (overlap > bestOverlap) { bestOverlap = overlap; best = seg }
		}
		return best ? { ...cue, text: `<v ${best.speaker}>${cue.text}</v>` } : { ...cue }
	})
}

// ── Pipeline lifecycle (mirrors @audio/neural-asr's createPipeline/loadModel) ───────────

async function createModel(model, opts = {}) {
	if (opts.cache) env.cacheDir = opts.cache
	else if (IS_NODE) env.cacheDir = defaultCacheDir()
	let dtype = opts.dtype ?? (IS_NODE ? 'fp32' : 'q8') // same fp32-in-Node / q8-in-browser split as neural-asr — see its README for why
	let device = opts.device ?? 'auto'
	let [processor, xvector] = await Promise.all([
		AutoProcessor.from_pretrained(model),
		WavLMForXVector.from_pretrained(model, { dtype, device, progress_callback: opts.progress }),
	])
	return { processor, xvector }
}

async function runDiarize({ processor, xvector }, audio, opts = {}) {
	let { data: mixed, sampleRate } = resolveAudio(audio, opts)
	let data16k = to16k(mixed, sampleRate)

	let window = opts.window ?? 1.5
	let hop = opts.hop ?? 0.75
	let minSegment = opts.minSegment ?? DEFAULT_MIN_SEGMENT

	let regions = speechRegions(data16k, minSegment)
	let windows = windowsFor(regions, window, hop)

	if (!windows.length) {
		let result = { segments: [], speakers: 0 }
		if (opts.embeddings) result.embeddings = []
		return result
	}

	let embeddings = []
	for (let i = 0; i < windows.length; i++) {
		let w = windows[i]
		let s = Math.round(w.start * SR), e = Math.round(w.end * SR)
		let inputs = await processor(data16k.subarray(s, e))
		let out = await xvector(inputs)
		embeddings.push(Float32Array.from(out.embeddings.data))
		opts.progress?.({ status: 'embed', index: i, total: windows.length })
	}

	let labels = cluster(embeddings, {
		threshold: opts.threshold ?? DEFAULT_THRESHOLD,
		speakers: opts.speakers,
		minSpeakers: opts.minSpeakers ?? 1,
		maxSpeakers: opts.maxSpeakers ?? 8,
	})
	let speakers = labels.length ? Math.max(...labels) + 1 : 0

	// Relabel so S0 is whichever speaker's first window comes first chronologically —
	// cluster() itself has no notion of time, only merge order.
	let order = new Map()
	for (let l of labels) if (!order.has(l)) order.set(l, order.size)
	let relabeled = Int32Array.from(labels, l => order.get(l))

	let centroids = computeCentroids(embeddings, relabeled, speakers)
	let scores = embeddings.map((e, i) => cosineSim(e, centroids[relabeled[i]]))

	let segments = toSegments(relabeled, windows, { minSegment, scores })

	let result = { segments, speakers }
	if (opts.embeddings) result.embeddings = embeddings
	return result
}

/** diarize(audio, opts) → { segments, speakers, embeddings? } — one-shot: loads the model,
 * diarizes, frees it. For repeated calls, use `loadModel` to keep the model warm. */
export default async function diarize(audio, opts = {}) {
	let model = opts.model ?? DEFAULT_MODEL
	let m = await createModel(model, opts)
	try {
		return await runDiarize(m, audio, opts)
	} finally {
		await m.xvector.dispose?.()
	}
}

/** loadModel(model, opts) → { diarize(audio, opts), free() } — keeps the WavLM x-vector
 * model (and its ONNX Runtime session) warm across calls; call free() to release it. */
export async function loadModel(model = DEFAULT_MODEL, opts = {}) {
	let m = await createModel(model, opts)
	let freed = false
	return {
		diarize: (audio, callOpts = {}) => {
			if (freed) throw new Error('neural-diarize: diarize() called after free()')
			return runDiarize(m, audio, { ...opts, ...callOpts })
		},
		free: async () => {
			if (freed) return
			freed = true
			await m.xvector.dispose?.()
		},
	}
}
