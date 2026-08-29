// Core (ctc.js): deterministic, no network, no model.
// Adapter (align.js): network-gated — needs @huggingface/transformers to fetch
// Xenova/wav2vec2-base-960h from the HF hub (cached after first run). Skips with
// t.skip when offline.
//
// References:
// - Graves et al. 2006, "Connectionist Temporal Classification" — the trellis/DP.
// - torchaudio's forced_align / CTC forced-alignment tutorial — same recurrence
//   (https://pytorch.org/audio/stable/tutorials/ctc_forced_alignment_api_tutorial.html).
// - Kürzinger et al. 2020, "CTC-Segmentation of Large Corpora for German
//   End-to-End Speech Recognition" — the alignment-quality framing this atom follows.

import { execSync } from 'node:child_process'
import test, { almost, ok, is, throws } from 'tst'
import raw from 'audio-lena/raw'
import { ctcAlign, mergeWords } from './ctc.js'
import align, { loadModel } from './align.js'

// deterministic PRNG (mulberry32, public domain) — seeded, reproducible trials
function mulberry32 (seed) {
  return function () {
    seed |= 0; seed = (seed + 0x6D2B79F5) | 0
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

// ── Core ─────────────────────────────────────────────────────────────────

test('ctcAlign — hand-made 6×4 unambiguous path: exact spans and score', () => {
  // vocab: 0=blank, 1=A, 2=B, 3=C. targets "AB" → extended [blank,A,blank,B,blank], S=5.
  // Designed state sequence per frame: 0,1,1,2,3,4 (labels: blank,A,A,blank,B,blank) —
  // a valid monotonic path (stay at s=1, advance elsewhere). Each frame's distribution
  // peaks (p=0.9) exactly on that label, with the remaining 0.1 split evenly over the
  // other 3 symbols (p=0.1/3 ≈ 0.0333) — a 27× margin, so this is the only path any
  // frame can win with more than a rounding error's worth of probability.
  let V = 4
  let peak = Math.log(0.9), rest = Math.log(0.1 / 3)
  let frameLabel = [0, 1, 1, 0, 2, 0]
  let logProbs = new Float64Array(6 * V)
  for (let t = 0; t < 6; t++) for (let v = 0; v < V; v++) logProbs[t * V + v] = v === frameLabel[t] ? peak : rest
  let targets = Int32Array.from([1, 2])   // A, B

  let { path, score, spans } = ctcAlign(logProbs, 6, V, targets, { blank: 0 })

  is(Array.from(path).join(','), frameLabel.join(','), 'path matches the designed labels')
  almost(score, 6 * peak, 1e-9, 'score = sum of the 6 peak log-probs')
  let expect = [
    { token: 0, start: 0, end: 1 },
    { token: 1, start: 1, end: 3 },
    { token: 0, start: 3, end: 4 },
    { token: 2, start: 4, end: 5 },
    { token: 0, start: 5, end: 6 },
  ]
  is(spans.length, expect.length, 'span count')
  spans.forEach((s, i) => {
    is(s.token, expect[i].token, `span ${i} token`)
    is(s.start, expect[i].start, `span ${i} start`)
    is(s.end, expect[i].end, `span ${i} end`)
    almost(s.score, 0.9, 1e-9, `span ${i} score = peak probability`)
  })
})

test('ctcAlign — property: random peaked paths (blanks + repeats) recovered exactly; repeats separated by a blank', () => {
  let rnd = mulberry32(12345)
  let V = 5   // blank=0, tokens 1..4
  let repeatsChecked = 0

  for (let trial = 0; trial < 25; trial++) {
    let L = 2 + Math.floor(rnd() * 5)                      // 2..6 target tokens
    let targets = Int32Array.from({ length: L }, () => 1 + Math.floor(rnd() * 4))
    let S = 2 * L + 1
    let label = s => (s % 2 === 0) ? 0 : targets[(s - 1) >> 1]

    // Visit every extended state s=0..S-1 in order for a random 1..3-frame dwell —
    // always a valid path (every transition is a plain advance), and it necessarily
    // passes through the blank between any two equal adjacent targets.
    let frameLabels = []
    for (let s = 0; s < S; s++) {
      let dwell = 1 + Math.floor(rnd() * 3)
      for (let d = 0; d < dwell; d++) frameLabels.push(label(s))
    }
    let T = frameLabels.length

    // Peak 0.9 on the designed label; remaining 0.1 split unevenly (seeded noise) over
    // the other symbols — never enough to unseat the peak.
    let logProbs = new Float64Array(T * V)
    for (let t = 0; t < T; t++) {
      let lbl = frameLabels[t]
      let w = Array.from({ length: V }, () => rnd())
      w[lbl] = 0
      let sum = w.reduce((a, b) => a + b, 0) || 1
      for (let v = 0; v < V; v++) logProbs[t * V + v] = Math.log(v === lbl ? 0.9 : 0.1 * w[v] / sum)
    }

    let { spans } = ctcAlign(logProbs, T, V, targets, { blank: 0 })
    let recovered = spans.filter(s => s.token !== 0).map(s => s.token)
    is(recovered.join(','), Array.from(targets).join(','), `trial ${trial}: exact token recovery`)

    for (let i = 0; i < L - 1; i++) {
      if (targets[i] !== targets[i + 1]) continue
      repeatsChecked++
      let nonBlank = spans.filter(s => s.token !== 0)
      let a = nonBlank[i], b = nonBlank[i + 1]
      let blankBetween = spans.find(s => s.token === 0 && s.start === a.end && s.end === b.start)
      ok(blankBetween, `trial ${trial}: repeated token at ${i} is separated by a blank span`)
    }
  }
  ok(repeatsChecked > 0, 'at least one adjacent-repeat case was exercised across the 25 trials')
})

test('ctcAlign — guards: T<L, V mismatch, no valid path all throw', () => {
  throws(() => ctcAlign(new Float64Array(3 * 4), 3, 4, Int32Array.from([1, 2, 3, 1, 2])), 'T=3 < L=5')
  throws(() => ctcAlign(new Float64Array(6 * 4), 6, 5, Int32Array.from([1, 2])), 'logProbs.length (24) !== T·V (30)')

  // every frame's mass is entirely on blank (p=1, others p=0) — no path can emit
  // targets [1,2] at all, so the best path's score is -Infinity.
  let allBlank = new Float64Array(6 * 4)
  for (let t = 0; t < 6; t++) { allBlank[t * 4 + 0] = 0; allBlank[t * 4 + 1] = allBlank[t * 4 + 2] = allBlank[t * 4 + 3] = -Infinity }
  throws(() => ctcAlign(allBlank, 6, 4, Int32Array.from([1, 2])), 'all-blank input has no valid alignment')
})

test('mergeWords — spans of "HELLO|WORLD" at frameDuration 0.02 → two words, correct seconds', () => {
  let tokenText = { 0: '', 1: 'H', 2: 'E', 3: 'L', 4: 'O', 5: 'W', 6: 'R', 7: 'D', 8: '|' }
  let spans = [
    { token: 1, start: 0, end: 1, score: 0.9 }, { token: 2, start: 1, end: 2, score: 0.9 },
    { token: 3, start: 2, end: 3, score: 0.9 }, { token: 3, start: 3, end: 4, score: 0.9 },
    { token: 4, start: 4, end: 5, score: 0.9 },
    { token: 8, start: 5, end: 6, score: 1 },   // delimiter
    { token: 5, start: 6, end: 7, score: 0.8 }, { token: 4, start: 7, end: 8, score: 0.8 },
    { token: 6, start: 8, end: 9, score: 0.8 }, { token: 3, start: 9, end: 10, score: 0.8 },
    { token: 7, start: 10, end: 11, score: 0.8 },
  ]
  let words = mergeWords(spans, tokenText, { delimiter: '|', frameDuration: 0.02 })
  is(words.length, 2)
  is(words[0].text, 'HELLO'); is(words[1].text, 'WORLD')
  almost(words[0].start, 0, 1e-9); almost(words[0].end, 0.1, 1e-9)
  almost(words[1].start, 0.12, 1e-9); almost(words[1].end, 0.22, 1e-9)
  almost(words[0].score, 0.9, 1e-9, 'geometric mean of 5×0.9')
  almost(words[1].score, 0.8, 1e-9, 'geometric mean of 5×0.8')
})

// ── Adapter (network-gated) ─────────────────────────────────────────────

// Synchronous connectivity probe — tst auto-runs once test() registration goes
// quiet, so the online/offline decision must land before any top-level await
// (an async check here would let tst's auto-run fire on the sync tests alone,
// before the network-gated ones get registered).
let online = true
try { execSync('curl -sS -m 3 -o /dev/null https://huggingface.co', { stdio: 'ignore' }) }
catch { online = false }
let netTest = online ? test : test.skip
if (!online) console.log('neural-align: offline — skipping @huggingface/transformers-gated tests')

let lena = new Float32Array(raw)          // 44.1 kHz mono, 12.27 s (audio-lena)
let LENA_DURATION = lena.length / 44100

// Xenova/wav2vec2-base-960h has no ground truth for this fixture (audio-lena is
// German speech; the model is English-only), so per the atom brief we use the
// model's own greedy CTC decode as the reference transcript: forced-aligning a
// transcript the model itself produced is a self-consistent test of the alignment
// mechanics, independent of transcription accuracy.
// Obtained 2026-08-28 via: argmax per frame → collapse repeats → drop blanks →
// map ids to chars → replace '|' with a space.
const REFERENCE = 'ALL LAREDEDULIDI D RINA DULIGA DE RINA ULIGALL E RIA DULIGAL O LAREDE DULIRI DERINA ULIA DE IA TLIGO HA PANI AL'

function assertMonotonic (words, msg) {
  for (let i = 1; i < words.length; i++) ok(words[i].start >= words[i - 1].end, `${msg}: word ${i} does not overlap word ${i - 1}`)
}

// Warm-up: loads (and, on a cold cache, downloads ~360 MB for) the default model
// once via loadModel() — every later test reuses this same (model, dtype, device,
// cache) key from align.js's module-level cache, so only this one test needs a
// timeout long enough to cover a first-run download on a normal connection.
netTest('loadModel — warm-up + logits() exposes the raw log-prob matrix and wav2vec2 frame duration', async () => {
  let t0 = Date.now()
  let net = await loadModel('Xenova/wav2vec2-base-960h')
  console.log(`neural-align: loadModel() took ${Date.now() - t0}ms (cold = downloads the model; warm cache = local load only)`)

  let { logProbs, T, V, frameDuration, vocab } = await net.logits(lena, { sampleRate: 44100 })
  is(logProbs.length, T * V)
  almost(frameDuration, 0.02, 1e-9, 'wav2vec2 conv-stem stride is 320 samples @ 16 kHz')
  is(vocab.length, V)
  // every row is a proper log-softmax: probabilities sum to ~1
  let row0 = 0
  for (let v = 0; v < V; v++) row0 += Math.exp(logProbs[v])
  almost(row0, 1, 1e-4, 'frame 0 is a normalized distribution')
}, { timeout: 600000 })   // covers a cold ~360 MB download on a normal connection

netTest('align — word count, monotonic timestamps, confidence, in-bounds', async () => {
  let t0 = Date.now()
  let r = await align(lena, REFERENCE, { sampleRate: 44100 })
  let ms = Date.now() - t0
  console.log(`neural-align: align() on ${LENA_DURATION.toFixed(2)}s took ${ms}ms (${(LENA_DURATION * 1000 / ms).toFixed(1)}× realtime)`)

  is(r.words.length, REFERENCE.split(' ').length, 'one word per space-separated token in the reference')
  assertMonotonic(r.words, 'reference transcript')
  ok(r.words[0].start >= 0 && r.words[r.words.length - 1].end <= LENA_DURATION, 'all timestamps within [0, duration]')
  ok(r.words[0].start > 0.05, `first word starts after warmup: ${r.words[0].start.toFixed(3)}s`)
  ok(r.words[r.words.length - 1].end < LENA_DURATION, `last word ends before audio end: ${r.words[r.words.length - 1].end.toFixed(3)}s < ${LENA_DURATION.toFixed(3)}s`)
  ok(r.score > 0.3, `mean word score is a useful confidence signal: ${r.score.toFixed(3)}`)
}, { timeout: 60000 })

netTest('align — normalization: lowercase + punctuation gives identical timestamps', async () => {
  let r1 = await align(lena, REFERENCE, { sampleRate: 44100 })
  let noisy = REFERENCE.toLowerCase().split(' ').join(', ') + '.'
  let r2 = await align(lena, noisy, { sampleRate: 44100 })
  is(r2.words.length, r1.words.length, 'same word count')
  r1.words.forEach((w, i) => {
    is(r2.words[i].start, w.start, `word ${i} start unchanged`)
    is(r2.words[i].end, w.end, `word ${i} end unchanged`)
  })
}, { timeout: 60000 })

netTest('align — wrong transcript: mean score drops to less than half', async () => {
  let correct = await align(lena, REFERENCE, { sampleRate: 44100 })
  let wrong = Array(3).fill('the quick brown fox jumps over the lazy dog').join(' ')
  let r = await align(lena, wrong, { sampleRate: 44100 })
  ok(r.score < correct.score / 2, `wrong-transcript score ${r.score.toFixed(3)} < half of correct ${correct.score.toFixed(3)}`)
}, { timeout: 60000 })

netTest('align — chunked long audio: 3× lena + 3× text → 3× words, monotonic across chunk boundaries', async () => {
  let long = new Float32Array(lena.length * 3)
  long.set(lena, 0); long.set(lena, lena.length); long.set(lena, lena.length * 2)
  let longText = [REFERENCE, REFERENCE, REFERENCE].join(' ')

  let r = await align(long, longText, { sampleRate: 44100, chunk: 20 })
  is(r.words.length, longText.split(' ').length, '3× the reference word count')
  assertMonotonic(r.words, 'concatenated 3×lena')
  ok(r.words[r.words.length - 1].end < long.length / 44100, 'last word ends before the concatenated audio ends')
}, { timeout: 60000 })
