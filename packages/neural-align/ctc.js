// CTC forced alignment — Viterbi over the trellis of Graves 2006 ("Connectionist
// Temporal Classification"), the exact recurrence used by torchaudio's
// `torchaudio.functional.forced_align` and its CTC forced-alignment tutorial
// (https://pytorch.org/audio/stable/tutorials/ctc_forced_alignment_api_tutorial.html),
// and specified in Kürzinger et al. 2020 ("CTC-Segmentation of Large Corpora
// for German End-to-End Speech Recognition"). No ML dependency — pure DP over
// a given log-probability matrix.
//
// Extended label sequence: blank, t0, blank, t1, blank, …, t(L-1), blank — length
// S = 2L+1. trellis[t][s] = best log-probability of any path reaching extended
// state s after consuming frames 0..t, ending exactly on the label at s. Three
// transitions into (t, s): stay (t-1, s), advance (t-1, s-1), or skip the blank
// between two distinct target tokens (t-1, s-2) when label(s) is a token (not
// blank) and differs from label(s-2) — repeated tokens ("LL") must pass through
// the separating blank, so skip is disallowed there.

/**
 * @param {Float32Array|Float64Array} logProbs — T×V row-major, log-softmax per frame
 * @param {number} T — frame count
 * @param {number} V — vocab size
 * @param {Int32Array} targets — target token ids, no blanks, length L
 * @param {{blank?: number}} [opts]
 * @returns {{path: Int32Array, score: number, spans: Array<{token, start, end, score}>}}
 */
export function ctcAlign (logProbs, T, V, targets, { blank = 0 } = {}) {
  let L = targets.length
  if (T < L) throw new RangeError(`ctcAlign: T=${T} frames < L=${L} target tokens — impossible alignment (each token needs at least one frame)`)
  if (logProbs.length !== T * V) throw new RangeError(`ctcAlign: logProbs.length=${logProbs.length} does not match T×V=${T * V}`)
  if (blank < 0 || blank >= V) throw new RangeError(`ctcAlign: blank=${blank} out of vocab range [0,${V})`)
  for (let i = 0; i < L; i++) {
    let tok = targets[i]
    if (tok < 0 || tok >= V) throw new RangeError(`ctcAlign: targets[${i}]=${tok} out of vocab range [0,${V})`)
    if (tok === blank) throw new RangeError(`ctcAlign: targets[${i}] equals blank (${blank}) — targets must exclude blank`)
  }

  let S = 2 * L + 1
  // label(s): extended sequence position → token id. Even s → blank, odd s → targets[(s-1)/2].
  let label = s => (s & 1) === 0 ? blank : targets[(s - 1) >> 1]

  let NEG_INF = -Infinity
  let trellis = new Float64Array(T * S).fill(NEG_INF)
  let backptr = new Int8Array(T * S)   // 0 = stay, 1 = advance, 2 = skip

  let lp = (t, v) => logProbs[t * V + v]

  trellis[0] = lp(0, blank)
  if (S > 1) trellis[1] = lp(0, label(1))

  for (let t = 1; t < T; t++) {
    let row = t * S, prev = row - S
    for (let s = 0; s < S; s++) {
      let best = trellis[prev + s], bp = 0
      if (s >= 1) {
        let adv = trellis[prev + s - 1]
        if (adv > best) { best = adv; bp = 1 }
      }
      if (s >= 2 && label(s) !== blank && label(s) !== label(s - 2)) {
        let skip = trellis[prev + s - 2]
        if (skip > best) { best = skip; bp = 2 }
      }
      trellis[row + s] = best === NEG_INF ? NEG_INF : best + lp(t, label(s))
      backptr[row + s] = bp
    }
  }

  let lastRow = (T - 1) * S
  let endS = S - 1
  if (S > 1 && trellis[lastRow + S - 2] > trellis[lastRow + S - 1]) endS = S - 2
  let score = trellis[lastRow + endS]
  if (!Number.isFinite(score))
    throw new Error('ctcAlign: no valid alignment path (score is -Infinity) — logProbs too low-confidence for these targets, or repeated tokens without a separating blank frame')

  let path = new Int32Array(T)
  let s = endS
  for (let t = T - 1; t >= 0; t--) {
    path[t] = label(s)
    if (t === 0) break
    let bp = backptr[t * S + s]
    s -= bp === 0 ? 0 : bp === 1 ? 1 : 2
  }

  let spans = []
  let start = 0, tok = path[0]
  let flush = end => {
    let sum = 0
    for (let t = start; t < end; t++) sum += lp(t, tok)
    spans.push({ token: tok, start, end, score: Math.exp(sum / (end - start)) })
  }
  for (let t = 1; t < T; t++) {
    if (path[t] !== tok) { flush(t); start = t; tok = path[t] }
  }
  flush(T)

  return { path, score, spans }
}

/**
 * Group token spans into words at delimiter tokens. Spans whose token has no
 * entry in `tokenText` (e.g. blank/pad) are dropped silently.
 * @param {Array<{token, start, end, score}>} spans
 * @param {(id: number) => string | Array<string> | Record<number,string>} tokenText — id → character
 * @param {{delimiter?: string, frameDuration: number}} opts
 * @returns {Array<{text, start, end, score}>} start/end in seconds
 */
export function mergeWords (spans, tokenText, { delimiter = '|', frameDuration } = {}) {
  if (!(frameDuration > 0)) throw new RangeError('mergeWords: opts.frameDuration (seconds per frame) required')
  let text = typeof tokenText === 'function' ? tokenText : id => tokenText[id]

  let words = []
  let buf = []
  let flush = () => {
    if (!buf.length) return
    let sum = 0
    for (let s of buf) sum += Math.log(s.score)
    words.push({
      text: buf.map(s => text(s.token)).join(''),
      start: buf[0].start * frameDuration,
      end: buf[buf.length - 1].end * frameDuration,
      score: Math.exp(sum / buf.length),
    })
    buf = []
  }
  for (let s of spans) {
    let ch = text(s.token)
    if (ch === delimiter) { flush(); continue }
    if (ch == null || ch === '') continue   // blank/pad — not part of any word
    buf.push(s)
  }
  flush()
  return words
}
