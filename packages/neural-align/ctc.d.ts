/** CTC forced alignment — trellis/Viterbi over blank-interleaved targets (Graves 2006 / torchaudio `forced_align`). No ML dependency. */

export interface CtcSpan {
  /** vocab id of the token this span was aligned to (may be the blank id) */
  token: number
  /** first frame index, inclusive */
  start: number
  /** last frame index, exclusive */
  end: number
  /** exp(mean log-prob) over the span's frames — geometric-mean confidence in [0,1] */
  score: number
}

export interface CtcResult {
  /** length T — token id (incl. blank) assigned to each frame */
  path: Int32Array
  /** total log-probability of the best path */
  score: number
  /** collapsed runs of `path` (consecutive equal ids merged), including blank runs */
  spans: CtcSpan[]
}

export interface CtcOptions {
  /** blank/pad token id, default 0 */
  blank?: number
}

/**
 * Best monotonic alignment of `targets` (token ids, no blanks) against `logProbs`
 * (T×V row-major log-softmax). Throws when T < targets.length (impossible), when
 * `logProbs.length !== T*V`, when a target id is out of range or equals `blank`,
 * or when the best path has score -Infinity (no valid alignment exists).
 */
export function ctcAlign(
  logProbs: Float32Array | Float64Array,
  T: number,
  V: number,
  targets: Int32Array,
  opts?: CtcOptions
): CtcResult

export interface MergeWordsOptions {
  /** token whose text equals this string ends the current word, default '|' */
  delimiter?: string
  /** seconds per frame — required */
  frameDuration: number
}

export interface Word {
  text: string
  /** seconds */
  start: number
  /** seconds */
  end: number
  /** geometric mean of the word's token scores, in [0,1] */
  score: number
}

/**
 * Groups `spans` into words at delimiter tokens. A span whose `tokenText` lookup
 * is undefined/null/empty (typically the blank id) is dropped, not counted as a
 * boundary.
 */
export function mergeWords(
  spans: CtcSpan[],
  tokenText: ((id: number) => string) | Record<number, string> | string[],
  opts: MergeWordsOptions
): Word[]
