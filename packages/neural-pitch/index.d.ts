/**
 * Monophonic pitch from a small transposition-equivariant network (MIT weights, plain JavaScript):
 * a pitch posterior over 288 bins, 3 a semitone from MIDI 23 (30.9 Hz), and a voicing probability per
 * frame, and a stage 1 for @audio/pitch-pyin's pitch HMM and note model.
 */

export interface PitchOptions {
  /** sample rate (Hz), default 44100 */
  fs?: number
  /** hop between frames (samples), default ≈ 5.8 ms: 256 at 44.1 kHz, as @audio/pitch-pyin */
  hopSize?: number
}

/** Frame-wise estimate, one entry per frame; frame i is centred at i·hopSize/fs. */
export interface Pitch {
  /** frame times (s) */
  times: Float64Array
  /** the posterior's peak refined over ±2 bins (Hz), on every frame */
  f0: Float32Array
  /** voicing probability, 0…1 */
  voicing: Float32Array
}

/** The network alone, frame by frame, without a pitch HMM. */
export default function pitch(data: Float32Array | Float64Array, options?: PitchOptions): Pitch

/** One frame at rate fs: its posterior (reused buffer), voicing probability and level (dBFS). */
export function frame(fs: number): (data: Float32Array | Float64Array, centre: number) => { posterior: Float32Array, voicing: number, level: number }

/**
 * Stage 1 for @audio/pitch-pyin (`track`, `notes`: option `candidates`): each frame's posterior peaks,
 * their mass weighted so pYIN's HMM adds up the network's voicing log-odds.
 */
export function candidates(frameSize: number, fs: number, minFreq: number, maxFreq: number, options?: {
  /** weight of the network's voicing log-odds in the HMM, default 1 */
  kappa?: number
}): {
  (frame: Float64Array): number
  freq: Float64Array
  prob: Float64Array
  rms: number
  frameSize: number
  lead: number
}

/** The model's input for sample rate fs: FFT size `n`, bin frequencies, dB spectrum around x[centre]. */
export function analyzer(fs: number, first?: number, bins?: number): {
  fs: number
  first: number
  bins: number
  n: number
  freq: Float64Array
  db(x: ArrayLike<number>, centre: number, out?: Float64Array): Float64Array
}

/** The network: model input (288 values, 0…1) and level (dBFS) → pitch logits (reused buffer) and voicing. */
export function network(): (input: Float32Array, level: number) => { logits: Float32Array, voicing: number }

/** Layer sizes of the shipped weights. */
export const CONFIG: { C: number, T: number, blocks: number[], D1: number, D2: number, H: number, K: number }
