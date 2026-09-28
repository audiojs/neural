/**
 * Polyphonic note transcription with pitch bends: Basic Pitch (Bittner, Bosch, Rubinstein,
 * Meseguer-Brocal, Ewert, ICASSP 2022), its ONNX model run through @audio/neural-runtime.
 */

/** Mono samples (with opts.sampleRate), one array per channel (averaged), or a decode() result. */
export type AudioInput = Float32Array | Float32Array[] | { channelData: Float32Array[]; sampleRate: number }

export interface Note {
	/** onset, seconds */
	time: number
	/** seconds */
	duration: number
	/** MIDI note number, 21..108 */
	midi: number
	/** Hz of `midi`, equal temperament at A4 = 440 Hz */
	freq: number
	/** 0..1: mean note posterior over the note (Basic Pitch's amplitude; its MIDI velocity is round(127 · velocity)) */
	velocity: number
	/**
	 * Cents from `midi`, one per model frame of the note (~11.6 ms), in steps of 100/3: Basic Pitch's
	 * 1/3-semitone contour bins. Basic Pitch spaces them evenly from `time` to `time + duration`.
	 * An in-tune note reads 0; with `upstream: true`, +33.3, as in Basic Pitch (README, Pipeline).
	 */
	bends: number[]
}

/** Frame posteriors, one row per model frame. */
export interface Posteriors {
	/** onset posterior per frame: 88 keys, MIDI 21 + index */
	onset: Float32Array[]
	/** note (frame) posterior per frame: 88 keys */
	note: Float32Array[]
	/** pitch contour posterior per frame: 264 bins, 3 per semitone; MIDI m peaks at bin 3(m - 21) + 1 (27.5 Hz is the second bin) */
	contour: Float32Array[]
	/** frame times, seconds (Basic Pitch's model_frames_to_time) */
	times: Float64Array
}

/** A neural-runtime-shaped session: enough of it to drive posteriors() with a custom ORT setup or a test double. */
export interface Session {
	run(feeds: Record<string, { data: Float32Array; dims: number[]; type: string }>): Promise<Record<string, { data: Float32Array; dims: number[] }>>
	inputs?: { name: string }[]
	free?(): void
}

export interface PosteriorOptions {
	/** required unless audio is the { channelData, sampleRate } form */
	sampleRate?: number
	/** ONNX model: URL or bytes (default: MODEL, fetched and cached by @audio/neural-runtime) */
	model?: string | Uint8Array
	/** passed to @audio/neural-runtime's load() as `backend`: 'node' | 'wasm' | 'webgpu' (default 'auto') */
	device?: string
	/** 2 s windows per model run (default 8); ORT picks other kernels above 1, posteriors move by up to 1e-4 */
	batch?: number
	/** replaces @audio/neural-runtime's load(); the session is freed after the call */
	session?: (model: string | Uint8Array, opts: PosteriorOptions) => Session | Promise<Session>
}

export interface NoteOptions {
	/** onset posterior peak needed to start a note (default 0.5) */
	onsetThreshold?: number
	/** note posterior needed to sustain it (default 0.3, ≥ 0) */
	frameThreshold?: number
	/** seconds, rounded to frames; a note must span more frames than this (default 0.1277 s = 11 frames: kept notes last 12 frames, 139 ms, or longer) */
	minDuration?: number
	/** Hz: keys below the one nearest this frequency are ignored */
	minFreq?: number
	/** Hz: the key nearest this frequency and those above it are ignored */
	maxFreq?: number
	/** add onsets where the note posterior jumps (default true) */
	inferOnsets?: boolean
	/** also grow notes from remaining energy without an onset (default true) */
	melodiaTrick?: boolean
	/** bends exactly as Basic Pitch gives them, centered one contour bin below the note: an in-tune note reads +33.3 (default false) */
	upstream?: boolean
}

export type TranscribeOptions = PosteriorOptions & NoteOptions

/** The ONNX model URL: basic_pitch/saved_models/icassp_2022/nmp.onnx at the ported commit. */
export const MODEL: string

/** Notes with pitch bends, sorted by time then pitch. */
export default function transcribe(audio: AudioInput, opts?: TranscribeOptions): Promise<Note[]>

/** The model's frame posteriors: resampling to 22050 Hz, windowing and unwrapping as Basic Pitch's run_inference. */
export function posteriors(audio: AudioInput, opts?: PosteriorOptions): Promise<Posteriors>

/** Notes from posteriors (Basic Pitch's model_output_to_notes): re-threshold without running the model again. */
export function toNotes(posteriors: Pick<Posteriors, 'onset' | 'note' | 'contour'> & { times?: Float64Array }, opts?: NoteOptions): Note[]
