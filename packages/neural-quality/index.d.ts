/**
 * Speech quality with no clean reference: Microsoft's DNSMOS P.835 (microsoft/DNS-Challenge at 82f1b17e77,
 * sig_bak_ovr.onnx, CC BY 4.0) through @audio/neural-runtime, as its dnsmos_local.py computes it.
 */

/** a mono Float32Array, one array per channel (mixed down to their mean), or a decoded sound */
export type Audio = Float32Array | ArrayLike<number> | Float32Array[] | { channelData: Float32Array[], sampleRate: number }

export interface LoadOptions {
	/** the model: bytes, a URL, or in Node a file or directory path holding sig_bak_ovr.onnx; default the neural cache's copy when of SHA256, else MODEL */
	weights?: string | URL | Uint8Array | ArrayBuffer
	/** @audio/neural-runtime's backend: 'auto' (default), 'node', 'wasm', 'webgpu', 'coreml' */
	backend?: string
	/** ONNX Runtime session options, e.g. { intraOpNumThreads: 1 } */
	sessionOptions?: Record<string, unknown>
	progress?: (p: { loaded: number, total: number | null }) => void
	fetch?: typeof fetch
}

export interface Model {
	session: { inputs: { name: string }[], outputs: { name: string }[], run: (feeds: Record<string, unknown>) => Promise<Record<string, { data: Float32Array }>> }
	free(): void
}

export interface Options extends LoadOptions {
	/** required unless `audio` carries it */
	sampleRate?: number
	/** a load() handle, kept by the caller across calls; without it the model is loaded and freed per call */
	model?: Model
	/** 'upstream' (default): the windows dnsmos_local.py keeps, which leaves 15.01 to 24 s of a longer clip unscored; 'all': every 1 s hop */
	windows?: 'upstream' | 'all'
}

/** The three P.835 scales, 1 (bad) to 5 (excellent) */
export interface Scores {
	/** the speech: how clean and undistorted the voice sounds */
	sig: number
	/** the background: how little and how unobtrusive the noise is */
	bak: number
	/** the whole: the quality a listener gives the recording overall */
	ovrl: number
}

export interface Segment extends Scores {
	/** the window's start in seconds; in the doubled signal for a clip under 9.01 s */
	time: number
	/** the network's own outputs, before the polynomials: [sig, bak, ovrl] */
	raw: [number, number, number]
}

/** SIG, BAK, OVRL: each the mean over the windows of the polynomial of the network's output */
export default function dnsmos(audio: Audio, opts?: Options): Promise<Scores & { segments: Segment[] }>
export function dnsmos(audio: Audio, opts?: Options): Promise<Scores & { segments: Segment[] }>

/** The network, loaded once, for dnsmos() to reuse through opts.model */
export function load(opts?: LoadOptions): Promise<Model>

/** get_polyfit_val's polynomials: the network's raw outputs to the P.835 scale */
export function polyfit(raw: Scores): Scores

/** The windows of a 16 kHz signal of n samples: the doubled length, and each window's first sample */
export function windows(n: number, opts?: { windows?: 'upstream' | 'all' }): { length: number, starts: number[] }

/** What the windows are cut from: the channels' mean at 16 kHz, librosa's length, ceil(n · 16000 / rate) */
export function prepare(audio: Audio, opts?: { sampleRate?: number }): Float32Array

/** 16000, 9.01 s, 144160 samples */
export const SAMPLE_RATE: 16000
export const INPUT_LENGTH: 9.01
export const WINDOW: 144160

/** microsoft/DNS-Challenge's commit, sig_bak_ovr.onnx's SHA-256 there, and its URL at that commit */
export const REVISION: string
export const SHA256: string
export const MODEL: URL
