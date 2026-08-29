/**
 * Source separation (stems) — Open-Unmix-class spectrogram masking models
 * (Stöter, Uhlich, Liutkus, Mitsufuji, JOSS 2019) plus Demucs-class waveform
 * models, run through @audio/neural-runtime's model-agnostic ONNX adapter.
 */

/** Mono duplicates internally; multichannel arrays must share the same length. */
export type AudioInput = Float32Array[] | { channelData: Float32Array[]; sampleRate: number }

/** A model spec resolvable by @audio/neural-runtime's load(): URL string or raw ONNX bytes. */
export type ModelSpec = string | Uint8Array

export type ModelOption =
	| ModelSpec // single target, named 'stem'
	| Record<string, ModelSpec> // one ONNX graph per target — Open-Unmix's own layout
	| { url: ModelSpec; targets: string[] } // one multi-target graph, output stacks a target axis

export type ModelType =
	| 'openunmix' // target model outputs the estimated magnitude directly
	| 'mask' // target model outputs a [0,1] mask; multiplied by the mixture magnitude
	| 'waveform' // Demucs-class: [1,C,N] waveform in, [1,S,C,N] stacked waveforms out

/** A neural-runtime-shaped session — enough of it to drive separate() with a test double. */
export interface Session {
	run(feeds: Record<string, { data: Float32Array; dims: number[]; type: string }>): Promise<Record<string, { data: Float32Array; dims: number[] }>>
	inputs?: { name: string }[]
	outputs?: { name: string }[]
	free?(): void
}

export interface SeparateOptions {
	/** required unless audio is the { channelData, sampleRate } form */
	sampleRate?: number
	model: ModelOption
	modelType?: ModelType
	/** iterations of multichannel Wiener EM refinement (default 1); 0 = raw masks. Ignored for modelType 'waveform'. */
	wiener?: number
	/** wienerFilter's softmask option (default false, matching open-unmix-pytorch) */
	softmask?: boolean
	/** wienerFilter's eps (default 1e-10) */
	eps?: number
	/** chunk length in seconds (default 30) */
	chunk?: number
	/** crossfade overlap in seconds between chunks (default 2); must be < chunk */
	overlap?: number
	/** resample to this rate for model inference; output is resampled back to the input rate (default: input rate, no resampling) */
	targetRate?: number
	/** STFT size for the spectral pipeline (default 4096, Open-Unmix's n_fft) */
	n?: number
	/** STFT hop for the spectral pipeline (default 1024, Open-Unmix's n_hop) */
	hop?: number
	/** passed through to @audio/neural-runtime's load() as `backend` */
	device?: string
	/** only 'float32' is implemented; anything else throws */
	dtype?: 'float32'
	progress?: (p: { chunk: number; totalChunks: number }) => void
	/** overrides @audio/neural-runtime's load() — for tests, or a custom ORT setup */
	session?: (model: ModelSpec, opts: SeparateOptions) => Session | Promise<Session>
}

export interface SeparateResult {
	/** keys are the model's target names ('stem' for a single bare model) */
	stems: Record<string, Float32Array[]>
	sampleRate: number
	/** mixture minus the sum of all stems, per channel, at the input rate */
	residual: Float32Array[]
}

/** Separate a mixture into stems. See README for the ONNX I/O contract per modelType. */
export default function separate(audio: AudioInput, opts: SeparateOptions): Promise<SeparateResult>

// ------------------------------------------------------------- STFT / iSTFT

export interface StftOptions {
	/** FFT size, power of 2 (default 4096) */
	n?: number
	/** hop size (default 1024) */
	hop?: number
	/** analysis window, length n (default: periodic Hann) */
	window?: Float64Array
	/** torch.stft-compatible reflect-padding by n/2 on both ends (default true) */
	center?: boolean
}

export interface ComplexStft {
	/** re[frame] is a Float64Array(bins), bins = n/2+1 */
	re: Float64Array[]
	im: Float64Array[]
	n: number
	hop: number
	bins: number
	center: boolean
}

/** Mono complex STFT. Frame count = 1 + floor(x.length / hop) when center=true (torch.stft-compatible). */
export function stft(x: Float32Array | Float64Array, opts?: StftOptions): ComplexStft

export interface IstftOptions {
	/** exact output length (torch.istft-compatible crop/pad); default: paddedLength − 2·(n/2) when center */
	length?: number
	/** synthesis window (default: the same periodic Hann as stft()) */
	window?: Float64Array
}

/** Exact inverse of stft() via squared-window-normalized overlap-add (WOLA). */
export function istft(frames: ComplexStft, opts?: IstftOptions): Float32Array

// ------------------------------------------------------------- Wiener filter

export interface WienerOptions {
	/** EM refinement steps (default 1); 0 returns the initial softmask/phase-substituted estimate untouched */
	iterations?: number
	/** true = ratio mask (sums to the mixture exactly); false = magnitude with the mixture's phase (default, open-unmix's recommendation) */
	softmask?: boolean
	/** appends a 'residual' target = mixture − Σ(other targets), computed before EM */
	residual?: boolean
	/** regularization floor (default 1e-10) */
	eps?: number
	/** EM numerical-stability rescaling divisor (default 10, open-unmix-pytorch's scale_factor) */
	scaleFactor?: number
}

/**
 * Multichannel Wiener EM (Liutkus & Stöter, github.com/sigsep/norbert; Duong,
 * Vincent, Gribonval, IEEE TASLP 2010) with open-unmix-pytorch's defaults.
 * estimates: per-target magnitude, one Float64Array(bins) per frame per
 * channel — a single-channel estimate broadcasts to all mixture channels.
 */
export function wienerFilter(
	mixStft: ComplexStft[],
	estimates: Record<string, Float64Array[][]>,
	opts?: WienerOptions
): Record<string, ComplexStft[]>
