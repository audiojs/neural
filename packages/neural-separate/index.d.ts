/**
 * Source separation (stems) through @audio/neural-runtime's ONNX adapter: Open-Unmix-class
 * spectrogram models (Stöter, Uhlich, Liutkus, Mitsufuji, JOSS 2019), Hybrid Transformer Demucs
 * (Rouard, Massa, Défossez, ICASSP 2023), SCNet (Tong et al., ICASSP 2024) and MRX (Petermann et al., ICASSP 2022)
 * with STFT and iSTFT outside the graph, and waveform graphs.
 */

/** Mono duplicates internally (a model hearing channels apart, mrx and tiger, keeps it mono); multichannel arrays must share the same length. */
export type AudioInput = Float32Array[] | { channelData: Float32Array[]; sampleRate: number }

/** A model spec resolvable by @audio/neural-runtime's load(): URL string or raw ONNX bytes. */
export type ModelSpec = string | Uint8Array

/** Presets whose files the export scripts write: <weights>/<name>/<target>.onnx or <weights>/<name>/<name>.onnx */
export type ModelName = 'umxhq' | 'htdemucs' | 'htdemucs_ft' | 'scnet-large' | 'scnet' | 'mrx' | 'tiger'

/** One graph whose output stacks several sources on its S axis */
export type MultiGraph = { url: ModelSpec; targets: string[] }

export type ModelOption =
	| ModelName
	| ModelSpec // single target, named 'stem'
	| Record<string, ModelSpec | MultiGraph> // one graph per target (Open-Unmix's layout); a multi-source graph contributes its own target
	| MultiGraph

export type ModelType =
	| 'openunmix' // target model outputs the estimated magnitude directly
	| 'mask' // target model outputs a [0,1] mask; multiplied by the mixture magnitude
	| 'hybrid' // demucs.onnx contract: mix [1,C,L] + CaC spectrogram [1,2C,F,T] in; [1,S,2C,F,T] + [1,S,C,L] out
	| 'complex' // SCNet-class: CaC spectrogram [1,2C,F,T] in; [1,2SC,F,T] out (source, channel, re/im)
	| 'multires' // MRX-class: one channel's magnitudes mag_<n> [1,F,T] at each window n in; real masks mask_<n> [1,S,F,T] out
	| 'waveform' // Demucs v2-class: [1,C,N] waveform in, [1,S,C,N] stacked waveforms out

export interface ModelPreset {
	modelType: ModelType
	sampleRate: number
	/** in the order stems come back */
	targets: string[]
	/** one graph per target (<target>.onnx), else one graph for all (<name>.onnx) */
	perTarget?: boolean
	/** 'complex': the model's STFT (size, hop, window, normalized) and segment (samples, frames) */
	n?: number
	hop?: number
	window?: 'ones' | 'hann'
	normalized?: boolean
	segment?: number
	frames?: number
	/** the graph's source order, when not the targets' (or Demucs's for 'hybrid' and 'complex') */
	sources?: string[]
	/** 'multires': its windows, one hop */
	windows?: number[]
	/** integrated loudness (LUFS) the input is set to, its stems scaled back */
	loudness?: number
	/** channels heard apart: mono is not duplicated */
	mono?: boolean
	/** 'complex': segments' step and fade (fractions of a segment), their padding past the ends, the input normalized or not */
	step?: number
	fade?: number
	pad?: 'reflect' | 'zero'
	standardize?: boolean
	/** chunk length (s) when not 30 */
	chunk?: number
	/** the export script that writes its weights */
	script?: string
	/** its compact file (scripts/compact.py: int8 weights, a few float16, float32 compute), read before <name>.onnx */
	file?: string
	/** the Hugging Face repository hosting it */
	repo?: string
	/** the compact file's SHA-256, checked when it is fetched */
	sha256?: string
}

/** Model presets by name */
export const models: Record<ModelName, ModelPreset>

/** The hosted revision (commit) of each compact preset's repository: https://huggingface.co/<repo>/resolve/<revision>/<file> is read without opts.weights in the browser, and in Node when the cache holds neither the file nor its export. Empty: not hosted. */
export const REVISIONS: Record<'scnet-large' | 'scnet' | 'mrx' | 'tiger', string>

/** A neural-runtime-shaped session — enough of it to drive separate() with a test double. */
export interface Session {
	run(feeds: Record<string, { data: Float32Array; dims: number[]; type: string }>): Promise<Record<string, { data: Float32Array; dims: number[] }>>
	/** 'hybrid' reads its segment length off inputs[0].dims[2] when declared, 'complex' its frames off inputs[0].dims[3] */
	inputs?: { name: string; dims?: number[] }[]
	outputs?: { name: string }[]
	free?(): void
}

export interface SeparateOptions {
	/** required unless audio is the { channelData, sampleRate } form */
	sampleRate?: number
	model: ModelOption
	/** ignored for presets, which know theirs (default 'openunmix') */
	modelType?: ModelType
	/** subset of the model's targets to return; spectral models skip the other graphs, and one target runs Wiener EM against the residual */
	targets?: string[]
	/** where a preset's files are: URL, or a directory in Node (default $AUDIO_NEURAL_CACHE or ~/.cache/audiojs/neural in Node; a compact preset's hosted file, REVISIONS) */
	weights?: string
	/** iterations of multichannel Wiener EM refinement (default 1); 0 = raw masks. Ignored for modelType 'hybrid' and 'waveform'. */
	wiener?: number
	/** wienerFilter's softmask option (default false, matching open-unmix-pytorch) */
	softmask?: boolean
	/** wienerFilter's eps (default 1e-10) */
	eps?: number
	/** frames per Wiener EM window (default 300, open-unmix Separator's wiener_win_len) */
	wienerWindow?: number
	/** 'hybrid': segment length in samples when the graph does not declare it (default 343980, htdemucs's 7.8 s); 'complex': the preset's (SCNet 485100, 11 s) */
	segment?: number
	/** 'complex': segments start every `step` of a segment (default 0.25: each sample in four, as Music-Source-Separation-Training's num_overlap 4) */
	step?: number
	/** 'complex': each segment's linear fade in and out, a fraction of it (default 0.1, demix's; tiger 0) */
	fade?: number
	/** 'complex': past the input's ends, reflected (default) or zeros (tiger) */
	pad?: 'reflect' | 'zero'
	/** 'complex': the input normalized by its mean and deviation (default true; tiger false) */
	standardize?: boolean
	/** 'complex': frames per segment when the graph does not declare them (inputs[0].dims[3]) */
	frames?: number
	/** 'complex': the STFT window, 'ones' (torch.stft's window=None, SCNet's) or 'hann' (default: the preset's, else 'hann') */
	window?: 'ones' | 'hann'
	/** 'complex': the STFT scaled by 1/√n and back, torch.stft's normalized=True (default: the preset's, else true) */
	normalized?: boolean
	/** 'openunmix' | 'mask' | 'waveform' | 'multires': chunk length in seconds (default 30; mrx 20) */
	chunk?: number
	/** crossfade overlap in seconds between chunks (default 2); must be < chunk */
	overlap?: number
	/** resample to this rate for model inference, stems back to the input rate (default: the preset's rate, else the input's) */
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
