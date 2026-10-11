/**
 * Timbre recognition: a small encoder maps one note to a unit vector; a catalog holds one vector
 * per preset (the mean over its rendered reference notes), bound to the encoder's SHA-256; a query
 * is ranked against the catalog by cosine. Runs through @audio/neural-runtime (ONNX).
 */

/** One note: mono samples, or channels to average. */
export type Audio = Float32Array | Float32Array[]

/** A model or catalog source: URL, file path (Node), or bytes. */
export type Source = string | URL | Uint8Array | ArrayBuffer

/** The commit of the Hugging Face repository audiojs/dx7-timbre this version reads the encoder and catalogs from. */
export const REVISION: string

/** The hosted int8 encoder, https://huggingface.co/audiojs/dx7-timbre/resolve/<REVISION>/encoder.int8.onnx: the graph the hosted catalogs bind. */
export const MODEL: URL

/** The hosted catalogs, catalogs/<name>.json beside MODEL. */
export const catalogs: Record<CatalogName, URL>

/** 'dx7-factory': Yamaha's ROM and VRC cartridges, 988 sounds; 'dx7-all': every audible voice of every source, 29,840 (README, Catalogs). */
export type CatalogName = 'dx7-factory' | 'dx7-all'

/** The encoder's input contract; a catalog binds its hash beside the encoder's. */
export interface Features {
	/** constant-Q power spectrogram (Brown & Puckette 1992) */
	kind: 'cqt'
	/** analysis rate; other rates are resampled to it */
	sampleRate: number
	/** FFT size (the longest window) and hop, samples */
	n: number
	hop: number
	/** bins, `perOctave` an octave from `fmin` Hz */
	bands: number
	fmin: number
	perOctave: number
	/** frames read from the note's onset; frame t is centred `center` samples after t hops */
	frames: number
	center: number
	/** dB range below the loudest cell */
	top: number
}

export const FEATURES: Readonly<Features>

/** One note → Float32Array(bands × frames), band-major, in [-1, 1]: cut at the onset, level-normalized. */
export function features(audio: Audio, opts?: { sampleRate?: number; trim?: boolean }): Float32Array

/** Constant-Q features of samples at the features' rate that start at the note. */
export function cqt(x: ArrayLike<number>, features?: Features): Float32Array

export interface Binding {
	encoderSha256: string
	featuresSha256: string
}

export interface EncoderOptions {
	/** ONNX encoder, default the hosted one (MODEL, the int8 graph the hosted catalogs are built with) */
	model?: Source
	/**
	 * Where the hosted files are read from instead of the Hugging Face revision: a URL, or a directory in Node, holding
	 * a copy laid out as the repository (encoder.int8.onnx, catalogs/<name>.json). Each file is checked against its
	 * SHA-256 wherever it comes from. Without it, Node fetches each file once into the neural cache
	 * ($AUDIO_NEURAL_CACHE or ~/.cache/audiojs/neural, dx7-timbre/) and reads it there offline after; the browser keeps
	 * it in the Cache API.
	 */
	weights?: string | URL
	/** @audio/neural-runtime backend, default 'auto' */
	backend?: 'auto' | 'node' | 'wasm' | 'webgpu' | 'webgl'
}

export interface Encoder {
	/** the hashes a catalog must carry to fit this encoder */
	binding: Binding
	/** one note → its unit embedding */
	embed(audio: Audio, opts?: { sampleRate?: number }): Promise<Float32Array>
	/** prepared feature patches → unit embeddings */
	project(patches: Float32Array[]): Promise<Float32Array[]>
	free(): void
}

export function createEncoder(opts?: EncoderOptions): Promise<Encoder>

/** Catalog entry metadata; DX7 catalogs carry name, cartridge, index (the voice's number in the file, from 1), source, algorithm (1-32) and id. */
export interface Entry {
	name: string
	[field: string]: unknown
}

export interface CatalogJson {
	version: 1
	kind: 'timbre-catalog'
	encoderSha256: string
	featuresSha256: string
	dimensions: number
	entries: Entry[]
	vectors: { encoding: 'int8-base64' | 'int4-base64'; shape: [number, number]; scales: number[]; data: string }
	[field: string]: unknown
}

export interface Catalog {
	entries: Entry[]
	/** entries × dimensions, rows unit length */
	vectors: Float32Array
	dimensions: number
}

/** Catalog JSON from entries and their unit vectors, quantized per row (default int8). */
export function catalog(entries: Entry[], vectors: Float32Array[], binding: Binding, info?: { encoding?: 'int8-base64' | 'int4-base64'; [field: string]: unknown }): CatalogJson

/** Reads a catalog; throws when `binding` differs from the one it was built with. */
export function readCatalog(data: CatalogJson, binding?: Binding): Catalog

/** Entries best first by cosine; ties by index. */
export function rank(embedding: ArrayLike<number>, catalog: Catalog, top?: number): { index: number; score: number }[]

export interface Match extends Entry {
	/** cosine similarity, -1 to 1 */
	score: number
}

/** A hosted catalog by name, a URL, a path (Node), or parsed catalog JSON. */
export type CatalogSource = CatalogName | string | URL | CatalogJson

export interface Matcher {
	catalog: Catalog
	encoder: Encoder
	/** entries best first */
	match(audio: Audio, opts?: { sampleRate?: number; top?: number }): Promise<Match[]>
	free(): void
}

/** Loads a catalog ('dx7-factory' by default) and the encoder; throws when the catalog was built with another, or a hosted file differs from its SHA-256. */
export function createMatcher(catalog?: CatalogSource, opts?: EncoderOptions): Promise<Matcher>

/** One-shot: load, match, free. */
export default function identify(audio: Audio, opts?: EncoderOptions & { catalog?: CatalogSource; sampleRate?: number; top?: number }): Promise<Match[]>

export function sha256(bytes: Uint8Array | ArrayBuffer): Promise<string>
export function featuresSha256(features?: Features): Promise<string>
export function unit(vector: ArrayLike<number>): Float32Array
export function pack(values: ArrayLike<number>, bits: number): Uint8Array
export function unpack(bytes: Uint8Array, bits: number, count: number): Int8Array
