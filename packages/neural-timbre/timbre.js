// Name the sound in a recording: a small encoder maps one note to a unit vector, and a catalog of
// reference vectors (one per preset, the mean over its rendered notes) is ranked by cosine. The
// encoder is trained once; catalogs are swappable, each bound to the encoder bytes that built it
// (gpu-font's split: the renderer labels, the network only encodes).
//
// The int8 encoder and two catalogs are hosted on Hugging Face, not in the package, so a source can
// come out at its owner's request (README, Catalogs); scripts/ rebuild them.

import { load, tensor, fetchModel } from '@audio/neural-runtime'
import { features, cqt, FEATURES } from './features.js'

export { features, cqt, FEATURES }

// The commit of audiojs/dx7-timbre this version reads, so a version always gets the files it was
// tested with.
export const REVISION = '2dd88e107183463c5a40609a92b0545d8857c635'
const REPO = 'audiojs/dx7-timbre'
const HOSTED = `https://huggingface.co/${REPO}/resolve/${REVISION}/`

// The hosted files, laid out as in the repository, each checked against its SHA-256 when read
const FILES = {
	encoder: { file: 'encoder.int8.onnx', sha256: 'aaa9fae0337325fc9ae0f15c45f711073255c09264a31a9779da635ec21b8484' },
	'dx7-factory': { file: 'catalogs/dx7-factory.json', sha256: '7ab9d48c1b90a4ec04af3aeb63737352d63f959db8b1459d6b238cea365a0900' },
	'dx7-all': { file: 'catalogs/dx7-all.json', sha256: '0dd0fd646e8aef7e417151e8ae76b5c2bec7fde3cb531e01a3bc4e5b964e3671' },
}
export const MODEL = new URL(FILES.encoder.file, HOSTED)
export const catalogs = {
	'dx7-factory': new URL(FILES['dx7-factory'].file, HOSTED),
	'dx7-all': new URL(FILES['dx7-all'].file, HOSTED),
}

const isNode = globalThis.process?.release?.name === 'node'  // Node, Bun, Deno: not a process polyfill (esm.sh's, a bundler's), whose versions.node is set
// a URL scheme has two characters at least, so a Windows drive (C:\) stays a path
const isUrl = s => /^[a-z][a-z0-9+.-]+:/i.test(s)

// Web Crypto: global in browsers and Node 19+, node:crypto's webcrypto in Node 18
const subtle = async () => globalThis.crypto?.subtle ?? (await import('node:crypto')).webcrypto.subtle
export const sha256 = async bytes => Array.from(new Uint8Array(await (await subtle()).digest('SHA-256', bytes)), b => b.toString(16).padStart(2, '0')).join('')
const canonical = v => Array.isArray(v) ? v.map(canonical) : v && typeof v === 'object' ? Object.fromEntries(Object.keys(v).sort().map(k => [k, canonical(v[k])])) : v
export const featuresSha256 = (f = FEATURES) => sha256(new TextEncoder().encode(JSON.stringify(canonical(f))))

// Bytes of a model or catalog: bytes as given; a URL, or a path in Node, read through
// @audio/neural-runtime's fetchModel (file: URLs from disk).
async function bytesOf(src, opts) {
	if (src instanceof Uint8Array) return src
	if (src instanceof ArrayBuffer) return new Uint8Array(src)
	let href = String(src)
	if (isNode && !isUrl(href)) href = (await import('node:url')).pathToFileURL(href).href
	return fetchModel(href, opts)
}

// Base URL of the hosted files: opts.weights (a URL, or a directory in Node) holding a copy laid out
// as the repository, else the pinned revision
async function base(w) {
	if (w == null) return HOSTED
	w = String(w)
	if (!isUrl(w)) {
		if (!isNode) throw new Error(`neural-timbre: opts.weights must be a URL in the browser, got '${w}'`)
		w = (await import('node:url')).pathToFileURL(w).href
	}
	return w.endsWith('/') ? w : w + '/'
}

// A hosted file's bytes ('encoder', 'dx7-factory', 'dx7-all') from opts.weights or the pinned
// revision, refused unless of its SHA-256. A local copy is read in place; Node keeps a fetched one
// in the neural cache, dx7-timbre/<file>, written whole once checked and read while its SHA-256
// holds, so it loads offline after the first time; the browser keeps it in neural-runtime's Cache API.
async function hosted(key, opts) {
	let { file, sha256: want } = FILES[key], url = await base(opts.weights) + file
	let store = isNode && !url.startsWith('file:') ? await cached(file) : null, bytes = store?.read()
	if (bytes && await sha256(bytes) === want) return bytes
	try { bytes = await fetchModel(url, { progress: opts.progress, fetch: opts.fetch, cache: !store }) }
	catch (e) { throw new Error(`neural-timbre: ${file} not read (${e.message}): opts.weights takes a URL or directory holding a copy of ${HOSTED}`) }
	let hex = await sha256(bytes)
	if (hex !== want) throw new Error(`neural-timbre: ${url} has SHA-256 ${hex}, expected ${want}: a partial or altered download${isNode ? '' : " (clear the 'audio-neural' cache)"}, or another build of the file`)
	store?.write(bytes)
	return bytes
}

// <neural cache>/dx7-timbre/<file>: { read() → bytes or null, write(bytes) whole or not at all }
async function cached(file) {
	let [fs, path, os] = await Promise.all([import('node:fs'), import('node:path'), import('node:os')])
	let dest = path.join(process.env.AUDIO_NEURAL_CACHE || path.join(os.homedir(), '.cache', 'audiojs', 'neural'), REPO.split('/')[1], file)
	return {
		read() { try { let b = fs.readFileSync(dest); return new Uint8Array(b.buffer, b.byteOffset, b.length) } catch { return null } },
		write(bytes) {
			let part = `${dest}.${process.pid}.${Math.random().toString(36).slice(2)}.part`
			fs.mkdirSync(path.dirname(dest), { recursive: true })
			fs.writeFileSync(part, bytes)
			fs.renameSync(part, dest)
		},
	}
}

// createEncoder({ model, backend, weights }) → { embed(audio, { sampleRate }) → unit Float32Array,
// project(patches), binding: { encoderSha256, featuresSha256 }, free() }. model: URL, path or
// bytes, default the hosted encoder (MODEL), the graph the hosted catalogs are built with.
export async function createEncoder(opts = {}) {
	let bytes = await (opts.model != null ? bytesOf(opts.model, opts) : hosted('encoder', opts))
	let binding = { encoderSha256: await sha256(bytes), featuresSha256: await featuresSha256() }
	let session = await load(bytes, { backend: opts.backend })
	let input = session.inputs[0]?.name ?? 'features'
	let output = session.outputs.some(o => o.name === 'embedding') ? 'embedding' : session.outputs[0]?.name
	// prepared feature patches (bands × frames each) → unit embeddings
	let project = async patches => {
		let { bands, frames } = FEATURES, n = patches.length, x = new Float32Array(n * bands * frames)
		patches.forEach((p, i) => x.set(p, i * bands * frames))
		let e = (await session.run({ [input]: tensor(x, [n, 1, bands, frames]) }))[output].data, d = e.length / n
		return Array.from({ length: n }, (_, i) => unit(e.subarray(i * d, (i + 1) * d)))
	}
	return {
		binding, project,
		// one note (Float32Array, or channels to average) → its unit embedding
		embed: async (audio, { sampleRate = FEATURES.sampleRate } = {}) => (await project([features(audio, { sampleRate })]))[0],
		free: () => session.free(),
	}
}

export function unit(v) {
	let s = 0
	for (let x of v) s += x * x
	let norm = Math.sqrt(s)
	if (!(norm > 1e-12)) throw new Error('neural-timbre: zero embedding')
	return Float32Array.from(v, x => x / norm)
}

// Catalog vectors: one row per entry, `bits` a number with one scale per row (gpu-font's
// int8-base64 and int4-base64: signed values in a little-endian bit stream).
const BITS = { 'int8-base64': 8, 'int4-base64': 4 }

export function pack(values, bits) {
	let bytes = new Uint8Array(Math.ceil(values.length * bits / 8)), mask = (1 << bits) - 1
	for (let i = 0, bit = 0; i < values.length; i++, bit += bits) {
		let word = (values[i] & mask) << (bit & 7)
		bytes[bit >> 3] |= word
		if (word > 255) bytes[(bit >> 3) + 1] |= word >> 8
	}
	return bytes
}

export function unpack(bytes, bits, count) {
	let values = new Int8Array(count), mask = (1 << bits) - 1, sign = 1 << (bits - 1)
	for (let i = 0, bit = 0; i < count; i++, bit += bits) {
		let v = ((bytes[bit >> 3] | bytes[(bit >> 3) + 1] << 8) >> (bit & 7)) & mask
		values[i] = v >= sign ? v - (1 << bits) : v
	}
	return values
}

const toBase64 = bytes => { let s = ''; for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode(...bytes.subarray(i, i + 0x8000)); return btoa(s) }
const fromBase64 = text => Uint8Array.from(atob(text), c => c.charCodeAt(0))

// catalog(entries, vectors, binding, info) → catalog JSON: entries are the metadata of each row
// ({ name, … }), vectors their unit embeddings, binding the encoder's.
export function catalog(entries, vectors, binding, { encoding = 'int8-base64', ...info } = {}) {
	let bits = BITS[encoding], top = 2 ** (bits - 1) - 1, d = vectors[0]?.length
	if (!bits) throw new Error(`neural-timbre: unknown encoding ${encoding}`)
	if (entries.length !== vectors.length || !d) throw new Error('neural-timbre: one vector per entry')
	let q = new Int8Array(vectors.length * d), scales = []
	vectors.forEach((v, r) => {
		let m = Math.max(...Array.from(v, Math.abs)) || 1, s = m / top
		scales.push(+s.toPrecision(7))
		for (let i = 0; i < d; i++) q[r * d + i] = Math.max(-top, Math.min(top, Math.round(v[i] / scales[r])))
	})
	return { version: 1, kind: 'timbre-catalog', ...binding, dimensions: d, ...info, entries,
		vectors: { encoding, shape: [vectors.length, d], scales, data: toBase64(pack(q, bits)) } }
}

// readCatalog(json, binding) → { entries, vectors: Float32Array (rows unit), dimensions }; refuses a
// catalog built by another encoder or frontend.
export function readCatalog(data, binding) {
	if (data?.kind !== 'timbre-catalog' || data.version !== 1) throw new Error('neural-timbre: not a timbre catalog')
	if (binding && (data.encoderSha256 !== binding.encoderSha256 || data.featuresSha256 !== binding.featuresSha256))
		throw new Error('neural-timbre: catalog was built with a different model: rebuild it with this one')
	let { entries, dimensions: d, vectors: v } = data, bits = BITS[v?.encoding]
	if (!bits || !Array.isArray(entries) || v.shape?.[0] !== entries.length || v.shape[1] !== d || v.scales?.length !== entries.length) throw new Error('neural-timbre: malformed catalog')
	let bytes = fromBase64(v.data)
	if (bytes.length !== Math.ceil(entries.length * d * bits / 8)) throw new Error('neural-timbre: truncated catalog vectors')
	let q = unpack(bytes, bits, entries.length * d), out = new Float32Array(entries.length * d)
	for (let r = 0; r < entries.length; r++) {
		let s = 0
		for (let i = 0; i < d; i++) { let x = q[r * d + i] * v.scales[r]; out[r * d + i] = x; s += x * x }
		let n = Math.sqrt(s) || 1
		for (let i = 0; i < d; i++) out[r * d + i] /= n
	}
	return { entries, vectors: out, dimensions: d }
}

// rank(embedding, catalog, top) → [{ index, score }] best first; ties by index
export function rank(e, { vectors, dimensions: d }, top = 10) {
	let n = vectors.length / d, scores = new Float32Array(n)
	for (let r = 0; r < n; r++) { let s = 0; for (let i = 0; i < d; i++) s += e[i] * vectors[r * d + i]; scores[r] = s }
	let order = Array.from({ length: n }, (_, i) => i).sort((a, b) => scores[b] - scores[a] || a - b)
	return order.slice(0, top).map(index => ({ index, score: scores[index] }))
}

// A catalog's JSON: parsed JSON as given; a hosted catalog by name; a URL, or a path in Node.
async function catalogJson(c, opts) {
	let text = bytes => JSON.parse(new TextDecoder().decode(bytes))
	if (typeof c === 'string' && !/[/\\.:]/.test(c)) {
		if (!Object.hasOwn(catalogs, c)) throw new Error(`neural-timbre: no catalog '${c}' is hosted (${Object.keys(catalogs).join(', ')}): pass a URL, a path or parsed JSON`)
		return text(await hosted(c, opts))
	}
	return typeof c === 'string' || c instanceof URL ? text(await bytesOf(c, { cache: false })) : c
}

// createMatcher(catalog, { model, backend, weights }) → { match(audio, { sampleRate, top }), catalog,
// encoder, free }. catalog: a hosted one by name ('dx7-factory', the default, or 'dx7-all'), a URL,
// a path in Node, or parsed JSON. It loads while the encoder does and must carry the encoder's binding.
export async function createMatcher(catalog = 'dx7-factory', opts = {}) {
	let data = catalogJson(catalog, opts)
	data.catch(() => {}) // awaited once the encoder is up
	let encoder = await createEncoder(opts)
	try {
		let cat = readCatalog(await data, encoder.binding)
		return {
			catalog: cat, encoder,
			// best first: { ...entry, score } with score the cosine, -1 to 1
			async match(audio, { sampleRate, top = 5 } = {}) {
				let e = await encoder.embed(audio, { sampleRate })
				return rank(e, cat, top).map(({ index, score }) => ({ ...cat.entries[index], score }))
			},
			free: encoder.free,
		}
	} catch (e) { encoder.free(); throw e }
}

// identify(audio, { sampleRate, top, catalog, model, backend }) → matches; loads and frees a
// matcher per call.
export default async function identify(audio, opts = {}) {
	let m = await createMatcher(opts.catalog, opts)
	try { return await m.match(audio, opts) } finally { m.free() }
}
