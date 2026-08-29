// ONNX inference adapter — one small interface over onnxruntime-node (Node) and
// onnxruntime-web (browser/worker/worklet), so every neural atom (asr, align,
// separate, diarize, tts, denoise, …) loads and runs models the same way.
//
// Neither runtime is a hard dependency: both are optional peers, resolved by
// dynamic import the first time a backend is used and memoized after. Missing
// the right one throws naming the package to install, rather than failing to
// resolve a static import at bundle time.

const isNode = typeof process !== 'undefined' && !!process?.versions?.node

const ORT_SPEC = {
  node: 'onnxruntime-node',
  wasm: 'onnxruntime-web',
  webgl: 'onnxruntime-web',
  webgpu: 'onnxruntime-web/webgpu',
}

function resolveBackend(want) {
  if (!want || want === 'auto') return isNode ? 'node' : 'wasm'
  return want
}

const ortCache = new Map()

function getOrt(backend) {
  if (isNode && backend !== 'node')
    throw new Error(`neural-runtime: backend '${backend}' needs a browser (onnxruntime-web) — Node only runs backend 'node' (onnxruntime-node)`)
  if (!isNode && backend === 'node')
    throw new Error(`neural-runtime: backend 'node' needs Node.js (onnxruntime-node) — in a browser use 'wasm' or 'webgpu' (onnxruntime-web)`)
  let spec = ORT_SPEC[backend]
  if (!spec) throw new Error(`neural-runtime: unknown backend '${backend}' (want 'auto', 'node', 'wasm', 'webgpu', or 'webgl')`)
  let p = ortCache.get(spec)
  if (!p) {
    p = import(spec).then(m => m.default ?? m).catch(err => {
      ortCache.delete(spec)
      let pkg = spec.split('/')[0]
      throw new Error(`neural-runtime: ${pkg} is required for backend '${backend}' — install it: npm install ${pkg} (${err.message})`)
    })
    ortCache.set(spec, p)
  }
  return p
}

// data ctor → onnx type, for tensor() when type is omitted (spec: DataTypeMap
// in onnxruntime-common/lib/tensor.ts — the four ctors we can infer unambiguously).
const TYPE_BY_CTOR = new Map([
  [Float32Array, 'float32'],
  [Int32Array, 'int32'],
  [BigInt64Array, 'int64'],
  [Uint8Array, 'uint8'],
])

// tensor(data, dims, type?) → plain { data, dims, type } — no ORT dependency at
// call time. run() converts it to ort.Tensor lazily, once a backend is loaded.
export function tensor(data, dims, type) {
  if (!dims) throw new Error('neural-runtime: tensor() needs dims')
  if (!type) {
    type = TYPE_BY_CTOR.get(data?.constructor)
    if (!type && Array.isArray(data) && data.every(v => typeof v === 'string')) type = 'string'
    if (!type) throw new Error(`neural-runtime: tensor() can't infer a type for ${data?.constructor?.name ?? typeof data} — pass type explicitly`)
  }
  return { data, dims, type }
}

function toOrtTensor(ort, t) {
  if (!t || t.data == null || !t.dims || !t.type)
    throw new Error('neural-runtime: feed tensor needs { data, dims, type } — build it with tensor()')
  return new ort.Tensor(t.type, t.data, t.dims)
}

function fromOrtTensor(t) {
  return { data: t.data, dims: Array.from(t.dims), type: t.type }
}

// session.input/outputMetadata (name, isTensor, type, shape) landed after
// session.input/outputNames in onnxruntime-common — fall back to names only
// when the installed version doesn't expose it.
function ioMeta(session, kind) {
  let names = session[kind + 'Names'] || []
  let meta = session[kind + 'Metadata']
  if (!meta) return names.map(name => ({ name }))
  return meta.map((m, i) => (m.isTensor
    ? { name: m.name ?? names[i], dims: Array.from(m.shape), type: m.type }
    : { name: m.name ?? names[i] }))
}

// backends() → what onnx runtime this environment can actually use.
export async function backends() {
  if (isNode) return ['node']
  let list = ['wasm']
  if (typeof navigator !== 'undefined' && navigator.gpu) list.push('webgpu')
  return list
}

async function resolveBytes(model, opts) {
  if (model instanceof Uint8Array) return model
  if (model instanceof ArrayBuffer) return new Uint8Array(model)
  if (typeof model === 'string') return fetchModel(model, opts)
  throw new Error('neural-runtime: load() expects a URL string, Uint8Array, or ArrayBuffer')
}

// load(model, opts) → Session. model: URL string | Uint8Array | ArrayBuffer.
export async function load(model, opts = {}) {
  let bytes = await resolveBytes(model, opts)
  let backend = resolveBackend(opts.backend)
  let ort = await getOrt(backend)

  if (opts.wasmPaths !== undefined && ort.env?.wasm) ort.env.wasm.wasmPaths = opts.wasmPaths
  if (opts.threads !== undefined && ort.env?.wasm) ort.env.wasm.numThreads = opts.threads
  if (opts.logLevel !== undefined && ort.env) ort.env.logLevel = opts.logLevel

  let sessionOpts = {}
  let eps = opts.executionProviders ?? (backend === 'node' ? undefined : [backend])
  if (eps) sessionOpts.executionProviders = eps
  if (opts.graphOptimizationLevel) sessionOpts.graphOptimizationLevel = opts.graphOptimizationLevel

  let session
  try {
    session = await ort.InferenceSession.create(bytes, sessionOpts)
  } catch (err) {
    throw new Error(`neural-runtime: failed to load model — ${err.message}`)
  }

  let freed = false
  let queue = Promise.resolve() // serializes run() so concurrent callers never overlap a session.run()

  return {
    inputs: ioMeta(session, 'input'),
    outputs: ioMeta(session, 'output'),
    backend,
    async run(feeds, outputNames) {
      if (freed) throw new Error('neural-runtime: run() called after free()')
      let ortFeeds = {}
      for (let k in feeds) ortFeeds[k] = toOrtTensor(ort, feeds[k])
      let call = () => (outputNames ? session.run(ortFeeds, outputNames) : session.run(ortFeeds))
      let p = queue.then(call, call)
      queue = p.then(() => {}, () => {})
      let raw = await p
      let out = {}
      for (let k in raw) out[k] = fromOrtTensor(raw[k])
      return out
    },
    free() {
      if (freed) return
      freed = true
      session.release().catch(() => {})
    },
  }
}

// --- model fetching + caching -------------------------------------------
//
// Node: ~/.cache/audiojs/neural/<sha256(url)>, override with $AUDIO_NEURAL_CACHE.
// A sidecar <file>.size records the expected byte count; a mismatch (a
// truncated or otherwise corrupted cache entry) is treated as a miss and
// re-fetched.
// Browser: Cache API, store 'audio-neural', keyed by URL.

async function readResponseBody(res, onLoaded) {
  let body = res.body
  if (!body?.getReader) {
    let bytes = new Uint8Array(await res.arrayBuffer())
    onLoaded(bytes.length)
    return bytes
  }
  let reader = body.getReader()
  let chunks = [], loaded = 0
  for (;;) {
    let { done, value } = await reader.read()
    if (done) break
    chunks.push(value)
    loaded += value.length
    onLoaded(loaded)
  }
  let out = new Uint8Array(loaded), o = 0
  for (let c of chunks) { out.set(c, o); o += c.length }
  return out
}

async function readFileUrl(href) {
  let [{ readFileSync }, { fileURLToPath }] = await Promise.all([import('node:fs'), import('node:url')])
  return new Uint8Array(readFileSync(fileURLToPath(href)))
}

async function cacheFile(url) {
  let [path, os, crypto] = await Promise.all([import('node:path'), import('node:os'), import('node:crypto')])
  let dir = process.env.AUDIO_NEURAL_CACHE || path.join(os.homedir(), '.cache', 'audiojs', 'neural')
  let file = path.join(dir, crypto.createHash('sha256').update(url).digest('hex'))
  return { dir, file }
}

async function readCached(file) {
  let fs = await import('node:fs')
  if (!fs.existsSync(file) || !fs.existsSync(file + '.size')) return null
  let expected = Number(fs.readFileSync(file + '.size', 'utf8'))
  let bytes = fs.readFileSync(file)
  if (bytes.length !== expected) return null // corrupted (or partial) — re-fetch
  return new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength)
}

async function writeCached(dir, file, bytes) {
  let fs = await import('node:fs')
  fs.mkdirSync(dir, { recursive: true })
  fs.writeFileSync(file, bytes)
  fs.writeFileSync(file + '.size', String(bytes.length))
}

async function fetchNode(url, { cache, progress, fetch: f }) {
  let dir, file
  if (cache) {
    ({ dir, file } = await cacheFile(url))
    let cached = await readCached(file)
    if (cached) { progress?.({ loaded: cached.length, total: cached.length }); return cached }
  }
  let res = await f(url)
  if (!res.ok) throw new Error(`neural-runtime: fetch failed for ${url}: ${res.status} ${res.statusText}`)
  let total = Number(res.headers.get('content-length')) || null
  let bytes = await readResponseBody(res, loaded => progress?.({ loaded, total }))
  if (cache) await writeCached(dir, file, bytes)
  return bytes
}

async function fetchBrowser(url, { cache, progress, fetch: f }) {
  let store = cache && typeof caches !== 'undefined' ? await caches.open('audio-neural') : null
  if (store) {
    let hit = await store.match(url)
    if (hit) {
      let buf = new Uint8Array(await hit.arrayBuffer())
      progress?.({ loaded: buf.length, total: buf.length })
      return buf
    }
  }
  let res = await f(url)
  if (!res.ok) throw new Error(`neural-runtime: fetch failed for ${url}: ${res.status} ${res.statusText}`)
  if (store) await store.put(url, res.clone())
  let total = Number(res.headers.get('content-length')) || null
  return readResponseBody(res, loaded => progress?.({ loaded, total }))
}

// fetchModel(url, opts) → Uint8Array, cached with progress. Also the primitive
// behind fetchJson, for tokenizer/config siblings hosted next to a model.
export async function fetchModel(url, opts = {}) {
  let href = String(url)
  let { cache = true, progress, fetch: fetchFn } = opts
  if (href.startsWith('file://')) {
    if (!isNode) throw new Error('neural-runtime: file:// URLs are only readable in Node')
    let bytes = await readFileUrl(href)
    progress?.({ loaded: bytes.length, total: bytes.length })
    return bytes
  }
  let f = fetchFn || globalThis.fetch
  if (!f) throw new Error('neural-runtime: no fetch available in this environment — pass opts.fetch')
  return isNode ? fetchNode(href, { cache, progress, fetch: f }) : fetchBrowser(href, { cache, progress, fetch: f })
}

// fetchJson(url, opts) → parsed JSON, same cached-fetch path as fetchModel.
export async function fetchJson(url, opts) {
  let bytes = await fetchModel(url, opts)
  return JSON.parse(new TextDecoder().decode(bytes))
}
