// A voice restored to a studio take: the network trained in scripts/ (a ConvNeXt stack with self-attention over STFT
// frames, 17.9M parameters, its STFT outside the graph), its STFT, chunks and level made here as scripts/infer.py makes
// them, the network through @audio/neural-runtime. 48 kHz in and out; other rates are resampled in and back.

import { fft, ifft } from 'fourier-transform'
import resample from '@audio/resample-sinc'
import { load, tensor, fetchModel, backends } from '@audio/neural-runtime'

/** The model's rate, frame and hop (scripts/dsp.py), the level it hears its input at, dBFS RMS */
export const SAMPLE_RATE = 48000, N = 2048, HOP = 512, BINS = N / 2 + 1, LEVEL = -20

// Runs of 10 s overlapping by 1 s, raised-cosine crossfaded (scripts/infer.py)
const CHUNK = 10 * SAMPLE_RATE, OVERLAP = SAMPLE_RATE

/** The file scripts/export.py writes: float16 weights, float32 compute */
export const FILE = 'restore.fp16.onnx'

// The hosted copy: audiojs/restore at REVISION, checked against its SHA-256; in Node kept in the neural cache's
// restore/restore.fp16.onnx, where scripts/export.py writes it too, read when of this SHA-256. While REVISION is empty
// MODEL is null, and the model is opts.weights or that copy
export const REVISION = '', SHA256 = '46eb6ebb05cc50bb6c3b1af87e3c4542bf30cedbcc356e2c68646d3bbf9a65f1'
const REPO = 'audiojs/restore'
export const MODEL = REVISION ? new URL(FILE, `https://huggingface.co/${REPO}/resolve/${REVISION}/`) : null

const isNode = globalThis.process?.release?.name === 'node'
const isUrl = s => /^[a-z][a-z0-9+.-]+:/i.test(s)

const hann = Float64Array.from({ length: N }, (_, i) => 0.5 - 0.5 * Math.cos(2 * Math.PI * i / N))  // torch.hann_window: periodic
const reflect = (i, n) => { if (n === 1) return 0; let p = 2 * (n - 1); i = ((i % p) + p) % p; return i < n ? i : p - i }  // torch 'reflect'

/** torch.stft(x, 2048, 512, window=hann, center=True, pad_mode='reflect') of mono `x`: Float32Array(2 × 1025 × T), re then im, bin-major — the graph's [1, 2, 1025, T] */
export function stft(x) {
  let L = x.length, T = 1 + Math.floor(L / HOP), frame = new Float64Array(N), out = new Float32Array(2 * BINS * T)
  for (let t = 0; t < T; t++) {
    let at = t * HOP - N / 2
    for (let i = 0; i < N; i++) frame[i] = (at + i >= 0 && at + i < L ? x[at + i] : x[reflect(at + i, L)]) * hann[i]
    let [re, im] = fft(frame)
    for (let k = 0; k < BINS; k++) { out[k * T + t] = re[k]; out[(BINS + k) * T + t] = im[k] }
  }
  return { data: out, frames: T }
}

/** torch.istft of the graph's output (as stft() lays it out): `length` samples */
export function istft(spec, T, length) {
  let out = new Float64Array((T - 1) * HOP + N), env = new Float64Array(out.length), fr = new Float64Array(BINS), fi = new Float64Array(BINS)
  for (let t = 0; t < T; t++) {
    for (let k = 0; k < BINS; k++) { fr[k] = spec[k * T + t]; fi[k] = spec[(BINS + k) * T + t] }
    let y = ifft(fr, fi)
    for (let i = 0, at = t * HOP; i < N; i++) { out[at + i] += y[i] * hann[i]; env[at + i] += hann[i] * hann[i] }
  }
  let y = new Float32Array(length)
  for (let i = 0, j = N / 2; i < length && j < out.length; i++, j++) if (env[j] > 1e-11) y[i] = out[j] / env[j]
  return y
}

// The model's bytes: opts.weights (bytes, a URL, or in Node a file or a directory holding restore.fp16.onnx), else in
// Node the neural cache's copy when of its SHA-256, else the hosted file, refused unless of its SHA-256
async function bytesOf(opts) {
  let w = opts.weights
  if (w instanceof Uint8Array || w instanceof ArrayBuffer) return new Uint8Array(w)
  if (w == null) {
    let store = isNode ? await cached() : null, bytes = store?.read()
    if (bytes && await sha256(bytes) === SHA256) return bytes
    if (!MODEL) throw new Error(`neural-restore: ${FILE} is not hosted yet (REVISION unset)${store ? ` and ${store.dest} holds no copy of it` : ''}: pass opts.weights, a URL${isNode ? ', a path' : ''} or the bytes of ${FILE} (scripts/export.py makes it)`)
    try { bytes = await fetchModel(MODEL.href, { progress: opts.progress, fetch: opts.fetch, cache: !store }) }
    catch (e) { throw new Error(`neural-restore: ${FILE} not read (${e.message}): opts.weights takes a URL holding a copy`) }
    let hex = await sha256(bytes)
    if (hex !== SHA256) throw new Error(`neural-restore: ${MODEL} has SHA-256 ${hex}, expected ${SHA256}: a partial or altered download`)
    store?.write(bytes)
    return bytes
  }
  let href = String(w)
  if (!isUrl(href)) {
    if (!isNode) throw new Error(`neural-restore: opts.weights must be a URL in the browser, got '${href}'`)
    let fs = await import('node:fs')
    if (fs.existsSync(href) && fs.statSync(href).isDirectory()) href = (await import('node:path')).join(href, FILE)
    if (!fs.existsSync(href)) throw new Error(`neural-restore: no model at ${href}: scripts/export.py writes it, or pass opts.weights`)
    let b = fs.readFileSync(href); return new Uint8Array(b.buffer, b.byteOffset, b.length)
  }
  if (!/\.onnx(\?|$)/.test(href)) href = (href.endsWith('/') ? href : href + '/') + FILE
  return fetchModel(href, { progress: opts.progress, fetch: opts.fetch })
}

const subtle = async () => globalThis.crypto?.subtle ?? (await import('node:crypto')).webcrypto.subtle
const sha256 = async bytes => Array.from(new Uint8Array(await (await subtle()).digest('SHA-256', bytes)), b => b.toString(16).padStart(2, '0')).join('')

// <neural cache>/restore/restore.fp16.onnx: { dest, read() → bytes or null, write(bytes) whole or not at all }
async function cached() {
  let [fs, path, os] = await Promise.all([import('node:fs'), import('node:path'), import('node:os')])
  let dest = path.join(process.env.AUDIO_NEURAL_CACHE || path.join(os.homedir(), '.cache', 'audiojs', 'neural'), 'restore', FILE)
  return {
    dest,
    read() { try { let b = fs.readFileSync(dest); return new Uint8Array(b.buffer, b.byteOffset, b.length) } catch { return null } },
    write(bytes) {
      let part = `${dest}.${process.pid}.${Math.random().toString(36).slice(2)}.part`
      fs.mkdirSync(path.dirname(dest), { recursive: true })
      fs.writeFileSync(part, bytes)
      fs.renameSync(part, dest)
    },
  }
}

// 'auto': the GPU in a page that has one (WebGPU), else wasm; Node's own runtime in Node
async function backendOf(want) {
  if (want && want !== 'auto') return want
  if (isNode) return 'node'
  return (await backends()).includes('webgpu') ? 'webgpu' : 'wasm'
}

const sessions = new Map()
/** The network, loaded once per weights and backend: { run(spec, T) → Float32Array }, as stft() lays it out */
export function session(opts = {}) {
  let w = opts.weights, key = `${w instanceof Uint8Array || w instanceof ArrayBuffer ? 'bytes' : w ?? ''}:${opts.backend ?? 'auto'}`
  if (!sessions.has(key)) sessions.set(key, (async () => {
    let [bytes, backend] = await Promise.all([bytesOf(opts), backendOf(opts.backend)])
    let s = await load(bytes, { backend })
    return { backend, run: async (spec, T) => (await s.run({ spec: tensor(spec, [1, 2, BINS, T]) })).spec_out.data }
  })().catch(e => { sessions.delete(key); throw e }))
  return sessions.get(key)
}

const fade = Float32Array.from({ length: OVERLAP }, (_, i) => 0.5 - 0.5 * Math.cos(Math.PI * i / OVERLAP))

/**
 * `x` (mono samples at `sampleRate`) restored: brought to −20 dBFS RMS, in runs of 10 s overlapping by 1 s, the gain
 * undone, at `sampleRate` again and as long. opts: { sampleRate = 48000, weights, backend, progress, fetch }.
 */
export default async function restore(x, opts = {}) {
  let fs = opts.sampleRate ?? SAMPLE_RATE
  if (!(x instanceof Float32Array)) x = Float32Array.from(x)
  if (!x.length) return new Float32Array(0)
  let s = await session(opts)
  let v = fs === SAMPLE_RATE ? x : resample(x, { from: fs, to: SAMPLE_RATE }), n = v.length, e = 0
  for (let i = 0; i < n; i++) e += v[i] * v[i]
  let g = 10 ** ((LEVEL - 10 * Math.log10(e / n + 1e-12)) / 20)
  let y = new Float64Array(n), w = new Float64Array(n), step = CHUNK - OVERLAP
  for (let at = 0; at < n; at += step) {
    let len = Math.min(CHUNK, n - at), seg = new Float32Array(len)
    for (let i = 0; i < len; i++) seg[i] = v[at + i] * g
    let { data, frames } = stft(seg), out = istft(await s.run(data, frames), frames, len), last = at + CHUNK >= n
    for (let i = 0; i < len; i++) {
      let k = at > 0 && i < OVERLAP ? fade[i] : !last && i >= len - OVERLAP ? 1 - fade[i - (len - OVERLAP)] : 1
      y[at + i] += out[i] * k; w[at + i] += k
    }
    opts.progress?.({ done: Math.min(n, at + len), total: n })
    if (last) break
  }
  let z = new Float32Array(n)
  for (let i = 0; i < n; i++) z[i] = y[i] / Math.max(w[i], 1e-6) / g
  if (fs === SAMPLE_RATE) return z
  z = resample(z, { from: SAMPLE_RATE, to: fs })
  if (z.length === x.length) return z
  let out = new Float32Array(x.length); out.set(z.subarray(0, x.length)); return out
}
