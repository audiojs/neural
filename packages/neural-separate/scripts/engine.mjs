#!/usr/bin/env node
// roformer.js, the package's own WebGPU engine for Mel-Band RoFormer, against onnxruntime on the same weights: Chromium on
// the GPU (playwright's 'chromium' channel: Metal here) runs the engine on one segment of scripts/reference.py's mix,
// onnxruntime-node's CPU the float16 ONNX file on the same spectrum; prints how far apart they are and the engine's time.
//
//   node scripts/engine.mjs            (needs playwright, onnxruntime-node, and in the neural cache, mel-roformer/:
//                                       export-roformer.py --verify --engine, compact.py --model mel-roformer --as fp16)
import { createServer } from 'node:http'
import { readFileSync } from 'node:fs'
import path from 'node:path'
import os from 'node:os'
import { fileURLToPath } from 'node:url'
import { chromium } from 'playwright'
import * as ort from 'onnxruntime-node'
import { stft } from '../separate.js'

const HERE = path.dirname(fileURLToPath(import.meta.url)), CACHE = path.join(process.env.AUDIO_NEURAL_CACHE || path.join(os.homedir(), '.cache', 'audiojs', 'neural'), 'mel-roformer')
const F = 1025, T = 801, L = 352800, f32 = f => { let b = readFileSync(f); return new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4) }

// one segment's spectrum, as separate.js's complexSegment makes it: [L re, L im, R re, R im][f][t]
const mix = f32(path.join(CACHE, 'test.f32')), N = mix.length / 2, spec = new Float32Array(4 * F * T)
for (let c = 0; c < 2; c++) {
  let { re, im } = stft(Float64Array.from(mix.subarray(c * N, c * N + L)), { n: 2048, hop: 441 })
  for (let f = 0; f < F; f++) for (let t = 0; t < T; t++) { spec[(2 * c * F + f) * T + t] = re[t][f]; spec[((2 * c + 1) * F + f) * T + t] = im[t][f] }
}
let cpu = await ort.InferenceSession.create(path.join(CACHE, 'mel-roformer.fp16.onnx'))
let want = (await cpu.run({ mix_spec: new ort.Tensor('float32', spec, [1, 4, F, T]) })).stems_spec.data

const files = { '/roformer.js': path.join(HERE, '..', 'roformer.js'), '/meta': path.join(CACHE, 'mel-roformer.engine.json'), '/bin': path.join(CACHE, 'mel-roformer.engine.bin') }
const server = createServer((q, r) => {
  if (q.url === '/') return r.writeHead(200, { 'content-type': 'text/html' }).end('<!doctype html><title>engine</title>')
  if (q.url === '/spec') return r.writeHead(200).end(Buffer.from(spec.buffer))
  if (!files[q.url]) return r.writeHead(404).end()
  r.writeHead(200, { 'content-type': q.url.endsWith('.js') ? 'text/javascript' : 'application/octet-stream' }).end(readFileSync(files[q.url]))
})
await new Promise(r => server.listen(0, '127.0.0.1', r))
const browser = await chromium.launch({ channel: 'chromium', args: ['--enable-unsafe-webgpu'] }), page = await browser.newPage()
await page.goto(`http://127.0.0.1:${server.address().port}/`)
const got = await page.evaluate(async () => {
  const { engine } = await import('/roformer.js')
  const meta = await (await fetch('/meta')).json(), bytes = new Uint8Array(await (await fetch('/bin')).arrayBuffer())
  let t = performance.now()
  const s = await engine(meta, bytes)
  if (!s) return { none: true }
  const ready = performance.now() - t, spec = new Float32Array(await (await fetch('/spec')).arrayBuffer()), times = []
  let out
  for (let k = 0; k < 4; k++) { t = performance.now(); out = (await s.run({ mix_spec: { data: spec } })).stems_spec.data; times.push(performance.now() - t) }
  return { ready, times, out: [...out] }
})
await browser.close(); server.close()
if (got.none) console.log('no GPU with subgroup matrices here: the engine stands aside for onnxruntime')
else {
  let e = 0, p = 0
  for (let i = 0; i < want.length; i++) { e += (got.out[i] - want[i]) ** 2; p += want[i] ** 2 }
  console.log(`engine ready in ${(got.ready / 1000).toFixed(1)} s; a segment ${got.times.map(t => t.toFixed(0)).join(', ')} ms; ` +
    `${(10 * Math.log10(p / e)).toFixed(1)} dB from onnxruntime's CPU on the same weights`)
}
process.exit(0)
