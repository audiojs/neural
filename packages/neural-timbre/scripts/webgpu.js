#!/usr/bin/env node
// The encoder in headless Chromium, on WebGPU and WASM, through this package, @audio/neural-runtime
// and onnxruntime-web: embeddings of scripts/export.py's parity fixtures against PyTorch's (largest
// difference, identical top-5).
//
//   node scripts/webgpu.js [--chrome path] [--weights dir]
//
// Needs onnxruntime-web and playwright-core resolvable from here, and a Chromium with WebGPU
// (--chrome, else Playwright's own). weights: default ~/.cache/audiojs/neural.
import http from 'node:http'
import { readFile } from 'node:fs/promises'
import path from 'node:path'
import os from 'node:os'
import { fileURLToPath } from 'node:url'

let o = {}
for (let a = process.argv.slice(2), i = 0; i < a.length; i += 2) o[a[i].replace(/^--/, '')] = a[i + 1]
const dirOf = spec => path.dirname(fileURLToPath(import.meta.resolve(spec)))
const roots = {
	pkg: path.join(path.dirname(fileURLToPath(import.meta.url)), '..'),
	runtime: dirOf('@audio/neural-runtime'), fft: dirOf('fourier-transform'), sinc: dirOf('@audio/resample-sinc'),
	ort: path.join(dirOf('onnxruntime-web'), '..'), // dist/
	cache: path.join(o.weights ?? process.env.AUDIO_NEURAL_CACHE ?? path.join(os.homedir(), '.cache/audiojs/neural'), 'timbre'),
}

const PAGE = `<!doctype html><meta charset="utf-8"><title>neural-timbre</title>
<script type="importmap">{ "imports": {
	"onnxruntime-web/webgpu": "/ort/dist/ort.webgpu.bundle.min.mjs", "onnxruntime-web": "/ort/dist/ort.wasm.bundle.min.mjs",
	"@audio/neural-runtime": "/runtime/runtime.js", "@audio/resample-sinc": "/sinc/sinc.js", "fourier-transform": "/fft/index.js" } }</script>
<script type="module">
import { createEncoder, rank } from '/pkg/timbre.js'
let out = { adapter: !!(navigator.gpu && await navigator.gpu.requestAdapter()) }
let fx = await (await fetch('/cache/parity.json')).json(), { bands, frames } = fx.features, size = bands * frames
let raw = Uint8Array.from(atob(fx.patches), c => c.charCodeAt(0)), half = new Float16Array(raw.buffer)
let patches = Array.from({ length: fx.rows.length }, (_, i) => Float32Array.from(half.subarray(i * size, (i + 1) * size)))
for (let name of ['encoder.onnx', 'encoder.int8.onnx']) for (let backend of ['webgpu', 'wasm']) {
	try {
		let enc = await createEncoder({ model: new Uint8Array(await (await fetch('/cache/' + name)).arrayBuffer()), backend })
		await enc.project(patches.slice(0, 1))
		let t = performance.now(), e = await enc.project(patches), ms = performance.now() - t
		enc.free()
		let want = fx[name], max = 0
		e.forEach((v, i) => v.forEach((x, d) => { max = Math.max(max, Math.abs(x - want.embeddings[i][d])) }))
		let cat = { vectors: Float32Array.from(want.catalog.flat()), dimensions: want.catalog[0].length }
		let same = e.filter((v, i) => rank(v, cat, 5).map(r => r.index).join() === want.top5[i].join()).length
		out[name + ' ' + backend] = { maxDiff: +max.toExponential(2), top5: same + '/' + e.length, msPerNote: +(ms / e.length).toFixed(2) }
	} catch (err) { out[name + ' ' + backend] = { error: String(err.message).slice(0, 300) } }
}
window.result = out; document.title = 'done'
</script>`

const TYPES = { '.js': 'text/javascript', '.mjs': 'text/javascript', '.json': 'application/json', '.wasm': 'application/wasm' }
const server = http.createServer(async (req, res) => {
	if (req.url === '/') { res.writeHead(200, { 'content-type': 'text/html' }); return res.end(PAGE) }
	let [, root, ...rest] = decodeURIComponent(req.url.split('?')[0]).split('/')
	try {
		let file = path.join(roots[root], ...rest), body = await readFile(file)
		res.writeHead(200, { 'content-type': TYPES[path.extname(file)] ?? 'application/octet-stream' }); res.end(body)
	} catch { res.writeHead(404); res.end() }
})
await new Promise(r => server.listen(0, r))
let { chromium } = await import('playwright-core')
let browser = await chromium.launch({ executablePath: o.chrome, headless: true, args: ['--enable-unsafe-webgpu'] })
try {
	let page = await browser.newPage()
	await page.goto(`http://localhost:${server.address().port}/`)
	await page.waitForFunction(() => document.title === 'done', null, { timeout: 600000 })
	console.log(JSON.stringify(await page.evaluate(() => window.result), null, 1))
} finally { await browser.close(); server.close() }
