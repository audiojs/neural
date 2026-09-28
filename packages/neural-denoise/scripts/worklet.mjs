// The Speed section's browser numbers: headless Chromium (Playwright) serves this package and measures
//   1. the worklet in an OfflineAudioContext (60 s of lena + noise, as fast as it renders),
//   2. the frame API on the page thread, with the SIMD kernel and without (1000 frames after 100),
//   3. the worklet live in a 48 kHz AudioContext, twice, each against a control without it: AudioContext.playbackStats'
//      underruns (base latency 5.3 ms: 256-sample buffers).
//
//   node scripts/worklet.mjs [seconds] [live seconds]      (npm install playwright; npx playwright install chromium)

import { createServer } from 'node:http'
import { readFile } from 'node:fs/promises'
import { fileURLToPath } from 'node:url'

const PKG = fileURLToPath(new URL('..', import.meta.url)), LENA = fileURLToPath(import.meta.resolve('audio-lena/raw')).replace(/raw\.js$/, 'lena.raw')
const PAGE = `<!doctype html><meta charset="utf-8"><script type="module">
import { model, create } from '/rnnoise.js'
window.bench = async ({ seconds, live }) => {
  const w = new Uint8Array(await (await fetch('/rnnoise.bin')).arrayBuffer()), lena = new Float32Array(await (await fetch('/lena.raw')).arrayBuffer()), out = {}
  const n = seconds * 48000, x = new Float32Array(n)
  for (let i = 0, s = 1; i < n; i++) { s = (s * 1664525 + 1013904223) % 4294967296; x[i] = lena[i % lena.length] + 0.05 * (s / 4294967296 - .5) }
  const ctx = new OfflineAudioContext(1, n, 48000)
  await ctx.audioWorklet.addModule('/worklet.js')
  const buf = ctx.createBuffer(1, n, 48000); buf.copyToChannel(x, 0)
  const src = ctx.createBufferSource(); src.buffer = buf
  const node = new AudioWorkletNode(ctx, 'neural-denoise', { processorOptions: { weights: w } }), errors = []
  node.onprocessorerror = e => errors.push(String(e.message || e.type))
  src.connect(node).connect(ctx.destination); src.start()
  const t0 = performance.now(), res = await ctx.startRendering(), ms = performance.now() - t0
  let e = 0; for (const v of res.getChannelData(0)) e += v * v
  out.offline = { seconds, ms: Math.round(ms), msPerFrame: +(ms / (seconds * 100)).toFixed(3), outRms: +Math.sqrt(e / n).toFixed(4), errors }
  for (const simd of [true, false]) {
    const net = model(w, { simd }), st = create(net), o = new Float32Array(480), t = []
    for (let k = 0; k < 1100; k++) {
      const f = x.subarray((k % 1000) * 480, (k % 1000) * 480 + 480).map(v => v * 32768), s = performance.now()
      st.process(f, o); if (k >= 100) t.push(performance.now() - s)
    }
    t.sort((a, b) => a - b)
    out[net.gru1.input.simd ? 'frameWasm' : 'frameJs'] = { meanMs: +(t.reduce((a, b) => a + b) / t.length).toFixed(3), p99Ms: +t[Math.floor(t.length * .99)].toFixed(3), maxMs: +t[t.length - 1].toFixed(3) }
  }
  // live, alternating: the same looped buffer through the worklet, and straight to the output (the control:
  // underruns a loaded machine causes on its own)
  const run = async worklet => {
    const rt = new AudioContext({ sampleRate: 48000 })
    await rt.audioWorklet.addModule('/worklet.js')
    const b2 = rt.createBuffer(1, n, 48000); b2.copyToChannel(x, 0)
    const s2 = rt.createBufferSource(); s2.buffer = b2; s2.loop = true
    const an = rt.createAnalyser(), errors = []; an.fftSize = 32768
    let head = s2
    if (worklet) { head = new AudioWorkletNode(rt, 'neural-denoise', { processorOptions: { weights: w } }); head.onprocessorerror = e => errors.push(String(e.message || e.type)); s2.connect(head) }
    head.connect(an).connect(rt.destination); s2.start(); await rt.resume()
    await new Promise(r => setTimeout(r, live * 1000))
    const probe = new Float32Array(an.fftSize); an.getFloatTimeDomainData(probe)
    let e2 = 0; for (const v of probe) e2 += v * v
    const ps = rt.playbackStats, r = { seconds: +ps?.totalDuration.toFixed(2), underrunEvents: ps?.underrunEvents, underrunMs: +(1000 * ps?.underrunDuration).toFixed(1), outRms: +Math.sqrt(e2 / probe.length).toFixed(4), errors }
    await rt.close()
    return r
  }
  out.live = []
  for (let k = 0; k < 2; k++) out.live.push({ worklet: await run(true), control: await run(false) })
  return out
}
document.title = 'ready'
</script>`

const { chromium } = await import('playwright').catch(() => { throw new Error('needs playwright: npm install playwright && npx playwright install chromium') })
const types = { js: 'text/javascript', html: 'text/html' }
const server = createServer(async (req, res) => {
	let u = decodeURIComponent(req.url.split('?')[0])
	try {
		let body = u === '/' ? PAGE : await readFile(u === '/lena.raw' ? LENA : PKG + u.slice(1))
		res.writeHead(200, { 'content-type': u === '/' ? 'text/html' : types[u.split('.').pop()] || 'application/octet-stream' }); res.end(body)
	} catch { res.writeHead(404); res.end() }
}).listen(0)
const browser = await chromium.launch({ args: ['--autoplay-policy=no-user-gesture-required'] })
try {
	const page = await browser.newPage()
	await page.goto(`http://localhost:${server.address().port}/`)
	await page.waitForFunction(() => document.title === 'ready')
	console.log(JSON.stringify(await page.evaluate(o => window.bench(o), { seconds: +(process.argv[2] || 60), live: +(process.argv[3] || 20) }), null, 1))
	console.log('chromium', browser.version())
} finally { await browser.close(); server.close() }
