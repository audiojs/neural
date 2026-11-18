// Offline: the hosted encoder and catalogs, read from this directory's copies where they are (the
// files uploaded to audiojs/dx7-timbre, in neither git nor npm; else fetched from the pinned
// revision), against ROM1A notes rendered once (fixtures/rom1a.*, fixtures/make-rom1a.js), and
// fixtures/mean-band.onnx, an encoder-shaped graph without weights (fixtures/make.py), for the
// plumbing; the fetch, cache and SHA-256 checks against a local server of the copies. With scripts/export.py's parity.json under
// $AUDIO_NEURAL_CACHE or ~/.cache/audiojs/neural, both exported encoders are checked against
// PyTorch; with @audio/synth-dx7 and the ROM1A cartridge, the fixture against a live render.
import test, { ok, is, almost, throws, rejects } from 'tst'
import { existsSync, readFileSync, writeFileSync, mkdtempSync, rmSync } from 'node:fs'
import http from 'node:http'
import { fileURLToPath } from 'node:url'
import os from 'node:os'
import path from 'node:path'
import identify, { createEncoder, createMatcher, catalog, catalogs, readCatalog, rank, pack, unpack, features, FEATURES, featuresSha256, sha256, MODEL, REVISION } from './timbre.js'
import { cqt, onset } from './features.js'

// resolved before any test registers: tst starts once registrations pause
const dx7 = await import('@audio/synth-dx7').catch(() => null)
const live = dx7 && await import('./fixtures/make-rom1a.js').catch(() => null)

const FIX = new URL('./fixtures/', import.meta.url)
// the copies of the hosted files, laid out as the repository: the weights directory when they are here
const HERE = fileURLToPath(new URL('./', import.meta.url))
const FILES = { encoder: path.join(HERE, 'encoder.int8.onnx'), 'dx7-factory': path.join(HERE, 'catalogs/dx7-factory.json'), 'dx7-all': path.join(HERE, 'catalogs/dx7-all.json') }
const LOCAL = Object.values(FILES).every(f => existsSync(f)), WEIGHTS = LOCAL ? HERE : undefined
const local = LOCAL ? test : test.skip
const CACHE = path.join(process.env.AUDIO_NEURAL_CACHE || path.join(os.homedir(), '.cache', 'audiojs', 'neural'), 'timbre')
const DATA = process.env.TIMBRE_DATA || path.join(os.homedir(), '.cache', 'audiojs', 'data', 'dx7')

// IEEE 754-2008 binary16, little-endian: a sign bit, 5 exponent bits biased by 15, 10 fraction bits; exponent 0 is
// subnormal, 31 infinity or NaN. Read by hand: Float16Array is a global from Node 24 on
const f16 = b => Float32Array.from({ length: b.length >> 1 }, (_, i) => {
	let h = b[2 * i] | b[2 * i + 1] << 8, e = h >> 10 & 31, m = h & 1023
	return (h >> 15 ? -1 : 1) * (e === 31 ? (m ? NaN : Infinity) : (e ? 1024 + m : m) * 2 ** ((e || 1) - 25))
})

// fixtures/rom1a: { voices: [{ index, name, id }], note, featuresSha256 } and one float16 feature patch per voice
const ROM = JSON.parse(readFileSync(new URL('rom1a.json', FIX), 'utf8'))
{
	let half = f16(readFileSync(new URL('rom1a.f16', FIX))), size = FEATURES.bands * FEATURES.frames
	ROM.patches = ROM.voices.map((_, i) => Float32Array.from(half.subarray(i * size, (i + 1) * size)))
}

function signal(n = 70000, fs = 44100) {
	let x = new Float32Array(n)
	for (let i = 0; i < n; i++) x[i] = 0.3 * Math.sin(2 * Math.PI * (100 + 3000 * i / n) * i / fs) + 0.2 * Math.sin(2 * Math.PI * 440 * i / fs) * Math.exp(-i / 20000)
	return x
}

const tone = (f, n = 70000, fs = 44100, harmonics = 1) => Float32Array.from({ length: n }, (_, i) => {
	let s = 0
	for (let h = 1; h <= harmonics; h++) s += Math.sin(2 * Math.PI * f * h * i / fs) / h
	return 0.3 * s * Math.exp(-i / 30000)
})

// Brown (1991) by definition: bin k of frame t is the inner product of the frame with a Hann window
// of round(Q fs / f_k) samples (at most n) times a complex exponential at f_k, over its length.
function direct(x, k, t, f = FEATURES) {
	let fk = f.fmin * 2 ** (k / f.perOctave), Q = 1 / (2 ** (1 / f.perOctave) - 1), len = Math.min(f.n, Math.round(Q * f.sampleRate / fk))
	let c = t * f.hop + f.center, start = c - (len >> 1), re = 0, im = 0
	for (let j = 0; j < len; j++) {
		let v = x[start + j] || 0, w = (0.5 - 0.5 * Math.cos(2 * Math.PI * j / len)) / len, ph = 2 * Math.PI * fk * (j - (len >> 1)) / f.sampleRate
		re += v * w * Math.cos(ph); im -= v * w * Math.sin(ph)
	}
	return 10 * Math.log10(re * re + im * im)
}

test('features: constant-Q by FFT and sparse kernel equals the direct transform (Brown 1991; Brown & Puckette 1992)', () => {
	let x = signal(), out = cqt(x), cells = []
	for (let k = 0; k < FEATURES.bands; k++) for (let t of [2, 40, 100]) {
		let db = (out[k * FEATURES.frames + t] - 1) * FEATURES.top / 2
		if (db > -30) cells.push([direct(x, k, t) - db, k, t])
	}
	ok(cells.length > 20, `${cells.length} cells above -30 dB`)
	// the loudest-cell reference cancels in the differences. The sparse kernel drops spectral tails
	// below 0.54% of each kernel's peak, so cells 45 dB and more under a strong partial carry its
	// dropped leakage; above -30 dB it stays within a quarter of a dB
	let spread = Math.max(...cells.map(c => c[0])) - Math.min(...cells.map(c => c[0]))
	ok(spread < 0.25, `offset spread ${spread.toFixed(4)} dB over ${cells.length} cells`)
}, { timeout: 30000 })

test('features: a tone peaks in its own bin', () => {
	for (let k of [5, 24, 50, 77, 94]) {
		let f = FEATURES.fmin * 2 ** (k / 12), x = Float32Array.from({ length: 70000 }, (_, i) => Math.sin(2 * Math.PI * f * i / 44100))
		let out = cqt(x), t = 60, col = Array.from({ length: FEATURES.bands }, (_, b) => out[b * FEATURES.frames + t])
		is(col.indexOf(Math.max(...col)), k, `${f.toFixed(1)} Hz peaks at bin ${k}`)
	}
})

test('features: onset skips leading silence, level does not matter', () => {
	let x = tone(440), padded = new Float32Array(x.length + 10000)
	padded.set(x, 10000)
	let at = onset(padded)
	ok(at <= 10000 && at >= 10000 - 2 * FEATURES.hop, `onset ${at} within two hops before 10000`)
	let a = features(x), b = features(Float32Array.from(x, v => v * 0.01)), c = features(padded)
	let d1 = Math.max(...a.map((v, i) => Math.abs(v - b[i]))), d2 = a.reduce((s, v, i) => s + Math.abs(v - c[i]), 0) / a.length
	ok(d1 < 1e-5, `-40 dB copy: max |diff| ${d1.toExponential(2)}`)
	ok(features(new Float32Array(70000)).every(v => v === -1), 'silence is the floor')
	ok(d2 < 0.02, `padded copy: mean |diff| ${d2.toFixed(4)}`)
	is(a.length, FEATURES.bands * FEATURES.frames)
	ok(a.every(v => v >= -1 && v <= 1), 'values in [-1, 1]')
})

test('features: resamples other rates, averages channels', () => {
	let x48 = tone(440, 76190, 48000), a = features(x48, { sampleRate: 48000 }), b = features(tone(440))
	let d = a.reduce((s, v, i) => s + Math.abs(v - b[i]), 0) / a.length
	ok(d < 0.05, `48 kHz vs 44.1 kHz tone: mean |diff| ${d.toFixed(4)}`)
	let x = tone(440), stereo = features([x, x]), mono = features(x)
	ok(stereo.every((v, i) => Math.abs(v - mono[i]) < 1e-6), 'identical channels average to the mono features')
	throws(() => features(x, { sampleRate: 0 }))
})

test('catalog: pack/unpack round trip at 8 and 4 bits', () => {
	let v = Int8Array.from([-128, -1, 0, 1, 127, 5, -7])
	ok(unpack(pack(v, 8), 8, v.length).every((x, i) => x === v[i]), 'int8')
	let w = Int8Array.from([-8, -1, 0, 1, 7, 3, -5])
	ok(unpack(pack(w, 4), 4, w.length).every((x, i) => x === w[i]), 'int4')
})

test('catalog: quantized vectors keep ranking, binding is enforced', async () => {
	let binding = { encoderSha256: 'a'.repeat(64), featuresSha256: await featuresSha256() }
	let r = (s => () => (s = (s * 1664525 + 1013904223) >>> 0) / 2 ** 32 - 0.5)(7)
	let vecs = Array.from({ length: 50 }, () => { let v = Float32Array.from({ length: 64 }, r), n = Math.hypot(...v); return v.map(x => x / n) })
	let entries = vecs.map((_, i) => ({ name: 'p' + i }))
	for (let encoding of ['int8-base64', 'int4-base64']) {
		let json = JSON.parse(JSON.stringify(catalog(entries, vecs, binding, { encoding })))
		let c = readCatalog(json, binding)
		is(c.entries.length, 50)
		let hits = vecs.filter((v, i) => rank(v, c, 1)[0].index === i).length
		ok(hits === 50, `${encoding}: each vector finds itself (${hits}/50)`)
	}
	let json = catalog(entries, vecs, binding)
	throws(() => readCatalog(json, { ...binding, encoderSha256: 'b'.repeat(64) }), /different model/)
	throws(() => readCatalog({ ...json, vectors: { ...json.vectors, data: json.vectors.data.slice(8) } }, binding), /truncated|malformed/)
})

test('matcher: encoder-shaped ONNX end to end through @audio/neural-runtime', async () => {
	let model = readFileSync(new URL('mean-band.onnx', FIX))
	let enc = await createEncoder({ model })
	let sounds = { low: tone(110, 70000, 44100, 8), mid: tone(440, 70000, 44100, 3), high: tone(1760) }
	let names = Object.keys(sounds), vecs = await Promise.all(names.map(k => enc.embed(sounds[k])))
	let json = catalog(names.map(name => ({ name })), vecs, enc.binding)
	enc.free()
	let m = await createMatcher(json, { model })
	for (let k of names) {
		let [best] = await m.match(sounds[k])
		is(best.name, k, `${k} finds itself`)
		ok(best.score > 0.99, `score ${best.score.toFixed(4)}`)
	}
	let quiet = await m.match(Float32Array.from(sounds.mid, v => v * 0.1), { top: 3 })
	is(quiet.length, 3); is(quiet[0].name, 'mid', 'level-independent')
	m.free()
	let [best] = await identify(sounds.high, { model, catalog: json })
	is(best.name, 'high', 'identify() one-shot')
	await rejects(() => createMatcher({ ...json, encoderSha256: '0'.repeat(64) }, { model }), /different model/)
	await rejects(() => createMatcher('dx7-heldout', { model }), /no catalog 'dx7-heldout' is hosted \(dx7-factory, dx7-all\)/)
}, { timeout: 60000 })

// f16 against the engine's own binary16 where it has one (Node 24 on): every one of the 65536 patterns
;(typeof Float16Array === 'function' ? test : test.skip)('f16: each binary16 pattern reads as Float16Array reads it', () => {
	let b = new Uint8Array(Uint16Array.from({ length: 65536 }, (_, i) => i).buffer), a = f16(b), want = new Float16Array(b.buffer)
	is(a.findIndex((v, i) => !Object.is(v, want[i])), -1, 'none differ, NaN and -0 included')
})

// ------------------------------------------------ the hosted encoder and catalogs, from local copies

local('hosted: encoder.int8.onnx and catalogs/ are bound to each other and to features.js', async () => {
	let binding = { encoderSha256: await sha256(readFileSync(FILES.encoder)), featuresSha256: await featuresSha256() }
	for (let [name, voices] of [['dx7-factory', 988], ['dx7-all', 29840]]) {
		let json = JSON.parse(readFileSync(FILES[name], 'utf8'))
		is(json.encoderSha256, binding.encoderSha256, `${name}: the hosted encoder's SHA-256`)
		is(json.featuresSha256, binding.featuresSha256, `${name}: the features' SHA-256`)
		let c = readCatalog(json, binding)
		is(c.entries.length, voices, `${name}: ${voices} voices`)
		is(new Set(c.entries.map(e => e.id)).size, voices, `${name}: one entry per sound`)
		ok(c.entries.every(e => typeof e.name === 'string' && e.cartridge && Number.isInteger(e.index) && e.index >= 1 && e.algorithm >= 1 && e.algorithm <= 32), `${name}: entries name a voice of a cartridge`)
	}
	is(ROM.featuresSha256, binding.featuresSha256, 'fixtures/rom1a: prepared by these features')
})

// without the copies here the hosted files are fetched (Hugging Face, the revision pinned): skipped offline
let online
const offline = async () => !(online ??= await fetch('https://huggingface.co/', { method: 'HEAD', signal: AbortSignal.timeout(8000) }).then(r => r.ok, () => false)) && (console.log('  offline: skipped'), true)

// F4 at velocity 90: a note and velocity no catalog reference uses (C2-C6 at 64 and 112)
// dx7-all holds near copies of Yamaha's voices from other cartridges, which may rank first
for (let [name, bar] of [['dx7-factory', 31], ['dx7-all', 24]]) test(`hosted ${name}: ROM1A voices at an unseen note find themselves (fixtures/rom1a)`, async () => {
	if (!LOCAL && await offline()) return
	let m = await createMatcher(name, { weights: WEIGHTS }) // the hosted encoder and catalog by name, from the copies here
	let e = await m.encoder.project(ROM.patches)
	let at = e.map((v, i) => rank(v, m.catalog, 5).findIndex(r => m.catalog.entries[r.index].id === ROM.voices[i].id))
	m.free()
	let top1 = at.filter(k => k === 0).length, top5 = at.filter(k => k >= 0).length
	console.log(`  (${name}, ${m.catalog.entries.length} voices: top-1 ${top1}/32, top-5 ${top5}/32)`)
	ok(top5 >= bar, `top-5 ${top5}/32`)
}, { timeout: 60000 })

test('hosted: identify(note) with no catalog ranks dx7-factory, best first', async () => {
	if (!LOCAL && await offline()) return
	let found = await identify(tone(220, 70000, 44100, 8), { weights: WEIGHTS })
	is(found.length, 5)
	ok(found.every(f => typeof f.name === 'string' && f.cartridge && f.score <= 1 && f.score >= -1), 'dx7-factory entries with scores')
	ok(found.every((f, i) => !i || f.score <= found[i - 1].score), 'best first')
}, { timeout: 60000 })

// ------------------------------------------------ fetched: a server holding a copy, the neural cache, SHA-256

// A server of this directory's copies at /good/ and of the same files with one byte changed at /bad/, and an empty
// neural cache, for the length of fn
async function serve(fn) {
	let requests = [], server = http.createServer((req, res) => {
		let [, kind, ...rest] = req.url.split('/'), file = path.join(HERE, ...rest)
		requests.push(req.url)
		if (!['good', 'bad'].includes(kind) || !Object.values(FILES).includes(file)) { res.writeHead(404); return res.end() }
		let body = readFileSync(file)
		if (kind === 'bad') body[body.length >> 1] ^= 1
		res.end(body)
	})
	await new Promise(r => server.listen(0, '127.0.0.1', r))
	let close = () => new Promise(r => { server.closeAllConnections(); server.close(r) })
	let cache = process.env.AUDIO_NEURAL_CACHE, dir = process.env.AUDIO_NEURAL_CACHE = mkdtempSync(path.join(os.tmpdir(), 'neural-timbre-'))
	try { return await fn({ url: `http://127.0.0.1:${server.address().port}/`, requests, dir, close }) }
	finally {
		if (server.listening) await close()
		if (cache == null) delete process.env.AUDIO_NEURAL_CACHE; else process.env.AUDIO_NEURAL_CACHE = cache
		rmSync(dir, { recursive: true, force: true })
	}
}
const inCache = (dir, key) => path.join(dir, 'dx7-timbre', path.relative(HERE, FILES[key]))
const NOWHERE = 'http://127.0.0.1:9/' // nothing listens

test('hosted: MODEL and catalogs are the files of audiojs/dx7-timbre at REVISION', () => {
	let at = `https://huggingface.co/audiojs/dx7-timbre/resolve/${REVISION}/`
	is(MODEL.href, at + 'encoder.int8.onnx')
	is(catalogs['dx7-factory'].href, at + 'catalogs/dx7-factory.json')
	is(catalogs['dx7-all'].href, at + 'catalogs/dx7-all.json')
})

local('fetched: Node fetches each file once into the neural cache, checked, then reads it from any source offline', () => serve(async ({ url, requests, dir, close }) => {
	let m = await createMatcher('dx7-factory', { weights: url + 'good/' })
	is(m.catalog.entries.length, 988)
	m.free()
	is(requests.slice().sort(), ['/good/catalogs/dx7-factory.json', '/good/encoder.int8.onnx'], 'each fetched once')
	for (let key of ['encoder', 'dx7-factory']) ok(readFileSync(inCache(dir, key)).equals(readFileSync(FILES[key])), `${key}: written whole to ${path.relative(dir, inCache(dir, key))}`)
	await close()
	m = await createMatcher('dx7-factory', { weights: NOWHERE })
	is(m.catalog.entries.length, 988, 'the cache serves, whatever the source: the files are known by their SHA-256')
	m.free()
	is(requests.length, 2, 'no further request')
}), { timeout: 60000 })

local('fetched: a file of another SHA-256 is refused before it is used or cached; a cached file altered is fetched again', () => serve(async ({ url, requests, dir }) => {
	await rejects(() => createEncoder({ weights: url + 'bad/' }), /bad\/encoder\.int8\.onnx has SHA-256 [0-9a-f]{64}, expected aaa9fae0/, 'encoder')
	await rejects(() => createMatcher('dx7-factory', { weights: url + 'bad/', model: FILES.encoder }), /bad\/catalogs\/dx7-factory\.json has SHA-256 [0-9a-f]{64}, expected 7ab9d48c/, 'catalog')
	ok(!existsSync(path.join(dir, 'dx7-timbre')), 'nothing cached')
	let copy = mkdtempSync(path.join(os.tmpdir(), 'neural-timbre-'))
	try {
		let bytes = readFileSync(FILES.encoder)
		bytes[1000] ^= 1
		writeFileSync(path.join(copy, 'encoder.int8.onnx'), bytes)
		await rejects(() => createEncoder({ weights: copy }), /encoder\.int8\.onnx has SHA-256 [0-9a-f]{64}, expected aaa9fae0/, 'a local copy too')
	} finally { rmSync(copy, { recursive: true, force: true }) }
	;(await createEncoder({ weights: url + 'good/' })).free()
	writeFileSync(inCache(dir, 'encoder'), 'not the file')
	let n = requests.length, enc = await createEncoder({ weights: url + 'good/' })
	is(enc.binding.encoderSha256, 'aaa9fae0337325fc9ae0f15c45f711073255c09264a31a9779da635ec21b8484')
	enc.free()
	is(requests.length, n + 1, 'fetched again')
	ok(readFileSync(inCache(dir, 'encoder')).equals(readFileSync(FILES.encoder)), 'and the cache restored')
}), { timeout: 60000 })

test('fetched: nothing cached and nothing reachable names the file and opts.weights', () => serve(async () => {
	await rejects(() => createMatcher('dx7-all', { weights: NOWHERE }), /neural-timbre: encoder\.int8\.onnx not read \(.+\): opts\.weights takes a URL or directory holding a copy of https:\/\/huggingface\.co\/audiojs\/dx7-timbre\/resolve\//)
	await rejects(() => createMatcher('dx7-all', { weights: path.join(os.tmpdir(), 'no-such-dir-neural-timbre') }), /encoder\.int8\.onnx not read \(.*ENOENT/)
}))

// The pinned revision serves the files this version expects, by the default path into an empty cache. Network
;(REVISION === 'main' ? test.skip : test)(`hosted: audiojs/dx7-timbre at ${REVISION} serves each file at its SHA-256${REVISION === 'main' ? " (REVISION is 'main': pin the upload's commit)" : ''}`, () => serve(async () => {
	if (!await fetch('https://huggingface.co/', { method: 'HEAD', signal: AbortSignal.timeout(8000) }).then(r => r.ok, () => false)) return console.log('  offline: skipped')
	for (let name of ['dx7-factory', 'dx7-all']) {
		let m = await createMatcher(name)
		ok(m.catalog.entries.length > 900, `${name}: ${m.catalog.entries.length} voices`)
		m.free()
	}
}), { timeout: 300000 })

// A live render needs @audio/synth-dx7 and Yamaha's ROM1A (scripts/voices.js reads it from $TIMBRE_DATA)
let rom = path.join(DATA, 'factory', 'rom1a.syx')
;(live && existsSync(rom) ? test : test.skip)('fixtures/rom1a: the stored features are a live render of ROM1A through features.js, to float16', () => {
	let voices = dx7.parse(readFileSync(rom)), max = 0
	is(voices.map(v => v.name), ROM.voices.map(v => v.name), 'the same 32 voices')
	voices.forEach((v, i) => features(live.render(v.patch)).forEach((x, j) => { max = Math.max(max, Math.abs(x - ROM.patches[i][j])) }))
	ok(max <= 2 ** -12, `max |diff| ${max.toExponential(2)} (float16 rounds values in [-1, 1] by 2^-12 at most)`)
}, { timeout: 120000 })

// ------------------------------------------------ against PyTorch

const parity = path.join(CACHE, 'parity.json')
for (let [name, file] of [['encoder.onnx', path.join(CACHE, 'encoder.onnx')], ['encoder.int8.onnx (hosted)', FILES.encoder]]) {
	let have = existsSync(file) && existsSync(parity)
	;(have ? test : test.skip)(`parity: ${name} in @audio/neural-runtime matches PyTorch: embeddings within 1e-4, same top-5`, async () => {
		let fx = JSON.parse(readFileSync(parity, 'utf8'))
		let { bands, frames } = fx.features, size = bands * frames
		let half = f16(Buffer.from(fx.patches, 'base64'))
		let patches = Array.from({ length: fx.rows.length }, (_, i) => Float32Array.from(half.subarray(i * size, (i + 1) * size)))
		let enc = await createEncoder({ model: file })
		let e = await enc.project(patches)
		enc.free()
		let want = fx[path.basename(file)], max = 0
		e.forEach((v, i) => v.forEach((x, d) => { max = Math.max(max, Math.abs(x - want.embeddings[i][d])) }))
		ok(max < 1e-4, `max |diff| ${max.toExponential(2)} over ${e.length} fixtures`)
		let cat = { vectors: Float32Array.from(want.catalog.flat()), dimensions: want.catalog[0].length }
		let same = e.filter((v, i) => rank(v, cat, 5).map(r => r.index).join() === want.top5[i].join()).length
		is(same, e.length, `identical top-5 for ${same}/${e.length} queries`)
	}, { timeout: 60000 })
}
