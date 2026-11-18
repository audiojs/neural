// Against PyTorch: fixtures/ are scripts/reference.py's (10.5 s of degraded speech at 48 kHz, the STFT of its first
// 4096 samples, and scripts/infer.py's restoration of it by the checkpoint the export was made from). The network's
// tests read the export from the neural cache (scripts/export.py writes it there) or from the hosting directory
// prepared for its upload, else the hosted copy once REVISION is set; without any they are skipped.
import test, { ok, is, rejects } from 'tst'
import { existsSync, readFileSync, writeFileSync, mkdirSync, mkdtempSync, rmSync } from 'node:fs'
import { createHash } from 'node:crypto'
import os from 'node:os'
import path from 'node:path'
import restore, { stft, istft, BINS, HOP, FILE, MODEL, REVISION, SHA256 } from './restore.js'

const f32 = name => new Float32Array(readFileSync(new URL(`./fixtures/${name}`, import.meta.url)).buffer.slice(0))
const x = f32('in.f32'), SPEC = f32('spec.f32'), OUT = f32('out.f32')
const CACHE = process.env.AUDIO_NEURAL_CACHE || path.join(os.homedir(), '.cache', 'audiojs', 'neural')
const CACHED = path.join(CACHE, 'restore', FILE), HOSTING = path.join(CACHE, 'hosting', 'restore', FILE)
const FP32 = path.join(CACHE, 'restore', 'restore.onnx'), FP16 = [CACHED, HOSTING].find(existsSync)
const W = FP16 ? { weights: FP16 } : REVISION ? {} : null
const model = W ? test : test.skip
if (!W) console.log(`  no ${FILE} in ${CACHE}/restore or ${CACHE}/hosting/restore, and REVISION unset: the model's tests are skipped (scripts/export.py writes it)`)
const RUN = { timeout: 300_000 }
const snr = (ref, y) => { let s = 0, e = 0; for (let i = 0; i < ref.length; i++) s += ref[i] * ref[i], e += (ref[i] - y[i]) ** 2; return 10 * Math.log10(s / e) }
const sha = file => createHash('sha256').update(readFileSync(file)).digest('hex')
// a module of its own, its sessions empty: the default weights read again, from the neural cache set at the time
const fresh = async () => (await import(`./restore.js?${Math.random()}`)).default
// AUDIO_NEURAL_CACHE pointed at an empty directory while fn runs
async function emptyCache(fn) {
  let dir = mkdtempSync(path.join(os.tmpdir(), 'neural-restore-')), was = process.env.AUDIO_NEURAL_CACHE
  process.env.AUDIO_NEURAL_CACHE = dir
  try { await fn(dir) }
  finally { was == null ? delete process.env.AUDIO_NEURAL_CACHE : process.env.AUDIO_NEURAL_CACHE = was; rmSync(dir, { recursive: true }) }
}

test('stft: torch.stft\'s, centered, reflect-padded, periodic Hann', () => {
  let { data, frames } = stft(x.subarray(0, 4096))
  is(frames, 1 + Math.floor(4096 / HOP))
  is(data.length, 2 * BINS * frames)
  ok(snr(SPEC, data) > 100, `${snr(SPEC, data).toFixed(1)} dB from torch`)
})

test('istft: stft\'s inverse, the length asked', () => {
  let seg = x.subarray(0, 48000), { data, frames } = stft(seg), y = istft(data, frames, seg.length)
  is(y.length, seg.length)
  ok(snr(seg, y) > 100, `round trip ${snr(seg, y).toFixed(1)} dB`)
})

;(existsSync(FP32) ? test : test.skip)('restore: scripts/infer.py\'s output, two runs crossfaded, float32 weights', RUN, async () => {
  let y = await restore(x, { weights: FP32 })
  is(y.length, x.length, 'as long as the input')
  ok(snr(OUT, y) > 60, `${snr(OUT, y).toFixed(1)} dB from PyTorch`)
})

// no weights: the neural cache's copy when of SHA256, else the hosted file
;(existsSync(CACHED) || REVISION ? test : test.skip)('restore: float16 weights, the default file', RUN, async () => {
  let y = await restore(x)
  ok(snr(OUT, y) > 50, `${snr(OUT, y).toFixed(1)} dB from PyTorch`)
})

model('restore: another rate in, the same rate and length out', RUN, async () => {
  let n = 44100 * 3, a = new Float32Array(n)
  for (let i = 0; i < n; i++) a[i] = x[Math.floor(i * 48000 / 44100)]
  let y = await restore(a, { ...W, sampleRate: 44100 })
  is(y.length, n)
  ok(y.every(Number.isFinite) && y.some(v => v !== 0), 'finite, not silent')
})

test('restore: empty in, empty out', async () => is((await restore(new Float32Array(0))).length, 0))

test('restore: without weights and without a hosted revision, an error that says what to pass', async () => {
  if (REVISION) return console.log('  REVISION is set: the hosted file is read without weights (tested below)')
  is(MODEL, null)
  await emptyCache(async dir => {
    let run = await fresh(), seg = x.subarray(0, 4800)
    await rejects(() => run(seg), /not hosted yet.*opts\.weights/, 'an empty cache')
    mkdirSync(path.join(dir, 'restore')), writeFileSync(path.join(dir, 'restore', FILE), new Uint8Array(1024))
    await rejects(() => run(seg), /holds no copy of it/, 'a file of another SHA-256 in the cache is not read')
  })
})

test('model: the files tested and hosted are the one SHA256 names', () => {
  let local = [CACHED, HOSTING].filter(existsSync)
  if (!local.length) return console.log('  no local copy: skipped')
  for (let file of local) is(sha(file), SHA256, file)
})

// The hosted file (Hugging Face, the revision pinned). Network: skipped offline or until REVISION is set
let online
const offline = async () => !(online ??= await fetch('https://huggingface.co/', { method: 'HEAD', signal: AbortSignal.timeout(8000) }).then(r => r.ok, () => false)) && (console.log('  offline: skipped'), true)
test('restore: the hosted file is pinned to its revision and SHA-256', async () => {
  if (!REVISION) return console.log('  REVISION unset (not uploaded yet): skipped')
  if (await offline()) return
  let res = await fetch(MODEL, { method: 'HEAD', redirect: 'manual' })
  is(res.headers.get('x-linked-etag')?.replace(/"/g, ''), SHA256, String(MODEL))
})

test('restore: no weights and an empty cache, the hosted copy is fetched into the cache and runs', RUN, async () => {
  if (!REVISION) return console.log('  REVISION unset (not uploaded yet): skipped')
  if (await offline()) return
  await emptyCache(async dir => {
    let y = await (await fresh())(x)
    ok(snr(OUT, y) > 50, `${snr(OUT, y).toFixed(1)} dB from PyTorch, on the fetched copy`)
    is(sha(path.join(dir, 'restore', FILE)), SHA256, 'kept whole in the cache')
  })
})
