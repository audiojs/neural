// Against PyTorch: fixtures/sound.* are audio's test/fixture.wav (2 s, 44.1 kHz) resampled to 24 kHz by torchaudio,
// its features as Vocos computes them (torchaudio, float32) and Vocos.decode of them (scripts/export.py --reference;
// charactr/vocos-mel-24khz at export.py's REVISION). The network's tests read the model from the neural cache
// (export.py, compact.py, or the hosted copy fetched) and are skipped without it.
import test, { ok, is } from 'tst'
import { existsSync, readFileSync, mkdtempSync, rmSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { fft } from 'fourier-transform'
import resample from '@audio/resample-sinc'
import vocode, { mel, frames, istft, N, HOP, MELS, SAMPLE_RATE, MODEL, SHA256 } from './vocode.js'

const f32 = name => new Float32Array(readFileSync(new URL(`./fixtures/${name}`, import.meta.url)).buffer.slice(0))
const x = f32('sound.24k.f32'), MEL = f32('sound.mel.f32'), DECODED = f32('sound.vocos.f32')
const DIR = path.join(process.env.AUDIO_NEURAL_CACHE || path.join(os.homedir(), '.cache', 'audiojs', 'neural'), 'vocos')
const FP16 = path.join(DIR, 'vocos.fp16.onnx'), FP32 = path.join(DIR, 'vocos.onnx')
const model = existsSync(FP16) ? test : test.skip

// mean |difference| of two sounds' log-mel spectra, dB, over the cells within 60 dB of the second's peak
function distance(a, b) {
	let n = Math.min(a.length, b.length), A = mel(a.subarray(0, n)), B = mel(b.subarray(0, n)), peak = Math.max(...B.data), s = 0, k = 0
	for (let i = 0; i < B.data.length; i++) if (B.data[i] > peak - 60 * Math.LN10 / 20) s += Math.abs(A.data[i] - B.data[i]), k++
	return s / k * 20 / Math.LN10
}

test('mel: torchaudio\'s features, frame for frame', () => {
	let m = mel(x), worst = 0
	is(m.frames, 1 + Math.floor(x.length / HOP), '1 + ⌊L / hop⌋ frames, centered')
	is(m.data.length, MELS * m.frames)
	// torchaudio builds its filterbank in float32, 7e-6 from the exact one, which shows only where a band is quiet beside a loud one
	for (let i = 0; i < MEL.length; i++) worst = Math.max(worst, Math.abs(m.data[i] - MEL[i]))
	ok(worst < 0.01, `log mel within 0.01 of torchaudio's: ${worst}`)
	is(mel(new Float32Array(1000)).data[0], Math.fround(Math.log(1e-7)), 'silence: the log floor')
})

test('istft: the inverse of a centered Hann STFT, torch.istft\'s length', () => {
	let T = 1 + Math.floor(x.length / HOP), bins = N / 2 + 1, re = new Float32Array(bins * T), im = new Float32Array(bins * T)
	let w = Float64Array.from({ length: N }, (_, i) => 0.5 - 0.5 * Math.cos(2 * Math.PI * i / N)), frame = new Float64Array(N)
	let at = i => { let L = x.length, p = 2 * (L - 1); i = ((i % p) + p) % p; return x[i < L ? i : p - i] }
	for (let t = 0; t < T; t++) {
		for (let i = 0; i < N; i++) frame[i] = at(t * HOP - N / 2 + i) * w[i]
		let [r, j] = fft(frame)
		for (let k = 0; k < bins; k++) re[k * T + t] = r[k], im[k * T + t] = j[k]
	}
	let y = istft(re, im, T), e = 0, s = 0
	is(y.length, HOP * (T - 1), 'hop × (frames − 1), as torch.istft without a length')
	for (let i = 0; i < y.length; i++) e += (y[i] - x[i]) ** 2, s += x[i] ** 2
	ok(10 * Math.log10(s / e) > 100, `x back: ${(10 * Math.log10(s / e)).toFixed(1)} dB SNR`)
	is(istft(re, im, T, x.length).length, x.length, 'a length asked for')
})

model('vocode: PyTorch\'s Vocos, as near as PyTorch is to itself', { timeout: 120000 }, async () => {
	// sample by sample the phase Vocos predicts follows its features' rounding: PyTorch's own float32 features against
	// float64 ones decode 35 dB apart, 0.047 dB apart in log-mel. So the measure is log-mel distance
	let y = await vocode(x, { weights: FP16 }), d = distance(y, DECODED), changed = distance(x, DECODED)
	is(y.length, x.length, 'as long as the input')
	ok(d < 0.1, `fp16 graph ${d.toFixed(3)} dB from PyTorch's decode`)
	ok(changed > 1, `while Vocos changes the sound by ${changed.toFixed(2)} dB`)
	if (existsSync(FP32)) { let d32 = distance(await vocode(x, { weights: FP32 }), DECODED); ok(d32 < 0.1, `fp32 graph ${d32.toFixed(3)} dB`) }
})

model('frames: runs of a few frames are one run of all', { timeout: 120000 }, async () => {
	let m = mel(x), whole = await frames(m, { weights: FP16, chunk: 1e9 }), parts = await frames(m, { weights: FP16, chunk: 40 }), worst = 0, peak = 0
	for (let i = 0; i < whole.re.length; i++) worst = Math.max(worst, Math.abs(whole.re[i] - parts.re[i]), Math.abs(whole.im[i] - parts.im[i])), peak = Math.max(peak, Math.abs(whole.re[i]))
	ok(worst / peak < 1e-4, `max |diff| / peak ${(worst / peak).toExponential(1)}`)
})

model('vocode: any rate in, that rate out, as long; nothing in, nothing out', { timeout: 120000 }, async () => {
	// the sound at 44.1 kHz: made again as at 24 kHz. (Not a tone: Vocos is a speech vocoder; a sine of exact period, its
	// spectrum's floor far under any recording's, comes back silent from PyTorch's Vocos too, −112 dB at 200 Hz)
	let x44 = resample(x, { from: SAMPLE_RATE, to: 44100 }), y = await vocode(x44, { sampleRate: 44100, weights: FP16 })
	is(y.length, x44.length, 'as long')
	// 0.45 dB measured: the two resamplings' roll-off past 0.8 of 12 kHz, in the top bands
	let d = distance(resample(y, { from: 44100, to: SAMPLE_RATE }), DECODED)
	ok(d < 1, `${d.toFixed(3)} dB from PyTorch's decode at 24 kHz`)
	is((await vocode(new Float32Array(0), { weights: FP16 })).length, 0)
	is((await vocode(new Float32Array(100), { weights: FP16 })).length, 100, 'shorter than a frame')
})

// The hosted file (Hugging Face, the revision pinned). Network: skipped offline
let online
const offline = async () => !(online ??= await fetch('https://huggingface.co/', { method: 'HEAD', signal: AbortSignal.timeout(8000) }).then(r => r.ok, () => false)) && (console.log('  offline: skipped'), true)
test('vocode: the hosted file is pinned to its revision and SHA-256', async () => {
	if (await offline()) return
	let res = await fetch(MODEL, { method: 'HEAD', redirect: 'manual' })
	is(res.headers.get('x-linked-etag')?.replace(/"/g, ''), SHA256, String(MODEL))
})

// No weights, nothing cached: the hosted copy fetched, checked, kept in the cache, and run
test('vocode: no weights and an empty cache, the hosted copy is fetched into the cache and runs', async () => {
	if (await offline()) return
	let dir = mkdtempSync(path.join(os.tmpdir(), 'vocode-')), was = process.env.AUDIO_NEURAL_CACHE
	process.env.AUDIO_NEURAL_CACHE = dir
	try {
		is((await vocode(x.subarray(0, 4800))).length, 4800, 'run on the fetched copy')
		let { createHash } = await import('node:crypto')
		is(createHash('sha256').update(readFileSync(path.join(dir, 'vocos', 'vocos.fp16.onnx'))).digest('hex'), SHA256, 'kept whole in the cache')
	}
	finally { was == null ? delete process.env.AUDIO_NEURAL_CACHE : process.env.AUDIO_NEURAL_CACHE = was; rmSync(dir, { recursive: true }) }
}, { timeout: 300_000 })

is(SAMPLE_RATE, 24000)
