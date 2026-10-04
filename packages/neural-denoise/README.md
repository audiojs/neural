# @audio/neural-denoise

> Neural speech enhancement: RNNoise ported to JS, bit-exact to its C, weights inside; DeepFilterNet3 through `@audio/neural-runtime`, matching the Python original.

The neural tier beside [`@audio/denoise`](https://github.com/audiojs/denoise)'s statistical denoisers (`omlsa`, `wiener`, `specsub`). A statistical denoiser tracks a noise floor it assumes to be steady; these models learned what speech is, so they also remove noise that moves: keyboards, traffic, a fan changing speed, a busy room.

```js
import denoise from '@audio/neural-denoise'

let clean = await denoise(pcm, { sampleRate: 44100 })                           // RNNoise: nothing to download
let better = await denoise(pcm, { sampleRate: 44100, model: 'deepfilternet3' })  // 8 MB model, fetched once
```

`audio`: a mono `Float32Array` (with `sampleRate`), one array per channel (denoised independently), or a `decode()` result `{ channelData, sampleRate }`. The result has the same shape and length and lines up with the input: the models' delay is removed. Both models run at 48 kHz; other rates are resampled in and out by [`@audio/resample-sinc`](https://github.com/audiojs/resample).

| Option | Default | |
|---|---|---|
| `sampleRate` | | required unless `audio` carries it |
| `model` | `'rnnoise'` | `'rnnoise'`, `'deepfilternet3'`, or a handle from `load()` |
| `limit` | `20` RNNoise, `18` DeepFilterNet3 | attenuation limit, dB: the input is mixed back in at 10^(−limit/20), so noise drops by at most `limit`; `0` for none. Unlimited, RNNoise removes the voice on 12 of the 824 VoiceBank+DEMAND files (STOI down by more than 0.2), on none at 20 dB, where PESQ rises from 2.11 to 2.46. DeepFilterNet3's 18 dB is the most before the voice itself suffers: on VoiceBank+DEMAND its DNSMOS SIG holds from 12 to 18 dB and falls past it, while BAK and PESQ keep rising; unlimited, it takes the pauses of home narrations to digital silence, which ACX Check flags ([Accuracy](#accuracy)) |
| `weights` | | RNNoise: bytes of an upstream weight blob; DeepFilterNet3: URL or bytes of an upstream ONNX export |
| `device` | `'auto'` | DeepFilterNet3: `@audio/neural-runtime` backend, `'node'`, `'wasm'` or `'webgpu'` |
| `sessionOptions` | | DeepFilterNet3: ONNX Runtime session options, e.g. `{ intraOpNumThreads: 4 }` |
| `chunk` / `warmup` / `fade` | `1000` / `300` / `50` | DeepFilterNet3: 10 ms frames per model run, context run before each chunk, crossfade between chunks |

For many files, `load()` once, pass the handle as `model`, `free()` it after:

```js
import denoise, { load } from '@audio/neural-denoise'

let model = await load('deepfilternet3')
for (let pcm of takes) out.push(await denoise(pcm, { sampleRate, model }))
model.free()
```

## Models

| `model` | Network | Weights | Runs on | Delay | Streaming |
|---|---|---|---|---|---|
| `'rnnoise'` | RNNoise (Valin 2018; upstream's current model, January 2025): 2 convolutions, 3 GRUs, 2.88M weights (2.80M int8); 32 band gains per 10 ms | in the package: 3.5 MB (2.8 MB gzipped), BSD-3-Clause | JS; int8 products in a 419-byte WebAssembly SIMD kernel where available | 20 ms (30 ms live) | frame API, AudioWorklet |
| `'deepfilternet3'` | DeepFilterNet3 (Schröter et al. 2023): ERB gains, then a deep filter on the lowest 96 bins; 2.13M parameters | fetched from upstream on first use (8 MB) and cached; weight terms unconfirmed, see Licenses | ONNX Runtime: Node, wasm, WebGPU | 30 ms (40 ms live) | offline, chunked |

Delay: algorithmic; live adds the 10 ms a frame takes to arrive.

**The RNNoise weights ship in the package.** `rnnoise.bin` is upstream's own loadable format (what `rnnoise_model_from_file` reads), byte for byte what upstream's `dump_weights_blob` writes from the pinned model tarball. Its terms are the BSD-3-Clause of the source it comes from, and RNNoise is worth having because it runs anywhere at once: in a worklet, offline, with no network and no runtime. A custom format would save about 10% after compression (the int32 block indices compress to almost nothing) and cost a converter and interop: any blob from upstream's tools, the little model or one you train, loads through `weights`.

## Live: AudioWorklet

```js
import { weights } from '@audio/neural-denoise'

let ctx = new AudioContext({ sampleRate: 48000 })
await ctx.audioWorklet.addModule(new URL('@audio/neural-denoise/worklet', import.meta.url))
let node = new AudioWorkletNode(ctx, 'neural-denoise', { processorOptions: { weights: await weights(), limit: 30 } })
mic.connect(node).connect(ctx.destination)
```

The processor collects 128-sample render quanta into 10 ms frames. Its output queue starts with 480 − gcd(128, 480) = 448 samples of silence, the least that never runs dry, so the node's delay is constant: 1408 samples, 29.3 ms. It allocates nothing per frame. The context must run at 48 kHz; the browser resamples the device. `limit` works as in `denoise()`: 20 dB unless given, `0` for none.

Frame by frame, as `rnnoise_process_frame` works:

```js
import { rnnoise, weights } from '@audio/neural-denoise'

let st = rnnoise(await weights())
let voice = st.process(input480, output480)  // int16 scale in and out; voice probability 0–1; output 960 samples late
```

## In `audio`: the contract atom

`@audio/neural-denoise/audio` is the package's [contract](https://github.com/audiojs/compile/blob/main/CONTRACT.md) manifest: `rnnoise`, a streaming atom at any sample rate, with one parameter, `limit` (dB, default 20, 0 for none). [`audio`](https://github.com/audiojs/audio) hosts it as its `rnnoise` op, and runs DeepFilterNet3 as its `deepfilter` op (the whole input before rendering, `limit` 18 by default):

```js
import audio from 'audio'

await audio('take.wav').rnnoise().save('clean.wav')     // streams
await audio('take.wav').deepfilter().save('clean.wav')  // DeepFilterNet3, room tone kept
```

Blocks of any size collect into frames; the output queue starts with 479 samples of silence (FRAME − 1, the least that never runs dry for any block size), so the delay is constant: 1439 samples at 48 kHz (30 ms). Other rates go to 48 kHz and back through a streaming form of `@audio/resample-sinc`'s kernel, in its arithmetic, and the delay grows by its lookahead: 1354 samples at 44.1 kHz (30.7 ms), 271 at 8 kHz. The output is `denoise()`'s, delayed: at 48 kHz sample for sample; at other rates, with resample-sinc 1.2.0, sample for sample but in the last 20–30 ms, where the offline resampler sees the end and a stream the host's trailing silence. The default `limit`, 20 dB, is `denoise()`'s for RNNoise: unlimited, the model removes the voice on some files (Accuracy, below).

## RNNoise

A translation of [xiph/rnnoise](https://gitlab.xiph.org/xiph/rnnoise) at 70f1d25, the network and all its DSP, following the portable C path (`vec.h` without SIMD) op for op in float32:

1. **Frame**: 480 samples at 48 kHz through a DC-blocking biquad; a 960-point FFT of the last two frames under a Vorbis window.
2. **Features** (65): the energies of 32 ERB-spaced bands up to 20 kHz as a log spectrum, its DCT; each band's correlation with a pitch-delayed copy of the signal, its DCT; the pitch period. The period comes from a 2:1 downsampled, LPC-whitened signal: a coarse search at 4:1, a fine one around the two best lags, octave errors removed.
3. **Network**: two 1-D convolutions (65·3 → 128 → 384, tanh), three GRUs of 384, dense layers to 32 band gains and a voice probability. The GRU and second convolution weights are int8 with per-row scales, and their input activations are quantized to int8, as upstream does.
4. **Pitch filter**: on the previous frame (one frame of lookahead), the pitch-delayed spectrum is added back per band where the gains would cut voiced harmonics, then the band energies are restored.
5. **Gains**: they fall at most by a factor 0.6 per frame (an RT60 of 135 ms), are interpolated across each band and applied; inverse FFT, overlap-add. Bins above 20 kHz are zeroed.

The float code keeps C's float semantics: each float op is `Math.fround(a op b)`, exact for +, −, ×, ÷ and √ of float32 operands (Figueroa 1995), and values stay double where the C promotes to double. The int8 products are integers under 2²⁴, exact in any order, so they run in a WebAssembly SIMD kernel ([gemv.wat](./gemv.wat): widening to i16, `i32x4.dot_i16x8_s`) where the runtime has SIMD, 8.5 times faster than the JS loop that runs elsewhere; both give the same bits.

## DeepFilterNet3

Upstream's three ONNX graphs (encoder, ERB-gain decoder, deep-filter decoder) run unchanged; the rest is ported from DeepFilterNet at d375b2d, `libDF` and `df.enhance.enhance()`:

1. **STFT**: 960-point Vorbis window, 480 hop, the input padded by 960 zeros; the same FFT as RNNoise.
2. **Features**: power in 32 ERB bands (libDF's `erb_fb`: 2 to 67 bins wide) in dB, mean-normalized with α = 0.99 (τ = 1 s); the lowest 96 bins, unit-normalized with the same α. Both run 2 frames ahead of the output (`conv_lookahead`). `denoise()` computes them as if the input's speech (the louder half of its 50 ms frames) were at −20 dBFS, the median of the VoiceBank+DEMAND test set (−20.2): they are not level-free, the ERB means start at −60 to −90 dB and the complex features scale with the root of the level, and the model was trained at speech gains of −6, 0 and +6 dB (libDF `dataset.rs`). Heard as it is, a take 24 dB quieter lost its pauses' noise by 11 to 14 dB instead of 22 to 24, and the noise after 2.7 s of digital silence by 10 to 24 dB; at −20 dBFS it does as the louder take, at no cost on VoiceBank+DEMAND (PESQ 3.077 against 3.076, 206 files). The spectrum the gains and filters apply to stays as it was.
3. **Network**: 32 gains (sigmoid) and 96 × 5 complex filter taps per frame.
4. **Enhance**: every bin takes its band's gain; below bin 96 (4.8 kHz) the output is instead Σₖ spec[t − 2 + k] · coef[t, k], k = 0–4, over the unmasked spectrum; `limit` mixes the noisy spectrum back in. Inverse FFT, overlap-add, the 480-sample STFT delay removed.
5. **Voice guard** (`denoise()`): the model takes held voicing for noise. A clean held note from VocalSet (Wilkins 2018) comes out 39 dB down, unlimited, its own local-SNR estimate at its floor (−15 dB), while speech keeps its level; sung phrases lose 24 to 34 dB. So where the input holds sustained voicing, runs of 0.3 s or more periodic at RNNoise's pitch (normalized correlation 0.45 or more, Praat's voicing threshold, the pitch searched every 20 ms) that stand 6 dB or more over the quietest 100 ms of the 20 s around (minimum statistics, Martin 2001: a buzz is that background, a note stands over it), each band holding a harmonic keeps at least c² of its input, c its correlation with the frame one and two periods earlier: the periodic share of its power, as RNNoise's pitch filter restores what its band gains would remove. Between the harmonics and in the pauses the model's output stands.

The parameters come from the export's `config.ini`, so upstream's low-latency export (`DeepFilterNet3_ll_onnx.tar.gz`: no lookahead, 36 MB) loads through `weights` too.

**Long inputs** run in chunks of `chunk` frames (10 s). Upstream feeds the whole file at once, about 8 MB of activations per second of audio (peak RSS 768 MB for 30 s, 1 GB for 60 s in onnxruntime-node); in 10 s chunks a 180 s file peaks at 563 MB, and any length stays there. The ONNX graphs are exactly causal (outputs on a prefix equal the whole run's) but take no GRU state, so each chunk after the first starts `warmup` frames (3 s) early from zero state, and its first `fade` frames (0.5 s) crossfade from the previous chunk's run, which continued that far. The GRUs remember how they started: 6 s after such a start the gains still differ from the whole-file run's by up to 0.1–0.5. Chunked output is another valid run, not the whole-file one; on VoiceBank+DEMAND in 1 s chunks (1 s warm-up, 0.2 s crossfade, 2 to 3 chunks per utterance) the scores do not move: PESQ 3.164 against 3.162 whole-file, STOI 0.9450 against 0.9452, SI-SDR 18.94 against 18.96 dB, though single files shift by up to 0.31 PESQ either way. The STFT and features stay exact across chunks. Exact chunking needs a stateful re-export (GRU states and convolution buffers as graph inputs and outputs), which is also what live DeepFilterNet3 needs.

## Verification

| Check | Result |
|---|---|
| RNNoise vs upstream C, portable build (`cc -O2 -ffp-contract=off -DDISABLE_NEON`), default and little models | **bit-exact**, output and voice probability: lena (1127 frames), lena + noise (1127), digital silence and near-silence (300), a VoiceBank+DEMAND utterance (174) |
| RNNoise vs upstream C as `configure && make` builds it on arm64 (NEON, fused multiply-adds) | 38.8–62.4 dB SNR, voice probability within 0.087: upstream's builds differ from each other by this much (NEON's `vrecpeq_f32` reciprocal estimate inside tanh and sigmoid) |
| `rnnoise.bin` | byte for byte what upstream's `dump_weights_blob` writes from the model tarball (sha256 `1ad07b42…`) |
| FFT, window and DCT tables | equal to upstream's `rnnoise_tables.c`: 960 twiddles, 480 window values, 1024 DCT entries, the bit reversal; the FFT matches a direct DFT to 3.2e-9 |
| DeepFilterNet3, `enhance()` with upstream's settings (no level change, no voice guard, whole file), vs Python `df.enhance.enhance()` (deepfilternet 0.5.6, PyTorch checkpoint), same input, `limit: 0` | 133.5 dB SNR on a 1.7 s utterance, 128.2 dB on 11.3 s (max \|d\| 5.7e-7), 133.8 dB on an utterance led by 2 s of noise, 134.5 dB on a 60 s narration, its first second included; ERB widths and α identical |
| DeepFilterNet3 voice guard, no model (`test.js`) | a held /a/ (190 Hz, 5 Hz vibrato) 20 dB over white noise through a model that removes everything: kept within 2.8 dB, the noise around it removed; a 120 Hz buzz under the same noise: no frame guarded |
| DeepFilterNet3 plumbing, no model (`test.js`) | identity gains and taps give the input back to 1.8e-7, whole and chunked (chunks identical to the whole run); a tap one frame ahead advances the band below 4.8 kHz by exactly 480 samples (to 8.9e-8) |
| Worklet (`test.js`, AudioWorkletGlobalScope simulated) | the frame API's output, delayed by 448 samples, at `limit: 0`; at 12 and at the default 20, that output with the input mixed back, to 1e-7 |
| Worklet in Chromium 153 (headless, Playwright, `scripts/worklet.mjs`) | runs, no processor error, output non-silent; live underruns in Speed |
| Contract atom (`test.js`) against `denoise()`, delayed by its declared latency | 48 kHz: equal sample for sample in blocks of 1, 128, 1024 and 1 to 3000 samples, limits 20 and 0; 44.1 kHz (resample-sinc 1.2.0): equal but in the last 50 ms; with resample-sinc 1.1.2, whose arithmetic differs, 59.5 dB SNR |

To rerun: `node scripts/rnnoise-reference.mjs` builds upstream and rewrites `fixtures/rnnoise.json` (and caches the little model); `python scripts/deepfilter-reference.py` (DeepFilterNet 0.5.6 in a Python 3.11 venv) writes the reference `test.js` compares with.

## Accuracy

The VoiceBank+DEMAND test set (Valentini-Botinhao 2017, CC BY 4.0): 824 utterances by 2 speakers at 48 kHz, 5 noises at 2.5 to 17.5 dB SNR. Scores at 16 kHz (`scipy.signal.resample_poly`, 1:3): wideband PESQ (ITU-T P.862.2, python-pesq 0.0.4), STOI (Taal 2011, pystoi 0.4.1), SI-SDR (Le Roux 2019), and the reference-free DNSMOS P.835 (Reddy 2022, `sig_bak_ovr.onnx` with the polynomial fits of `dnsmos_local.py`: SIG speech, BAK background, OVRL overall, 1 to 5). The classical rows are [`@audio/denoise`](https://github.com/audiojs/denoise)'s ops through their `audio` manifests after its September 2026 audit, measured by its `scripts/speech.mjs` and `scripts/speech.py` with this scoring; the 0.3.11 ops that `audio` 2.6 runs scored `omlsa` 1.88, `wiener` 2.19, `specsub` 2.11 PESQ.

| System | PESQ | STOI | SI-SDR dB | SIG | BAK | OVRL |
|---|---|---|---|---|---|---|
| noisy input | 1.97 | 0.921 | 8.45 | 3.32 | 3.11 | 2.68 |
| `omlsa()`, defaults | 2.40 | 0.915 | 14.55 | 3.37 | 3.48 | 2.85 |
| `wiener()`, defaults | 2.33 | 0.910 | 13.95 | 3.37 | 3.38 | 2.80 |
| `specsub()`, defaults | 2.24 | 0.920 | 12.52 | 3.35 | 3.33 | 2.77 |
| "Enhance speech", its denoising: `highpass(80).dehum().omlsa()` | 2.43 | 0.914 | 6.03 | 3.40 | 3.39 | 2.83 |
| "Enhance speech", the whole recipe, with 0.3.11's `omlsa` | 1.63 | 0.830 | 0.84 | 3.06 | 2.37 | 2.18 |
| RNNoise, `limit: 0` (this package = upstream C) | 2.11 | 0.890 | 12.28 | 3.28 | 3.85 | 2.94 |
| RNNoise, little model, `limit: 0` | 2.12 | 0.893 | 12.39 | 3.28 | 3.85 | 2.93 |
| RNNoise, `limit: 20`, the default | 2.46 | 0.918 | 12.86 | 3.40 | 3.71 | 2.97 |
| **DeepFilterNet3**, upstream's settings, `limit: 0` (`enhance()`) | **3.16** | **0.945** | **18.96** | **3.44** | **4.08** | **3.18** |
| DeepFilterNet3, Python (deepfilternet 0.5.6) | 3.16 | 0.945 | 18.96 | 3.44 | 4.08 | 3.18 |
| DeepFilterNet3 in 1 s chunks, `limit: 0` | 3.16 | 0.945 | 18.94 | 3.44 | 4.08 | 3.18 |
| DeepFilterNet3, `denoise()`, `limit: 0`: heard at −20 dBFS, voice guard | 3.15 | 0.945 | 18.95 | 3.44 | 4.08 | 3.18 |
| DeepFilterNet3, `denoise()`, `limit: 18`, the default | 2.90 | 0.942 | 17.91 | 3.48 | 3.84 | 3.09 |
| DeepFilterNet3, `limit: 12` (the default before 0.2) | 2.67 | 0.939 | 16.18 | 3.49 | 3.69 | 3.03 |

Over the noisy input, paired, with 95% intervals: DeepFilterNet3 adds 1.20 ± 0.03 PESQ, 0.024 ± 0.002 STOI and 10.5 ± 0.3 dB SI-SDR, and beats every classical row's PESQ on 804 of 824 files; RNNoise adds 0.14 ± 0.04 PESQ and 3.8 ± 0.3 dB SI-SDR but loses 0.031 ± 0.004 STOI; `omlsa`, the best classical row, adds 0.43 ± 0.03 PESQ and 6.1 ± 0.2 dB SI-SDR and loses 0.006 STOI. A limit leaves noise in on this set, and spares the voice: paired, on a quarter of the files (206), DNSMOS SIG moves −0.002 ± 0.015 from 12 to 18 dB, then −0.008 ± 0.003 to 20, −0.019 ± 0.006 more to 24 and −0.049 ± 0.017 more unlimited, while BAK rises 0.18, 0.04, 0.07 and 0.16 and PESQ 0.25, 0.06, 0.08 and 0.10. Past 18 dB the speech itself loses, which DNSMOS reads as SIG and a listener as a filtered, phone-call voice; from 8 to 16 kHz the unlimited output carries 1.5 dB less than the clean speech, at 18 dB 0.9 (all 824 files). The default 18 dB still adds 0.93 PESQ over the input and 0.50 over `omlsa`. Pass `limit: 0` where the most noise removal matters more than room tone. The level change and the voice guard cost 0.01 PESQ unlimited and 0.004 at 18 dB. The same scoring gives upstream's published anchors: noisy 1.97 and 0.921 (as the DeepFilterNet2 paper's table), DeepFilterNet3 3.16 and 0.945 against the DeepFilterNet3 paper's 3.17 and 0.944. That DeepFilterNet2 table measured RNNoise's 2022 model at 2.33 and 0.922; the current model scores lower here.

- **RNNoise's current model sometimes removes the voice.** STOI drops by more than 0.1 on 68 of 824 files and by more than 0.2 on 12; on p232_354 it goes from 0.870 to 0.154, the output 24 dB under the clean level while RNNoise's own voice probability averages 0.84. Upstream's C does the same, the portable and the NEON build alike (0.154, 0.153). DeepFilterNet3 never loses more than 0.066 STOI, `wiener` 0.065, `omlsa` 0.059. A 20 dB `limit` (the input mixed back at 0.1) caps it: losses over 0.1 fall to 6 files and over 0.2 to none (p232_354 keeps 0.785), and PESQ rises to 2.46, +0.50 ± 0.03 over the noisy input and +0.07 ± 0.02 over `omlsa`, at a BAK of 3.71 instead of 3.85. `denoise()`, the worklet and the atom apply that limit unless given another.
- **The recipe's delivery stages are not denoisers.** Its compressor, EQ, de-esser and loudness target change the signal against the clean reference on purpose: the full recipe scores under its own denoising part. SI-SDR also falls with the 80 Hz high-pass: on its own it takes clean speech to 6.2 dB SI-SDR at PESQ 4.45 (21 test utterances). The table says what each stage does to speech in noise, not how finished a podcast sounds.

**Real rooms.** Ten amateur narrations from Spoken Wikipedia (volunteers at home, 2007 to 2026, the first 60 s of each; Ogg Vorbis at 44.1 and 48 kHz, decoded and resampled to 48 kHz by ffmpeg), scored without a reference: DNSMOS P.835 as above; the noise floor as the RMS of the quietest 500 ms (ACX Check's measure; one file with pauses cut to digital silence left out); speech level change over the loudest half of 50 ms frames.

| System | SIG | BAK | OVRL | noise floor, dBFS (median) | speech level |
|---|---|---|---|---|---|
| raw | 3.45 | 4.04 | 3.17 | −83 to −54 (−62) | |
| `omlsa()` | 3.46 | 4.09 | 3.19 | −98 to −69 (−75) | 0.0 dB |
| "Enhance speech" denoising | 3.45 | 4.07 | 3.17 | −101 to −71 (−85) | −0.2 dB |
| RNNoise, `limit: 0` | 3.42 | 4.14 | 3.19 | −127 to −106 (−112) | −0.5 dB |
| DeepFilterNet3, upstream's settings, `limit: 0` | 3.47 | 4.12 | 3.21 | −152 to −61 (−89) | −0.5 dB |
| DeepFilterNet3, `denoise()`, `limit: 0` | 3.47 | 4.11 | 3.21 | −132 to −60 (−84) | −0.4 dB |
| DeepFilterNet3, `denoise()`, `limit: 18`, the default | 3.47 | 4.11 | 3.21 | −93 to −59 (−73) | −0.4 dB |
| DeepFilterNet3, `limit: 12` | 3.47 | 4.10 | 3.21 | −91 to −57 (−67) | −0.4 dB |

These rooms are mostly quiet (raw OVRL 3.17), so the gains are small. OM-LSA keeps SIG and raises BAK a little less than the models (0.3.11's lowered SIG by 0.20); both models raise BAK and keep SIG. Both can also take pauses to dead silence (RNNoise's median floor is −112 dBFS, DeepFilterNet3's lowest −152), which ACX Check flags; a `limit` of 18 dB keeps DeepFilterNet3's floors between −93 and −59 dBFS at the same OVRL, room tone left in. On lena (a film scene with music under the voice) RNNoise lowers OVRL from 2.76 to 2.47, DeepFilterNet3 raises it to 2.83, 2.90 with the limit.

**Singing.** VocalSet (Wilkins et al., ISMIR 2018, CC BY 4.0): 40 long tones (straight, forte, pianissimo, messa di voce; 20 singers) and 20 sung phrases (straight and vibrato), alone and in living-room noise (DEMAND, from the VoiceBank+DEMAND test set) at 15 dB SNR; level change of the voice over the frames within 20 dB of the loudest. At the default 18 dB, against 0.1's processing at the same limit (heard as it is, no voice guard): held notes −1.9 dB clean and −3.4 dB in noise, 14% of frames more than 6 dB down, against −14.2 and −16.2 dB, 96% (at 10 dB SNR −4.8 against −15.6); sung phrases −0.9 and −2.0 dB against −10.2 and −14.9; the same singers' spoken phrases −0.3 and −0.8 dB against −1.7 and −3.7 (they are quiet recordings: the −20 dBFS hearing helps them). Upstream's unlimited output takes the held notes 38.7 and 37.4 dB down. The noise in the pauses drops as before (−14.1 dB against −14.3), and a 100 or 120 Hz buzz 20 dB under the voice drops in them as without the guard (−17.2 dB), in quiet and in cafe noise: it is the background, not over it.

To rerun: `node scripts/accuracy.mjs vbdemand SYSTEMS` and `node scripts/accuracy.mjs rooms` write the outputs (the data sources and checksums are in its header), `python scripts/accuracy.py vbdemand SYSTEMS` and `python scripts/accuracy.py rooms` score them, and `scripts/deepfilter-reference.py --vbdemand` writes the Python row; the classical rows come from `@audio/denoise`'s `node scripts/speech.mjs vbdemand|rooms SYSTEMS` and `python scripts/speech.py score vbdemand|rooms SYSTEMS`. Per-file scores stay in `~/.cache/audiojs/data/` (`vbdemand/scores/`, `spoken/scores.json`); the scripts regenerate outputs, skipping any already there.

## Speed and memory

Apple M4 Max running other jobs (load average 28 to 50 on 14 cores), so wall times ran up to 10 times CPU times; CPU times are given.

**RNNoise**, per 10 ms frame at 48 kHz:

| Where | Per frame | |
|---|---|---|
| Chromium 153, page thread, SIMD kernel | 0.32 ms mean, 0.5 ms p99, 0.9 ms max | 1000 frames |
| Chromium 153, page thread, JS only | 1.46 ms mean, 1.6 ms p99, 2.9 ms max | |
| Chromium 153, the worklet in an OfflineAudioContext | 0.36 ms (30 s rendered in 1.09 s) | the whole node: FIFO, int16 scaling |
| Chromium 153, the worklet live: 48 kHz, 256-sample buffers, `AudioContext.playbackStats` | 5 runs of 19 s: no underrun in 3; 8 and 14 (43 and 75 ms) in 2. Controls without the worklet, at the same time: 0 and 2 (11 ms) | output checked non-silent; load average 21 to 50 on 14 cores |
| Node 25, SIMD kernel / JS only | 0.53 ms / 1.58 ms CPU | real-time factors 0.053 / 0.158 |
| Node 25, the contract atom, SIMD kernel, 1024-sample blocks | 0.92 ms CPU at 48 kHz; 1.32 ms a frame's worth at 44.1 kHz | real-time factors 0.092 / 0.132: the 44.1 kHz host resamples both ways; load average 65 on 14 cores |
| upstream C, portable / NEON build, same machine and load | 0.44 ms / 0.12 ms CPU | for scale |

A 128-sample render quantum lasts 2.67 ms and a frame is due every 3.75 quanta: the slowest SIMD frame measured (0.94 ms) fits a quantum 2.8 times over, the mean 8 times; the JS path's slowest (up to 4.0 ms) does not, and relies on device buffers of 256 samples or more, the browsers' usual size. The live underruns came with the machine oversubscribed, the control underran too; an idle machine is untested. Memory: 3.5 MB of weights (plus a 2.8 MB copy in the kernel's memory), about 70 KB of state per channel; the Node evaluation processes peaked at 83 MB RSS.

**DeepFilterNet3**, onnxruntime-node 1.30, 4 intra-op threads: real-time factor 0.047 in CPU on a 180 s file (8.4 s), 0.087 over the 824 short test utterances (2072 s of audio in 180 s; per-run overhead); Python DeepFilterNet (PyTorch 2.1, 4 threads) 0.067 on the same utterances. Peak RSS 436 MB over the test set, 563 MB for 180 s in 10 s chunks. The voice guard's pitch analysis adds 0.16 s of wall time per minute of audio to the model's 0.6 (one channel at 48 kHz, 4 threads). The browser path (onnxruntime-web, wasm and WebGPU) is untested here.

## Licenses

Audited 2026-09-27 against what each project states; a summary, not legal advice.

| Model | Code | Weights | Weight files and their terms | Here |
|---|---|---|---|---|
| **RNNoise**, model 0a8755f8 (January 2025) | BSD-3-Clause (`COPYING`: Valin, Amazon, Mozilla, Xiph.Org, Borgerding) | BSD-3-Clause | `rnnoise_data-0a8755f8….tar.gz` (media.xiph.org, sha256 pinned in upstream's `model_version`): `src/rnnoise_data.c` (default), `src/rnnoise_data_little.c`, two `.pth` checkpoints. No header of their own; upstream's `autogen.sh` downloads them into `src/` and compiles them into the BSD library, and the README calls them "the models distributed with RNNoise". Trained on the corpora in `datasets.txt`: OpenSLR 30–86 (CC BY-SA 4.0) and Hi-Fi TTS (OpenSLR 109, CC BY 4.0) for speech, Xiph's noise collections (donated noise "freely available", terms in that archive, not checked) | bundled: `rnnoise.bin`, the default model through upstream's `dump_weights_blob`; the little model (1.55 MB) loads through `weights` |
| **DeepFilterNet3**, d375b2d | MIT OR Apache-2.0 ("All code in this repository") | **unconfirmed** | `models/DeepFilterNet3_onnx.tar.gz`, `DeepFilterNet3.zip`, `DeepFilterNet3_ll_onnx.tar.gz`, in the same repository, no terms of their own. Four open issues ask whether "all code" covers them ([#697](https://github.com/Rikorose/DeepFilterNet/issues/697), [#700](https://github.com/Rikorose/DeepFilterNet/issues/700), [#709](https://github.com/Rikorose/DeepFilterNet/issues/709), [#712](https://github.com/Rikorose/DeepFilterNet/issues/712), July to September 2026), none answered; the last push was October 2024. The DeepFilterNet3 paper's abstract says "the framework as well as pretrained weights have been published under an open source license", naming none. Third parties republish them as MIT (Intel/deepfilternet-openvino on Hugging Face) or Apache-2.0 | fetched from upstream at the pinned commit, cached, never bundled or re-hosted |
| DeepFilterNet 1 and 2 | same | same as DeepFilterNet3 | `models/DeepFilterNet{,2}*.{zip,tar.gz}` | not used; libDF deprecates DeepFilterNet2 |
| GTCRN (Rong et al., ICASSP 2024) | MIT | MIT (checkpoints in the repository) | 48.2K parameters, trained on DNS3 or VCTK-DEMAND | not used: 16 kHz only, narration would lose everything above 8 kHz |
| Demucs speech denoiser (facebookresearch/denoiser) | CC BY-NC 4.0 | CC BY-NC 4.0 | | excluded: non-commercial; archived |
| DNSMOS P.835 (Microsoft) | CC BY 4.0 (microsoft/DNS-Challenge) | CC BY 4.0 | `DNSMOS/DNSMOS/sig_bak_ovr.onnx` at 82f1b17e77 | evaluation only |
| VoiceBank+DEMAND (Valentini-Botinhao 2017) | | CC BY 4.0 | 824 test utterances, DataShare checksums verified | evaluation only |
| Spoken Wikipedia narrations | | CC BY-SA 3.0/4.0, CC BY 4.0, CC0 | ten files from Wikimedia Commons | evaluation only, not redistributed |

In the browser, [shiguredo/rnnoise-wasm](https://github.com/shiguredo/rnnoise-wasm) (Apache-2.0) builds the same RNNoise commit with emscripten, and [sapphi-red/web-noise-suppressor](https://github.com/sapphi-red/web-noise-suppressor) (MIT) wraps its 2022 build next to Speex and GTCRN. This package needs no WebAssembly (it takes 419 bytes of it for speed where it can), is checked bit for bit against the C, and adds DeepFilterNet3.

The package is BSD-3-Clause (the RNNoise translation and weights); `deepfilter.js` carries DeepFilterNet's MIT notice ([NOTICE](./NOTICE)).

## Reference

J.-M. Valin, "A Hybrid DSP/Deep Learning Approach to Real-Time Full-Band Speech Enhancement", MMSP 2018, [arXiv:1709.08243](https://arxiv.org/abs/1709.08243). · H. Schröter, T. Rosenkranz, A. N. Escalante-B., A. Maier, "DeepFilterNet: Perceptually Motivated Real-Time Speech Enhancement", Interspeech 2023, [arXiv:2305.08227](https://arxiv.org/abs/2305.08227). · C. Valentini-Botinhao, "Noisy speech database for training speech enhancement algorithms and TTS models", University of Edinburgh, 2017, [doi:10.7488/ds/2117](https://doi.org/10.7488/ds/2117). · ITU-T P.862.2 (2007), wideband PESQ. · C. H. Taal, R. C. Hendriks, R. Heusdens, J. Jensen, "An Algorithm for Intelligibility Prediction of Time-Frequency Weighted Noisy Speech", IEEE TASLP 19(7), 2011. · J. Le Roux, S. Wisdom, H. Erdogan, J. R. Hershey, "SDR: Half-baked or Well Done?", ICASSP 2019. · C. K. A. Reddy, V. Gopal, R. Cutler, "DNSMOS P.835: A Non-Intrusive Perceptual Objective Speech Quality Metric to Evaluate Noise Suppressors", ICASSP 2022. · S. A. Figueroa, "When is double rounding innocuous?", SIGNUM Newsletter 30(3), 1995. · [xiph/rnnoise](https://gitlab.xiph.org/xiph/rnnoise) at 70f1d25 · [Rikorose/DeepFilterNet](https://github.com/Rikorose/DeepFilterNet) at d375b2d.

**Use when:** speech with noise a statistical denoiser leaves or smears (moving noise, a busy room); live voice in a browser (RNNoise, 30 ms); files where quality is worth an 8 MB download (DeepFilterNet3).<br>
**Not for:** music (both models keep speech and drop the rest; DeepFilterNet3's voice guard keeps held sung notes, RNNoise drops them); installs that must stay model-free ([`@audio/denoise`](https://github.com/audiojs/denoise)); reverb as such (`dereverb`); live DeepFilterNet3 (upstream's graphs take no state).

---

Part of the [@audio/neural](https://github.com/audiojs/neural) lane.

BSD-3-Clause © 2026 audiojs (port), Jean-Marc Valin, Xiph.Org Foundation, Mozilla, Amazon (RNNoise); DeepFilterNet parts MIT © 2021 Hendrik Schröter
