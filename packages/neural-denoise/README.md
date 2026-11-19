# @audio/neural-denoise

> Neural speech enhancement: RNNoise ported to JS, bit-exact to its C, weights inside; DeepFilterNet3 through `@audio/neural-runtime`, matching the Python original.

The neural tier beside [`@audio/denoise`](https://github.com/audiojs/denoise)'s statistical denoisers (`omlsa`, `wiener`, `specsub`). A statistical denoiser tracks a noise floor it assumes to be steady; these models learned what speech is, so they also remove noise that moves: keyboards, traffic, a fan changing speed, a busy room. To them everything that isn't speech is noise, music too, so a speech/music classifier stands guard: music, songs included, passes through untouched ([Music](#music)).

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
| `limit` | `16` RNNoise, `18` DeepFilterNet3 | attenuation limit, dB, the most the noise drops: the input is mixed back in at 10^(−limit/20) at every value, so `0` takes nothing off (at 48 kHz the input itself; DeepFilterNet3's `floor` still acts in the voice's pauses, `floor: 0` with it for the input) and `Infinity` is no limit, the model's whole removal ([The limit's scale](#the-limits-scale)). Each default is the most before the voice itself suffers. RNNoise turns down speech that has nothing to remove (clean word ends 37 dB down unlimited, 12 at 16 dB) and removes the voice on some noisy files; on VoiceBank+DEMAND's training speakers its DNSMOS SIG holds from 14 to 16 dB and falls past it, and at 16 no file loses more than 0.2 STOI (21 of 504 unlimited, 3 at 20, the default before 0.3). DeepFilterNet3's SIG holds from 12 to 18 dB and falls past it, while BAK and PESQ keep rising; unlimited, it takes the pauses of home narrations to digital silence, which ACX Check flags ([Accuracy](#accuracy)) |
| `floor` | `40` | DeepFilterNet3: dB under the voice. Under the voice what the model took comes back `limit` dB down, so `limit` is all that voice loses. In the voice's pauses, noise the limit would leave closer to the voice than that drops further, down to it, frame by frame (`mixback()`), where the model's output there stays more than 30 dB under the voice; **on real noise it mostly does not, and the pauses keep the limit: on 66 noisy takes their noise 0.3 s or more from the voice sits 26.5 dB under it, against 32.0 with `everywhere: true`** ([DeepFilterNet3's floor](#deepfilternet3s-floor)). `0` for none, as before 0.5 |
| `everywhere` | `false` | DeepFilterNet3: `true` takes the floor under the voice too, what the model took back at one gain: 0.7.0's mix, sample for sample. The cleanest pauses (PESQ 2.95 against 2.81 on every 8th VoiceBank+DEMAND test file), and the voice the model took goes with the noise ([DeepFilterNet3's floor](#deepfilternet3s-floor)) |
| `music` | `'pass'` | `'pass'`: speech and noise are enhanced, music passes through untouched, segment by segment, the gain ramped over 200 ms at each switch ([Music](#music)); `'enhance'`: everything, as before 0.4 |
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

### DeepFilterNet3's floor

`denoise()` mixes DeepFilterNet3's unlimited output y back with the input x (`mixback()`): y + a·(x − y) per 10 ms frame, a the share of what the model took that comes back. Two mixes, each better than the other on some measures and worse on others:

- **The default.** Where y shows the voice (its frames within 30 dB of the voice's level, gaps up to 0.15 s closed, 0.3 s either side), what the model took comes back `limit` dB down: a voice cannot lose more than the limit to the model, and a word the model removes whole stays audible, `limit` dB down (`test.js`: the voice a model took comes back 18.0 dB down at the default). Elsewhere the noise drops toward `floor` dB under the voice. On real noise the model leaves residue in the pauses, and a frame of it within 30 dB of the voice counts as voice for 0.3 s either side, so most of a pause keeps the limit: on 66 noisy takes (42 of VoiceBank+DEMAND's test set, 24 made of its clean utterances under DEMAND beds it does not use, with gaps of 0.12 to 2 s) the noise 0.3 s or more from the voice sits 26.5 dB under it, against 32.0 with `everywhere: true`, and at 0 dB SNR 20.2 against 31.8; a take with no voice at all (10 s of DEMAND's washing machine, field or park alone) comes back 18.0, 17.4 and 17.6 dB down, against 38.9, 38.1 and 40.3.
- **What the default keeps that `everywhere` takes:** on the 66 takes, voiced frames losing 10 dB or more of their voice 6.7 % against 9.0 %, DNSMOS SIG +0.09 ± 0.04 (paired, 95 %). The voice there is each voiced 20 ms frame's projection on the clean voice; counted by the frame's power (1.4 % against 6.2 %), the noise given back with the voice counts as voice.
- **What it costs against `everywhere`:** on every 8th VoiceBank+DEMAND test file (103), PESQ 2.81 against 2.95 (−0.142 ± 0.038), STOI 0.933 against 0.935; on the 66 takes SI-SDR −0.90 ± 0.29 dB, BAK −0.22 ± 0.07, OVRL −0.03 ± 0.02, and the pauses 4.7, 5.7 and 5.5 dB louder (short, long, and 0.3 s or more from the voice). All 824: [Accuracy](#accuracy).
- **`everywhere: true`** takes the floor under the voice too, what the model took back at one gain: 0.7.0's mix, sample for sample (`test.js`; and all 66 takes). The cleanest pauses and no voice; the voice the model took, at a word's start or a word it removes whole, goes down with the noise (36.9 dB in `test.js`, where the default gives back 18.0).

To rerun: `scripts/floor.mjs` (each command in its header) and `scripts/floor.py`, against `npm pack`ed 0.7.0 and 0.8.0.

### The limit's scale

`limit` is the most a model may take off, in dB. The share of the input mixed back is 10^(−limit/20) at every value, so it falls as the limit rises and no value means other than its number:

| `limit` | share of the input mixed back | the output |
|---|---|---|
| `0` | 1 | nothing taken off: the input itself, the latency late where it streams. At a rate other than 48 kHz, the input as resample-sinc carries it to 48 kHz and back, sample for sample (`test.js`): the band near its edge turned down by the kernel (−1.8 dB at 95 % of it, −4.9 dB at 98 %), and at 88.2 and 96 kHz the band over 24 kHz with it: a tone at 24.5 kHz comes back 12.6 dB down (its alias under 24 kHz counted), from 30 kHz 64 dB down and more (on noisy speech with noise up to 22.05 kHz, 35.7 dB SNR from the input at 44.1 kHz; `scripts/limit-zero.mjs`). DeepFilterNet3: so under the voice; in its pauses `floor` still takes noise nearer the voice than it, and `floor: 0` with it gives the input |
| `16`, `18` | 0.158, 0.126 | the defaults, RNNoise's and DeepFilterNet3's |
| `100` | 10^−5 | the top of the contract atom's range: no limit, to within 10^−5 of what the model took |
| `Infinity` | 0 | no limit: the model's whole removal, upstream's output (`denoise()`, `mixback()`, the worklet) |

`denoise()`, `mixback()` and the worklet refuse a limit that is no number with a TypeError (`denoise()` and the worklet take `null` as not given); the contract atom's hosts refuse one before it reaches the atom (`audio`'s: "rnnoise: limit takes dB, not NaN"), and the atom handed `NaN` directly renders `NaN`. Before 0.8 `NaN` and `''`, like `0`, meant no limit, and a numeric string its number. The scale is that of DeepFilterNet's real-time library and its LADSPA plugin: at [d375b2d](https://github.com/Rikorose/DeepFilterNet/tree/d375b2d8309e0935d165700c91da9de862a99c31), `DfTract::set_atten_lim` (libDF/src/tract.rs:387–398) reads `lim >= 100.` as no limit (`None`), `lim < 0.01` as `Some(1.)` ("No noise reduction will be performed") and any other value as 10^(−lim/20), and mixes `spec_enh·(1 − lim) + lim·spec_noisy` (tract.rs:625–629); its default is 100 (tract.rs:181), and the plugin's "Attenuation Limit (dB)" port runs from 0 to 100, 100 by default (ladspa/src/lib.rs:31–33, 596–603; "100 dB means no attenuation limit", ladspa/filter-chain-configs/deepfilter-mono-source.conf:19). This package leaves that library where it steps: under 0.01 dB the library passes the input frame whole and skips the network, undelayed (tract.rs:543–546), where here the formula holds down to 0 and the delay stays; at 100 the library mixes back none, here 10^−5, and none at `Infinity`. It leaves upstream's Python `enhance()`, whose `atten_lim_db` of 0 or `None` is no limit (DeepFilterNet/df/enhance.py:238–240: `if atten_lim_db is not None and abs(atten_lim_db) > 0:`), as `limit` was read before 0.8: there a value just above 0 takes almost nothing off and 0 everything, so a host's automation curve of the atom's `limit` into or out of 0 passed the input whole on its way. RNNoise has no such parameter upstream (include/rnnoise.h at 70f1d25: a state, a frame, the models); its limit is this package's, on the same scale.

## Models

| `model` | Network | Weights | Runs on | Delay | Streaming |
|---|---|---|---|---|---|
| `'rnnoise'` | RNNoise (Valin 2018; upstream's current model, January 2025): 2 convolutions, 3 GRUs, 2.88M weights (2.80M int8); 32 band gains per 10 ms | in the package: 3.5 MB (2.8 MB gzipped), BSD-3-Clause | JS; int8 products in a 419-byte WebAssembly SIMD kernel where available | 20 ms (30 ms live) | frame API, AudioWorklet |
| `'deepfilternet3'` | DeepFilterNet3 (Schröter et al. 2023): ERB gains, then a deep filter on the lowest 96 bins; 2.13M parameters | fetched from upstream on first use (8 MB) and cached; MIT OR Apache-2.0, see Licenses | ONNX Runtime: Node, wasm, WebGPU | 30 ms (40 ms live) | offline, chunked |

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

The processor collects 128-sample render quanta into 10 ms frames. Its output queue starts with 480 − gcd(128, 480) = 448 samples of silence, the least that never runs dry, so the node's delay is constant: 1408 samples, 29.3 ms. It allocates nothing per frame. The context must run at 48 kHz; the browser resamples the device. `limit` works as in `denoise()`: 16 dB unless given, `0` takes nothing off (the input, 1408 samples late), `Infinity` is no limit. The worklet is for a live voice and has no music guard: it denoises everything.

Frame by frame, as `rnnoise_process_frame` works:

```js
import { rnnoise, weights } from '@audio/neural-denoise'

let st = rnnoise(await weights())
let voice = st.process(input480, output480)  // int16 scale in and out; voice probability 0–1; output 960 samples late
st.run(input480, output480)                  // the same, no number returned: the probability in st.voice[0]
```

`run` is for real-time loops (the worklet and the atom call it): a float returned from a call V8 does not inline is a new object, one a frame.

## In `audio`: the contract atom

`@audio/neural-denoise/audio` is the package's [contract](https://github.com/audiojs/compile/blob/main/CONTRACT.md) manifest: `rnnoise`, a streaming atom at any sample rate, with two parameters, `limit` (dB, 0 to 100, default 16: 0 takes nothing off, 100 is no limit to within 10^−5; [The limit's scale](#the-limits-scale)) and `music` (`'pass'`, the default, or `'enhance'`). [`audio`](https://github.com/audiojs/audio) hosts it as its `rnnoise` op, and runs DeepFilterNet3 as its `deepfilter` op (the whole input before rendering, unlimited, the input mixed back after by `mixback()`: `limit` 18, `floor` 40 and `music: 'pass'` by default):

```js
import audio from 'audio'

await audio('take.wav').rnnoise().save('clean.wav')     // streams
await audio('take.wav').deepfilter().save('clean.wav')  // DeepFilterNet3, room tone kept, music passed
await audio('show.wav').deepfilter({ music: 'enhance' }) // music denoised too
```

Blocks of any size collect into frames; the output queue starts with 479 samples of silence (FRAME − 1, the least that never runs dry for any block size), so the delay is constant: 1439 samples at 48 kHz (30 ms). Other rates go to 48 kHz and back through a streaming form of `@audio/resample-sinc`'s kernel, in its arithmetic, and the delay grows by its lookahead: 1354 samples at 44.1 kHz (30.7 ms), 271 at 8 kHz. The output is `denoise()`'s, delayed: at 48 kHz sample for sample; at other rates, with resample-sinc 1.2.0, sample for sample but in the last 20–30 ms, where the offline resampler sees the end and a stream the host's trailing silence. The default `limit`, 16 dB, is `denoise()`'s for RNNoise: unlimited, the model gates clean speech and removes the voice on some noisy files (Accuracy, below). With `music: 'pass'` the atom decides as `denoise()` does for RNNoise, from what has arrived, at no added delay, and mixes the denoised signal with the input itself, delayed by the latency, at the host's rate: what passes is the input, sample for sample. Each of `ctx.maxChannels` channels has its state made with the processor (0.53 MB at 48 kHz in blocks of up to 128 samples, 1.0 MB at 8 kHz in 4096: RNNoise's 70 KB, the guard's 0.37 MB, queues sized by `maxBlockSize`, the 48 kHz queue's samples each with the limit's share it came with), and `process` makes nothing itself: what changes lives in typed arrays. V8 allocates while it compiles the code, though: on Node 25, about 120 KB in a process's first half second of streaming and 1.86 MB in the next, then under 1.4 KB a half second; on Node 22, up to 11.5 KB in a half second (once 319 KB, in a first half second); on Node 20 under its default flags, in 4 of 8 processes, 12 to 17 KB every half second for as long as measured, 20 s of a first stream and 10 s of a second ([Speed and memory](#speed-and-memory)). @audio/compile's validator reads 104 to 872 bytes over a run in the young generation and the typed arrays' stores (its line: 6144) on Node 20, 22 and 25, at rest, moving, in blocks of 1 to 128, loud, and in a second instance; in its runs at rest the old generation takes 0 to 6056 bytes more on Node 20 (6056 in one of four runs), 128 or 3104 on 22, 152 on 25.

`limit` glides: a new value's share of the input is reached in a straight line over 20 ms from the share in force at the call that brings it, and each input sample is mixed back at the share in force as it came in, so a change is heard the latency later, on the input it came with. A jump gives the same samples however the host cuts its calls; a host's automation curve hands each call the value at its first frame, so where the host cuts a curve shifts the output. At 48 kHz every output sample is what the two values give held, mixed by the line: within 2^-23 of the larger where the music guard's gain is 1 (as with `music: 'enhance'`); while that gain ramps under `'pass'` its blend rounds once more, within 2^-23 of twice the larger plus the input's sample. At other rates the converter back spreads the line's corners over its taps, which can carry a sample past both: over a jump across the range (from 0 to 100 and back, and from 16 to 100), the peak passes the louder held one by at most 0.01 dB on a full-scale square at 3 kHz, DC and a tone near the band's edge, at 8, 16, 22.05, 44.1 and 96 kHz (`test.js`). 20 ms because a line's corners spread over every frequency, the louder the shorter the line, and are heard where RNNoise's output and the input part (a held chord in `'enhance'`), while a host's curve lags by about the line's length; it is also the smoothing @audio/dynamics declares for its gains. An instance sounds at the limit it is built with from its first sample, and a value held gives what it gave before. It declares no `smoothing`, so that a host hands a jump as it is and the samples do not hang on where it cuts its calls. The share falls as the limit rises over the whole range, so a host's ramp or automation curve from one value to another passes only through shares between theirs: at 48 kHz every sample of a 200 ms curve from 16 or from 100 to 0 lies between the two renders held at its ends, at 44.1 kHz its peak at most the louder held one's (`test.js`). A switch of `music` moves the guard's gain along its 200 ms raised cosine from where it is ([Music](#music)). The tail is 77.3 ms at the default limit, 79.0 ms at the top and none at 0 (RNNoise's biquad, below, which rings in RNNoise's part of the output alone), less where the limit mixes back more of the input, which stops with it; `audio` renders that much more of every input, so a 3 s take at 44.1 kHz comes back 3407 samples longer. A call longer than `maxBlockSize` or wider than `maxChannels` throws a RangeError naming what it carried and the bound, a ctx without either a TypeError. The first processor a process makes runs test signals through processors of its own before it returns (two channels in calls of 8 frames: 0.5 s at 48 and at 44.1 kHz call by call, then 1 s at 48 kHz; then the music guard's network 100 times), 0.63 to 0.73 s of CPU once over all threads, 0.41 to 0.47 s of it on the calling thread and the rest on V8's compiler threads, so that V8 compiles the code on every kind of value, every path and both kinds of converter before a stream: without it, the first digital silence after a loud stretch deoptimized RNNoise's pitch search there (41 MB at 48 kHz, once), a stream that began in silence deoptimized most of the code at its first sound, and on Node 20 and 22 a stream at another rate than 48 kHz could deoptimize on every frame.

## RNNoise

A translation of [xiph/rnnoise](https://gitlab.xiph.org/xiph/rnnoise) at 70f1d25, the network and all its DSP, following the portable C path (`vec.h` without SIMD) op for op in float32:

1. **Frame**: 480 samples at 48 kHz through a DC-blocking biquad; a 960-point FFT of the last two frames under a Vorbis window.
2. **Features** (65): the energies of 32 ERB-spaced bands up to 20 kHz as a log spectrum, its DCT; each band's correlation with a pitch-delayed copy of the signal, its DCT; the pitch period. The period comes from a 2:1 downsampled, LPC-whitened signal: a coarse search at 4:1, a fine one around the two best lags, octave errors removed.
3. **Network**: two 1-D convolutions (65·3 → 128 → 384, tanh), three GRUs of 384, dense layers to 32 band gains and a voice probability. The GRU and second convolution weights are int8 with per-row scales, and their input activations are quantized to int8, as upstream does.
4. **Pitch filter**: on the previous frame (one frame of lookahead), the pitch-delayed spectrum is added back per band where the gains would cut voiced harmonics, then the band energies are restored.
5. **Gains**: they fall at most by a factor 0.6 per frame (an RT60 of 135 ms), are interpolated across each band and applied; inverse FFT, overlap-add. Bins above 20 kHz are zeroed.

**After the input stops** the DC-blocking biquad of step 1 rings on (upstream's `rnn_biquad`, `a = [−1.99599, 0.996]`): its poles lie at 0.998·e^(±2πi·18.65/48000), a decay of 10.4 ms. Unfed, its output is r^n·(P cos nθ + Q sin nθ), P and Q set by its state, and over every tone from 0.05 to 200 Hz (steps of 0.01 Hz, each stopping at every phase at once: the largest singular value of the map from the tone's phase to P and Q) the envelope √(P² + Q²) is at most 1.4183 of the tone's peak, at 16.3 Hz; 1.2952 after a DC step; 2.0058 of a tone's RMS. It falls 60 dB in 74.5 ms after a DC step, 79.0 ms after the worst tone against its RMS. RNNoise passes that ring ungated where it takes it for voice or for silence (a DC step at −46 dBFS rings 72 to 76 ms through the contract atom, at 22.05 to 96 kHz, limits above 0 to 100 and none; a 16.3 Hz tone of 0.002 stopping at the phase after which the biquad rings longest, 77.2 ms of the 77.3 ms tail at the default limit, 78.7 of 79.0 at the top, 100) and gates it otherwise (the same step at −6 dBFS: 1 ms); a copy of the code without the biquad rings 7 ms at most, the window's overlap-add past an edge. That ring is the contract atom's tail (In `audio`). A host reads the tail from the values in force when it asks, at the input's end, while a new `limit`'s share glides over 20 ms and RNNoise's part of the output rings on until it arrives: so a `limit` moved toward 0 5 ms before the stop leaves 6.7 to 8.7 ms of the ring over −60 dB unrendered, and one moved at the stop or after it up to 21.3 ms (100 → 0 25 ms after, at 48 kHz), where a move toward a larger limit leaves none (`scripts/limit-stop.mjs`, at 44.1 and 48 kHz). A host that moves `limit` there renders the ring whole by reading the tail at the largest value in force from 20 ms before the stop on, since under a glide toward 0 RNNoise's part only falls.

The float code keeps C's float semantics: each float op is `Math.fround(a op b)`, exact for +, −, ×, ÷ and √ of float32 operands (Figueroa 1995), and values stay double where the C promotes to double. The int8 products are integers under 2²⁴, exact in any order, so they run in a WebAssembly SIMD kernel ([gemv.wat](./gemv.wat): widening to i16, `i32x4.dot_i16x8_s`) where the runtime has SIMD, 8.5 times faster than the JS loop that runs elsewhere; both give the same bits.

## DeepFilterNet3

Upstream's three ONNX graphs (encoder, ERB-gain decoder, deep-filter decoder) run unchanged; the rest is ported from DeepFilterNet at d375b2d, `libDF` and `df.enhance.enhance()`:

1. **STFT**: 960-point Vorbis window, 480 hop, the input padded by 960 zeros; the same FFT as RNNoise.
2. **Features**: power in 32 ERB bands (libDF's `erb_fb`: 2 to 67 bins wide) in dB, mean-normalized with α = 0.99 (τ = 1 s); the lowest 96 bins, unit-normalized with the same α. Both run 2 frames ahead of the output (`conv_lookahead`). `denoise()` computes them as if the input's speech (the louder half of its 50 ms frames) were at −20 dBFS, the median of the VoiceBank+DEMAND test set (−20.2): they are not level-free, the ERB means start at −60 to −90 dB and the complex features scale with the root of the level, and the model was trained at speech gains of −6, 0 and +6 dB (libDF `dataset.rs`). Heard as it is, a take 24 dB quieter lost its pauses' noise by 11 to 14 dB instead of 22 to 24, and the noise after 2.7 s of digital silence by 10 to 24 dB; at −20 dBFS it does as the louder take, at no cost on VoiceBank+DEMAND (PESQ 3.077 against 3.076, 206 files). The ERB bands wholly above the input's band edge hear white noise 20 dB under that speech: upstream trained on full-band 48 kHz mixtures only (its `config.ini`: `p_bandwidth_ext = 0`, libDF's `BandwidthLimiterAugmentation` off; noise at −5 to 40 dB SNR), so the network never heard a band empty, yet a 16 kHz file leaves everything above 8 kHz empty and a codec's low-pass everything above 16 to 20 kHz (nine of the ten narrations below end between 16.7 and 21 kHz). Such bands reached it as libDF's constant 1e-10 floor, a band without a trace of noise, which no mixture it trained on had, and it cleaned the band below worse: on VoiceBank+DEMAND's training speakers at 16 kHz, unlimited, PESQ 2.26 against 2.57 full-band; over the floor, 2.42 ([Accuracy](#accuracy)). The edge is the highest frequency whose long-term power lies within 60 dB of the 1 to 4 kHz median, measured at the input's own rate before resampling (a resampler's images don't count); a band holding the edge keeps the input's own content, so a 44.1 kHz input, whose top band (20.7 to 24 kHz) still holds 20.7 to 22.05 kHz, is processed as before. Every bin of an empty band draws its power from an exponential distribution in every frame, from a fixed seed, as white noise's STFT does: a constant floor would help only the first seconds, since the mean normalization (τ = 1 s) takes a constant band's feature to zero, while noise keeps moving it. The spectrum the gains and filters apply to stays as it was, and a full-band input (all of VoiceBank+DEMAND at 48 kHz) is processed bit for bit as before.
3. **Network**: 32 gains (sigmoid) and 96 × 5 complex filter taps per frame.
4. **Enhance**: every bin takes its band's gain; below bin 96 (4.8 kHz) the output is instead Σₖ spec[t − 2 + k] · coef[t, k], k = 0–4, over the unmasked spectrum; `enhance()`'s `limit` (none unless given) mixes the noisy spectrum back in per bin at 10^(−limit/20), as upstream's `atten_lim_db` does above 0 (there 0 is no limit, here the input). Inverse FFT, overlap-add, the 480-sample STFT delay removed. `denoise()` runs it unlimited and mixes the input back after, in time, by `mixback()`: what the model took drops by `limit` dB, and in the voice's pauses down to `floor` dB under the voice where the limit would leave it closer (its power over the 0.5 s around each 10 ms frame against the voice's level, the louder half of the output's 50 ms frames). The floor weighs by speech absence, as OM-LSA's gain floor does (Cohen & Berdugo 2001: G_H1^p · G_min^(1−p)), speech read off the model's output as `@audio/vad` grows it: a 10 ms frame within 30 dB of the voice's level, gaps up to 0.15 s inside it, 0.3 s either side. At low SNR the model takes some of the voice for noise (a telephone-band voice under white noise 10 dB down: 10 dB or more from 7 % of its voiced frames, unlimited), and under the voice what it removed holds that too; the limit is what gives it back. `floor: 0` gives the per-bin limit's output, but for the voice guard's order.
5. **Voice guard** (`denoise()`): the model takes held voicing for noise. A clean held note from VocalSet (Wilkins 2018) comes out 39 dB down, unlimited, its own local-SNR estimate at its floor (−15 dB), while speech keeps its level; sung phrases lose 24 to 34 dB. So where the input holds sustained voicing, runs of 0.3 s or more periodic at RNNoise's pitch (normalized correlation 0.45 or more, Praat's voicing threshold, the pitch searched every 20 ms) that stand 6 dB or more over the quietest 100 ms of the 20 s around (minimum statistics, Martin 2001: a buzz is that background, a note stands over it), each band holding a harmonic keeps at least c² of its input, c its correlation with the frame one and two periods earlier: the periodic share of its power, as RNNoise's pitch filter restores what its band gains would remove. Between the harmonics and in the pauses the model's output stands.

The parameters come from the export's `config.ini`, so upstream's low-latency export (`DeepFilterNet3_ll_onnx.tar.gz`: no lookahead, 36 MB) loads through `weights` too.

**Long inputs** run in chunks of `chunk` frames (10 s). Upstream feeds the whole file at once, about 8 MB of activations per second of audio (peak RSS 768 MB for 30 s, 1 GB for 60 s in onnxruntime-node); in 10 s chunks a 180 s file peaks at 563 MB, and any length stays there. The ONNX graphs are exactly causal (outputs on a prefix equal the whole run's) but take no GRU state, so each chunk after the first starts `warmup` frames (3 s) early from zero state, and its first `fade` frames (0.5 s) crossfade from the previous chunk's run, which continued that far. The GRUs remember how they started: 6 s after such a start the gains still differ from the whole-file run's by up to 0.1–0.5. Chunked output is another valid run, not the whole-file one; on VoiceBank+DEMAND in 1 s chunks (1 s warm-up, 0.2 s crossfade, 2 to 3 chunks per utterance) the scores do not move: PESQ 3.164 against 3.162 whole-file, STOI 0.9450 against 0.9452, SI-SDR 18.94 against 18.96 dB, though single files shift by up to 0.31 PESQ either way. The STFT and features stay exact across chunks. Exact chunking needs a stateful re-export (GRU states and convolution buffers as graph inputs and outputs), which is also what live DeepFilterNet3 needs.

## Music

Both models enhance speech, and to them everything else is noise: at their default limits music lost 8 to 16 dB in every band, songs included (Accuracy, Music). So by default (`music: 'pass'`) a classifier segments the input into speech, music and noise, and the model's output is kept only where it hears speech or noise; where it hears music the input passes through, bit for bit. What counts as what follows the classifier's labels as its README gives them, "Zones corresponding to speech over music or speech over noise are tagged as speech. Singing voice is tagged as music." (inaSpeechSegmenter at [7a41b80](https://github.com/ina-foss/inaSpeechSegmenter/blob/7a41b809c3cc6d8d36b24e70784779f8d16f7b5e/README.md), lines 14–15; what it trained on is not stated, Licenses): speech over a music bed is speech, so the bed under a voice-over is cleaned with the noise; a song is music, its singer included, since a speech enhancer would strip the accompaniment from under the voice and dull the voice too. A voice singing alone sits between the two (below). `music: 'enhance'` enhances everything, as before 0.4.

1. **Classifier**: inaSpeechSegmenter's speech/music/noise CNN (Doukhan et al., ICASSP 2018; MIT; its `smn` engine), 782,403 weights, in the package as float16 (`guard.bin`, 1.56 MB; `scripts/guard.py` writes it from upstream's Keras export, the batch norms folded). Its input as ina computes it: 16 kHz (here a 31-tap low-pass and every third sample of the 48 kHz signal), pre-emphasis 0.97, 25 ms Hann frames every 10 ms, 24 mel bands from 100 Hz to 8 kHz, log; the lowest 21 bands over 0.68 s make a patch, standardized by its own mean and deviation, so the level does not matter. One patch every 100 ms (ina takes one every 20 ms; on the tuning material every 100 ms decides the same).
2. **Offline** (DeepFilterNet3, `denoise()` and the `deepfilter` op): segments as ina makes them, Viterbi over the whole input, three states, a switch costing 10⁻⁸⁰ per 20 ms (ina's own constant), so a segment lasts a second or more and a stray patch moves nothing; from 0.5 each patch counts 1 nat in favour of speech, the stream's prior (step 3). A voice under a music bed as loud as itself reaches ina as music about half the time: it hears speech there at a mean probability of 0.47, in songs at 0.01 (medians, training sets). With the prior, of VoiceBank's test takes under MUSDB18 test accompaniments at the voice's level 37% of the frames pass instead of 51%, 5 dB under it 0.6% instead of 2.9%, and the training sets' songs and accompaniments pass every frame they passed without it (at 1.5 nats songs lost 1.1%: the prior chosen there). An input that is music throughout never reaches the model.
3. **Streaming** (RNNoise: `denoise()`, the contract atom): a stream decides on what has arrived. The forward recursion of the same chain for two states (music, the rest) keeps the log-odds of music, bounded at ±20 nats; a patch moves them by at most ln 100, and by 1 nat less (a prior for speech), and the stream switches to music above 5 and back below 0. Each output frame takes the decision made once the input frame two on is in (RNNoise's own delay), so nothing is added to the latency, and `denoise()` and the atom give the same samples. The price: music is denoised for its first second or so, and each switch comes about a second late.
4. **The gain** between the enhanced signal and the input moves over 200 ms, raised cosine: offline centered on the segment boundary, streaming from the decision on. It is applied at the input's rate (the 48 kHz gain resampled with the signal), so what passes is the input itself, not its round trip through 48 kHz.

`guard(x, sampleRate)` gives that offline gain for any enhancement, per sample of `x` at its own rate: 1 where it is speech or noise, 0 where it is music, ramped as above; `g·y + (1 − g)·x` passes music untouched. `audio`'s `dereverb` guards its de-reverb model by it.

It runs in JS: a stream's `process()` can't await ONNX Runtime, and RNNoise needs no runtime. The three upper convolutions (through im2col) and the dense layers run in a 1,074-byte WebAssembly SIMD kernel (`gemm.wat`), the first convolution once per frame and shared by the overlapping patches (the max pooling after it commutes with each patch's standardization); 1.1 s of CPU per minute of audio, 11 s on the JS path where WebAssembly SIMD is missing (same bits).

`segments(x, sampleRate)` gives the classifier's own reading of an input: what it is, stretch by stretch, `[{ kind, start, end }]` in seconds, `kind` one of `'speech'`, `'music'`, `'noise'`, in order and end to end over the input. It is step 2 without the prior for speech, which is the guard's: ina's decision as it stands, each 10 ms frame taking the patch nearest it. Digital silence takes the kind around it; an input of nothing but silence has no segments. A host names a recording by it (`audio`'s editor: speech, music, both).

```js
import { segments } from '@audio/neural-denoise'

await segments(pcm, 44100)  // [{ kind: 'music', start: 0, end: 2.5 }, { kind: 'speech', start: 2.5, end: 11.27 }]
```

What it names the test sets below (`node scripts/accuracy.mjs kinds test`): the share of each set's active 10 ms frames named each kind, and its files by the kind most of their active frames have.

| Test set | Speech | Music | Noise | Files, by their most frames |
|---|---|---|---|---|
| VoiceBank+DEMAND, clean / noisy (824 each) | 99.9% / 99.8% | 0.0% / 0.0% | 0.1% / 0.2% | 823 / 822 speech, 1 / 2 noise |
| the same in white noise, 20 / 10 / 5 dB (275 each) | 99.6% / 99.6% / 98.9% | 0.0% | 0.4% / 0.4% / 1.1% | 274 / 274 / 273 speech, the rest noise |
| the same in pink noise, 20 / 10 / 5 dB | 99.6% / 99.4% / 99.4% | 0.0% | 0.4% / 0.6% / 0.6% | 274 speech each, 1 noise |
| over a music bed 20 / 15 / 10 dB down (275 each) | 99.6% / 99.3% / 98.6% | 0.4% / 0.7% / 1.4% | 0.0% | 274 / 274 / 272 speech, the rest music |
| ten home narrations (Rooms) | 99.9% | 0.0% | 0.1% | 10 speech |
| VocalSet, spoken excerpts (10) | 98.2% | 1.8% | 0.0% | 10 speech |
| MUSDB18 test previews, songs (50) | 0.0% | 100% | 0.0% | 50 music |
| the same, accompaniment only (50) | 0.0% | 100% | 0.0% | 50 music |
| Slakh2100 mixes 11 to 20 | 0.0% | 100% | 0.0% | 10 music |
| the four repair pieces | 0.0% | 95.7% | 4.3% | 3 music, 1 noise |
| the programmes (8), speech and music in turn | 53.3% | 46.7% | 0.0% | |
| MUSDB18 vocal stems (50) | 38.1% | 57.5% | 4.4% | 28 music, 19 speech, 3 noise |
| VocalSet sung phrases (20) / long tones (10) | 26.8% / 11.5% | 66.4% / 67.8% | 6.8% / 20.7% | 14 music, 6 speech / 8 music, 1 speech, 1 noise |
| DEMAND noise alone (noisy − clean, 824) | 13.6% | 7.8% | 78.6% | 637 noise, 120 speech, 67 music |

Speech is named speech and music music wherever the input is one of them (98.2% of the frames or more). A voice singing alone is either. What is neither is called noise four frames in five, speech or music otherwise: of DEMAND's noises alone, 187 of 824 files are named for a kind they are not, so its `'noise'` is the weak answer of the three.

**Accuracy**, on labelled sets (`python scripts/accuracy.py guard-sets`, `node scripts/accuracy.mjs guard train|test`): the share of active 10 ms frames passed (within 40 dB of the loudest). The constants (one patch per 100 ms, the streaming bound, prior and hysteresis) were chosen on the training sets: VoiceBank+DEMAND's training speakers, the MUSDB18 training previews, Slakh2100 mixes 1 to 10, VocalSet singers 1 to 5, the Spoken Wikipedia narrations not in Rooms; the test sets below were run once. Beds: clean utterances over a song's accompaniment (the MUSDB18 stems but the vocals) 20, 15 and 10 dB under the speech; hiss: white and pink Gaussian noise at 20, 10 and 5 dB SNR. Programmes: 8 sequences of six segments, speech (five utterances of one kind, or 15 s of narration) and music (up to 20 s of a song, its accompaniment or a Slakh mix) in turn, scored 0.5 s or more from each cut.

| Test set | Frames | Passed, offline (DeepFilterNet3) | Passed, streaming (RNNoise) |
|---|---|---|---|
| **Speech, to enhance** | | | |
| VoiceBank+DEMAND, clean / noisy (824 each) | | 0.0% / 0.0% | 0.0% / 0.0% |
| the same in white noise, 20 / 10 / 5 dB (275 each) | | 0.0% / 0.0% / 0.0% | 0.0% / 0.0% / 0.0% |
| the same in pink noise, 20 / 10 / 5 dB | | 0.0% / 0.0% / 0.0% | 0.0% / 0.0% / 0.0% |
| over a music bed 20 / 15 / 10 dB down (275 each) | | 0.4% / 0.4% / 0.8% (0.4: 0.4% / 0.7% / 1.4%) | 0.7% / 1.4% / 2.3% |
| ten home narrations (Rooms) | | 0.0% | 0.0% |
| VocalSet, spoken excerpts (10) | | 1.8% | 2.1% |
| DEMAND noise alone (noisy − clean) | | 6.6% (0.4: 7.8%) | 2.4% |
| **Music, to pass** | | | |
| MUSDB18 test previews, songs (50) | | 100% | 87.2% |
| the same, accompaniment only (50) | | 100% | 87.7% |
| Slakh2100 mixes 11 to 20, 60 s each | | 100% | 98.6% |
| the four repair pieces, 60 s each | | 95.7% | 93.8% |
| **Programmes** (8): speech enhanced / music passed | | 100% / 100% | 98.5% / 92.9%, switches 0.96 s late (median) |
| **Singing alone** | | | |
| MUSDB18 vocal stems (50) | | 47.7% (0.4: 57.5%) | 32.4% |
| VocalSet sung phrases (20) / long tones (10) | | 58.5% / 57.1% (0.4: 66.4% / 67.8%) | 34.3% / 40.0% |

Offline, every speech set keeps 98.2% or more of its frames enhanced (noisy and hissy speech all of them) and every music set but one passes all of its frames; on the programmes nothing errs 0.5 s from a cut. Streaming, speech is kept as well, and music loses its first second: the previews are 7 s long, so 13% of their frames; the 60 s Slakh mixes, 1.4%. The noise alone that passes is DEMAND's living room (30% of its frames), which has music in it. Of the pieces, `trumpet` (solo) is enhanced for its first 4 s and `brahms` for 5 s of a quiet passage. A voice singing alone is split, about half of its frames passed offline (two thirds before 0.5's prior): held notes and phrases the classifier calls music pass, the rest is enhanced, where DeepFilterNet3's voice guard keeps sustained voicing ([DeepFilterNet3](#deepfilternet3), step 5); enhanced, a VocalSet phrase comes out 1.3 dB down and a MUSDB18 vocal stem 3.1 dB (medians of 10; held tones 0.7). Speech over a bed 10 dB down, louder than a podcast's, still counts as speech.

What it changes, at the default limits (Accuracy: the same sets, `python scripts/accuracy.py music`; per band the median over the files, the worst in parentheses):

| | SI-SDR to the input (median) | 0–250 Hz | 0.25–1 kHz | 1–4 kHz | 4–8 kHz | 8–16 kHz | 16–22 kHz |
|---|---|---|---|---|---|---|---|
| DeepFilterNet3, songs (50 MUSDB18 previews): 0.3 | 4.0 dB | −16.9 dB | −13.7 | −16.5 | −13.2 | −12.7 | −16.0 |
| **0.4** | **the input, bit for bit (all 50)** | **0.0** | **0.0** | **0.0** | **0.0** | **0.0** | **0.0** |
| DeepFilterNet3, pieces (4): 0.3 | 1.1 dB | −8.6 (−13.0) | −9.0 (−12.1) | −10.9 (−14.8) | −14.4 (−16.6) | −15.5 (−17.4) | −14.0 (−16.1) |
| **0.4** | **2 of 4 bit for bit** | **0.0 (−0.0)** | **−0.0 (−1.9)** | **−0.1 (−1.7)** | **−0.1 (−4.5)** | **−0.7 (−10.2)** | **−1.0 (−13.0)** |
| DeepFilterNet3, VocalSet chords (5): 0.3 / **0.4** | 3.5 dB / **bit for bit** | −5.8 / **0.0** | −3.9 / **0.0** | −4.9 / **0.0** | −9.5 / **0.0** | −11.1 / **0.0** | −3.7 / **0.0** |
| RNNoise, songs: 0.3 | 3.9 dB | −14.4 (−16.0) | −8.0 (−15.1) | −10.1 (−15.8) | −11.4 (−16.0) | −12.2 (−16.0) | −14.2 (−16.0) |
| **0.4** (the first second denoised) | **11.4 dB** | **−0.6 (−2.3)** | **−0.4 (−2.5)** | **−0.4 (−2.5)** | **−0.5 (−2.2)** | **−0.6 (−2.2)** | **−0.6 (−3.7)** |
| RNNoise, pieces: 0.3 / **0.4** | 5.8 / **18.3 dB** | −11.5 / **−0.1** | −12.9 / **−0.1** | −13.7 / **−0.1** | −15.3 / **−0.1** | −14.2 / **−0.5** | −11.0 / **−0.6** |

Speech is untouched by the guard: on VoiceBank+DEMAND's 824 noisy test utterances both models' default outputs are 0.3's bit for bit (PESQ 2.90 and 2.49, the Accuracy table's rows), and so are those of the ten narrations; lena, a film scene with music under the voice, has its musical opening passed.

The classifier was chosen against two others, run in Python on the same training sets, each frame scored as music where its music score beats its speech score (Viterbi-smoothed alike): YAMNet (Google, Apache-2.0; AudioSet's 521 classes, 3.7M weights, 0.96 s frames) passed music as well (songs 100%, accompaniment 100%) but speech over a bed 10 / 15 / 20 dB down 29 / 9.6 / 0.9% of the time (ina, offline: 2.2 / 1.2 / 0.4%); Silero VAD (MIT) tells speech from everything else, noise included, and called 17 to 30% of VoiceBank's clean speech frames non-speech (the pauses). PANNs CNN14 (80M weights) and BEATs (90M) were not tried: a guard should cost a fraction of the model it guards.

## Verification

| Check | Result |
|---|---|
| RNNoise vs upstream C, portable build (`cc -O2 -ffp-contract=off -DDISABLE_NEON`), default and little models | **bit-exact**, output and voice probability: lena (1127 frames), lena + noise (1127), digital silence and near-silence (300), a VoiceBank+DEMAND utterance (174) |
| RNNoise vs upstream C as `configure && make` builds it on arm64 (NEON, fused multiply-adds) | 38.8–62.4 dB SNR, voice probability within 0.087: upstream's builds differ from each other by this much (NEON's `vrecpeq_f32` reciprocal estimate inside tanh and sigmoid) |
| `rnnoise.bin` | byte for byte what upstream's `dump_weights_blob` writes from the model tarball (sha256 `1ad07b42…`) |
| FFT, window and DCT tables | equal to upstream's `rnnoise_tables.c`: 960 twiddles, 480 window values, 1024 DCT entries, the bit reversal; the FFT matches a direct DFT to 3.2e-9 |
| DeepFilterNet3, `enhance()` with upstream's settings (no level change, no voice guard, whole file), vs Python `df.enhance.enhance()` (deepfilternet 0.5.6, PyTorch checkpoint), same input, unlimited (`atten_lim_db` not given, `enhance()`'s default) | 133.5 dB SNR on a 1.7 s utterance, 128.2 dB on 11.3 s (max \|d\| 5.7e-7), 133.8 dB on an utterance led by 2 s of noise, 134.5 dB on a 60 s narration, its first second included; ERB widths and α identical |
| DeepFilterNet3 voice guard, no model (`test.js`) | a held /a/ (190 Hz, 5 Hz vibrato) 20 dB over white noise through a model that removes everything: kept within 2.8 dB, the noise around it removed; a 120 Hz buzz under the same noise: no frame guarded |
| DeepFilterNet3 plumbing, no model (`test.js`) | identity gains and taps give the input back to 1.8e-7, whole and chunked (chunks identical to the whole run); a tap one frame ahead advances the band below 4.8 kHz by exactly 480 samples (to 8.9e-8) |
| DeepFilterNet3 band edge, no model (`test.js`) | the edge of white noise at 48 kHz: 24 kHz; at 16 kHz: 8 kHz; low-passed at 15 kHz: 15 to 16.5 kHz. A 16 kHz input's top band reaches the encoder as the floor (first feature 0.416, as computed; 0.2 fed −0.246, libDF's 1e-10) and moves frame to frame; a full-band input's features are bit for bit those without the floor; identity gains and taps still give the input back |
| Edge cases (`test.js`), both models, 48 and 16 kHz | empty input, one sample, 100 samples: the same length back, finite; a second of digital silence stays digital silence |
| Worklet (`test.js`, AudioWorkletGlobalScope simulated) | the frame API's output, delayed by 448 samples, with no limit (`Infinity`); at 12 and at the default 16, that output with the input mixed back, to 1e-7; at `limit: 0` the input itself, 1408 samples late, sample for sample; a limit that is no number, a TypeError |
| Worklet in Chromium 153 (headless, Playwright, `scripts/worklet.mjs`) | runs, no processor error, output non-silent; live underruns in Speed |
| Music guard (`test.js`) against the float32 Keras model, the same input (`scripts/guard.py` → `fixtures/guard.json`) | lena, 106 patches: probabilities within 1.7e-3 (float16 weights); the wasm kernel and the JS loop give the same bits |
| `mixback()` (`test.js`), a perfect model: a tone for 4.5 s, then a pause, and white noise | noise 40 and 25 dB under the voice: 18 dB down (58 and 43 under it), under the voice and in its pause; 10 dB under and 5 dB over: in the pause 40 under it, the floor, under the voice 18 down (28 and 13 under it); `floor: 0`: 18 down; `limit: 0`: under the voice nothing taken (noise 10 dB under stays 10 under), in its pause the floor, and with `floor: 0` too the input bit for bit; `limit: Infinity`: the model's output; a limit that is no number, `null` among them, a TypeError; a noise that drops 30 dB in the pause: each part as such 0.5 s from the change; where input and output agree, the input bit for bit; `denoise()`'s default is the unlimited output through it, bit for bit |
| `mixback()` (`test.js`), a model that takes the voice in places (0.2 s of every 0.5 s) under white noise 10 dB down | the voice it took comes back 18.0 dB down at the default limit, 6.0 at `limit: 6` (`everywhere`, 0.7.0's mix: 36.9 down at both) |
| `mixback()` (`test.js`), the perfect model's take with the noise loud then quiet, a stretch the model took and, in the pause, residue 27 dB under the voice | the default's samples are 0.8.0's and `everywhere`'s 0.7.0's, at `limit: 18`, `floor: 40` and at `limit: 6`, `floor: 30` (their SHA-256 from the packed releases, `scripts/floor.mjs fixture`); `floor: 0` the same either way; `denoise()` passes `everywhere` through |
| `denoise()` with DeepFilterNet3 (`test.js`, with the model and VoiceBank+DEMAND's test set in the cache): p232_003 through a telephone band (300–3400 Hz) under white noise through it 10 dB down | 0 of its 188 20 ms frames within 20 dB of its level more than 21 dB under the clean voice (0.7, and the model unlimited: 12) |
| Music guard's prior (`test.js`, with VoiceBank+DEMAND and the guard sets in the data cache) | speaker p226's first five training takes under a MUSDB18 training accompaniment at their level: 13% of the frames passed (0.4: all) |
| Music guard (`test.js`): a synthetic band (chords of plucked tones, a bass, a tick) | DeepFilterNet3, through a stand-in that removes everything: the band back bit for bit at 48 and 44.1 kHz, `music: 'enhance'` removes it; RNNoise: the band itself from 1.1 s on; band, noisy speech, band (with VoiceBank+DEMAND in the data cache): two switches, each within 0.5 s of its cut, gain exactly 0 and 1 between 200 ms raised-cosine ramps; RNNoise's output to frame f − 2 the same whatever follows frame f |
| Contract atom (`test.js`) against `denoise()`, delayed by its declared latency | 48 kHz: equal sample for sample in blocks of 1, 128, 1024 and 1 to 3000 samples, limits 20, 100 and 0; 44.1 kHz (resample-sinc 1.2.0): equal but in the last 50 ms; with resample-sinc 1.1.2, whose arithmetic differs, 59.5 dB SNR |
| Contract atom's allocation (`test.js`, V8's heap and the typed arrays' stores as @audio/compile's validator counts them, in a child Node) | a second instance's first block, and its blocks of 8 frames over a run, within the validator's 6144 bytes a run (where each channel's state was made on its first block, that block took 1.05 MB, and a block 1334 bytes) |
| RNNoise on digital silence after speech (`test.js`) | from 0.3 s into a second of it after two seconds of lena, every frame's voice probability 0, from `process()` and in `voice[0]`, as upstream's `vad_prob` stays 0 where the network does not run |
| Contract atom's first stream (`test.js`, as above) | each half second from 3 to 6 s within the validator's 6144 bytes, where the guard's frame counter first wraps (4.08 s): 2.06 MB at 4.0 to 4.5 s before its wrap was arithmetic |
| Contract atom, music passing in blocks of `maxBlockSize` (4096) (`test.js`) | `denoise()` sample for sample at 48 kHz, and at 44.1 kHz but the last 50 ms |
| Contract atom's tail (`test.js`: the test's own host, and `audio`'s toStream where it is installed or checked out beside, the same samples) | at the default limit the host renders 77.3 ms; past it every 64-frame window is 60 dB under the input's RMS at 48 and 44.1 kHz, limits 16 and 100, after a 16.3 Hz tone of 0.002 stopping at the phase after which the biquad alone rings longest (over that line in the 2 ms before the tail, so a tail 2 ms shorter fails), a DC step at −46 dBFS (which rings 72 ms) and a 312 Hz tone of 0.5 (37 ms); at the top, 100, the tail, 79.05 ms, covers the biquad's own 78.77 ms after that tone and 74.5 ms after a DC step |
| Contract atom's `limit` (`test.js`, as above) | a jump from 1 to 100 dB under a 220 Hz tone and from 16 to 100 under 1 kHz, `music: 'enhance'`: at 48 kHz the output is the renders held at the two values mixed by a 20 ms line from the call's first sample, within 2^-23 of the larger at every sample, the first itself before the call and the second itself from the line's end; at 44.1 kHz the same but for the converter's 17 taps either side, and on these tones its peak over the move never over the louder held one (before: 0.60 of the change in one sample at 48 kHz and 0.88 at 44.1, then a line as long as the call before the jump, up to 32 frames) |
| Contract atom, a jump of `limit` in calls of 1, 8, 31 and 33 frames and of 1 to 128 at random (`test.js`, as above) | the samples calls of 128 give, bit for bit, at 48 and 44.1 kHz, on and off the 32-frame grid (before: 0.018 to 0.072 apart on a 1 kHz tone of 0.5) |
| Contract atom, a value of `limit` while a line moves (`test.js`, as above) | 1 → 100 dB, then 16 half a line later, on a band at 48 kHz: the renders held at 1 and 100 mixed at each sample by a share computed from the two lines, the second from where the first is when 16 comes: within 2^-23 of the larger with `music: 'enhance'` (at most 0.71·2^-23 measured); with `music` switched to `'pass'` so that the lines fall in the guard's ramp (its gain between 0 and 1 at each of their 1440 samples), within 2^-23 of twice the larger plus the input's sample (at most 1.10·2^-23 of the larger measured); the render held at 16 from the second line's end |
| Contract atom, two instances (`test.js`, as above) | called in turn, a 1 kHz tone in `'enhance'` in calls of 128 with `limit` 16 → 100 → 40, and stereo speech in calls of 37 with 1 → 100: each the samples it gives alone, bit for bit |
| Contract atom's `limit` at other rates (`test.js`) | jumps 0 → 100, 100 → 0 and 16 → 100, `music: 'enhance'`, under a full-scale square at 3 kHz, DC of 0.5 and a tone of 0.5 at 0.45 of the rate (of 48 kHz at 96 kHz): the peak over the move at most 0.01 dB over the louder held one at 8, 16, 22.05, 44.1 and 96 kHz (measured: 0.0072 dB at 8 kHz, the square; 0.0008 to 0.0014 dB at the others, DC) |
| `limit: 0` (`test.js`, as above) | `denoise()` at 48 kHz: the input, sample for sample; at 44.1 kHz the input through resample-sinc to 48 kHz and back, sample for sample (35.7 dB SNR from that input: noisy speech with noise to 22.05 kHz); the contract atom the same, the latency late, at 48 kHz in `'pass'` and `'enhance'`, no tail declared or rendered, and after a jump from 16 the input itself from the 20 ms line's last sample; at 44.1 kHz 60 dB under the input from 128 frames past its end; a limit that is no number, a TypeError |
| Contract atom's top, `limit: 100` (`test.js`, as above) | at 48 kHz the render with no limit plus 10^−5 of the input's difference from it, within float32's rounding (2^-23 of the larger; in `'pass'`, of twice the larger plus the input's sample), at every sample: 3.1e-6 at most from the unlimited render on noisy speech |
| A host's curve of `limit` into 0 (`test.js`, as above) | 200 ms, linear in dB, a value every 32 frames, from 16 and from 100 to 0, under white noise, a 1 kHz tone and noisy speech, `music: 'enhance'`: at 48 kHz every sample at most the larger of the two renders held at its ends, within 2^-23; at 44.1 kHz the peak over the move at most the louder held one's, 0.0000 dB over at most (before 0.8, 0 no limit: 13.5 dB over on the white noise, 16 to 0) |
| Contract atom's bounds (`test.js`) | a call of 129 frames at `maxBlockSize` 128, or of 3 channels at `maxChannels` 2, throws a RangeError naming both numbers (before: NaN out, or a TypeError); a ctx without either, a TypeError naming it |

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
| RNNoise, unlimited (this package = upstream C) | 2.11 | 0.890 | 12.28 | 3.28 | 3.85 | 2.94 |
| RNNoise, little model, unlimited | 2.12 | 0.893 | 12.39 | 3.28 | 3.85 | 2.93 |
| RNNoise, `limit: 20` (the default before 0.3) | 2.46 | 0.918 | 12.86 | 3.40 | 3.71 | 2.97 |
| RNNoise, `limit: 16`, the default | 2.49 | 0.922 | 13.00 | 3.43 | 3.66 | 2.96 |
| **DeepFilterNet3**, upstream's settings, unlimited (`enhance()`) | **3.16** | **0.945** | **18.96** | **3.44** | **4.08** | **3.18** |
| DeepFilterNet3, Python (deepfilternet 0.5.6) | 3.16 | 0.945 | 18.96 | 3.44 | 4.08 | 3.18 |
| DeepFilterNet3 in 1 s chunks, unlimited | 3.16 | 0.945 | 18.94 | 3.44 | 4.08 | 3.18 |
| DeepFilterNet3, `denoise()`, `limit: Infinity`: heard at −20 dBFS, voice guard | 3.15 | 0.945 | 18.95 | 3.44 | 4.08 | 3.18 |
| DeepFilterNet3, `denoise()`, `limit: 18`, `floor: 0` (the default before 0.5) | 2.90 | 0.942 | 17.91 | 3.48 | 3.84 | 3.09 |
| **DeepFilterNet3, `denoise()`, `limit: 18`, `floor: 40`, the default** | **2.90** | **0.943** | **17.95** | **3.47** | **3.85** | **3.09** |
| DeepFilterNet3, `denoise()`, `everywhere: true` (the floor under the voice too: 0.7.0's mix) | 3.04 | 0.944 | 18.86 | 3.44 | 4.00 | 3.15 |
| DeepFilterNet3, `limit: 12` (the default before 0.2) | 2.67 | 0.939 | 16.18 | 3.49 | 3.69 | 3.03 |

Over the noisy input, paired, with 95% intervals: DeepFilterNet3 adds 1.20 ± 0.03 PESQ, 0.024 ± 0.002 STOI and 10.5 ± 0.3 dB SI-SDR, and beats every classical row's PESQ on 804 of 824 files; RNNoise adds 0.14 ± 0.04 PESQ and 3.8 ± 0.3 dB SI-SDR but loses 0.031 ± 0.004 STOI; `omlsa`, the best classical row, adds 0.43 ± 0.03 PESQ and 6.1 ± 0.2 dB SI-SDR and loses 0.006 STOI. A limit leaves noise in on this set, and spares the voice: paired, on a quarter of the files (206), DNSMOS SIG moves −0.002 ± 0.015 from 12 to 18 dB, then −0.008 ± 0.003 to 20, −0.019 ± 0.006 more to 24 and −0.049 ± 0.017 more unlimited, while BAK rises 0.18, 0.04, 0.07 and 0.16 and PESQ 0.25, 0.06, 0.08 and 0.10. Past 18 dB the speech itself loses, which DNSMOS reads as SIG and a listener as a filtered, phone-call voice; from 8 to 16 kHz the unlimited output carries 1.5 dB less than the clean speech, at 18 dB 0.9 (all 824 files). 18 dB alone added 0.93 PESQ over the input and 0.50 over `omlsa`; the floor everywhere (`everywhere: true`, 0.5 to 0.7's default) takes the noise that 18 dB would leave within 40 dB of the voice further down, on this set most of it (2.5 to 17.5 dB SNR): +0.15 PESQ, +0.17 BAK, +0.06 OVRL, −0.03 SIG against `floor: 0` (bench/speech.mjs `dfn-18` against `dfn-f0`, both through `audio`'s op, with 0.5). The default keeps the limit under the voice and, on this set, in most of the pauses ([DeepFilterNet3's floor](#deepfilternet3s-floor)). Pass `limit: Infinity` where the most noise removal matters more than room tone. The level change and the voice guard cost 0.01 PESQ unlimited and 0.004 at 18 dB. The same scoring gives upstream's published anchors: noisy 1.97 and 0.921 (as the DeepFilterNet2 paper's table), DeepFilterNet3 3.16 and 0.945 against the DeepFilterNet3 paper's 3.17 and 0.944. That DeepFilterNet2 table measured RNNoise's 2022 model at 2.33 and 0.922; the current model scores lower here.

- **RNNoise's current model sometimes removes the voice.** STOI drops by more than 0.1 on 68 of 824 files and by more than 0.2 on 12; on p232_354 it goes from 0.870 to 0.154, the output 24 dB under the clean level while RNNoise's own voice probability averages 0.84. Upstream's C does the same, the portable and the NEON build alike (0.154, 0.153). DeepFilterNet3 never loses more than 0.066 STOI, `wiener` 0.065, `omlsa` 0.059. A `limit` caps it: at the default 16 dB (the input mixed back at 0.16) losses over 0.1 fall to 4 files and over 0.2 to none (p232_354 keeps 0.824), and PESQ rises to 2.49, +0.53 ± 0.02 over the noisy input, at a BAK of 3.66 instead of 3.85. Against 0.2's 20 dB, paired: PESQ +0.029 ± 0.009, STOI +0.004, SIG +0.026 ± 0.004, BAK −0.047 ± 0.005 (6 files over 0.1, p232_354 0.786). `denoise()`, the worklet and the atom apply that limit unless given another.
- **The recipe's delivery stages are not denoisers.** Its compressor, EQ, de-esser and loudness target change the signal against the clean reference on purpose: the full recipe scores under its own denoising part. SI-SDR also falls with the 80 Hz high-pass: on its own it takes clean speech to 6.2 dB SI-SDR at PESQ 4.45 (21 test utterances). The table says what each stage does to speech in noise, not how finished a podcast sounds.

**RNNoise's limit**, chosen on VoiceBank+DEMAND's training speakers (504 utterances: 18 of each of its 28 speakers, none of them in the test set; noise 0 to 15 dB under the voice), scored as above, with the unlimited output mixed back with the input as `denoise()` mixes it. Clean word ends and quiet syllables: the same utterances' clean takes as input (Clean speech, below).

| `limit` | PESQ | STOI | DNSMOS SIG | BAK | STOI down > 0.2 | clean word ends | clean quiet syllables |
|---|---|---|---|---|---|---|---|
| noisy input | 1.47 | 0.841 | 3.00 | 2.50 | | | |
| none, `Infinity` (upstream) | 1.83 | 0.827 | 3.20 | 3.76 | 21 files | −36.7 dB | −11.4 dB |
| 12 | 1.84 | 0.858 | 3.33 | 3.19 | 0 | −8.9 | −8.1 |
| 14 | 1.88 | 0.857 | 3.34 | 3.26 | 0 | −10.3 | −8.8 |
| **16**, the default | **1.91** | **0.855** | **3.33** | **3.32** | **0** | **−11.7** | **−9.4** |
| 18 | 1.93 | 0.853 | 3.32 | 3.37 | 1 | −13.0 | −9.9 |
| 20 (before 0.3) | 1.93 | 0.851 | 3.31 | 3.42 | 3 | −14.4 | −10.2 |

The limit trades the noise removed (BAK, and PESQ, which rewards it) against the voice: SIG holds from 14 to 16 dB and falls past it (16 against 20, paired: SIG +0.022 ± 0.011, STOI +0.004 ± 0.001, PESQ −0.018 ± 0.008, BAK −0.10 ± 0.01), and 16 is the most at which no utterance loses more than 0.2 STOI; at 20 three do. The limit only bounds what the model takes from the voice; it never stops taking it: speech with nothing to remove still loses 12 dB at its word ends at 16, 9 at 12.

**Clean speech.** A repair with nothing to repair should leave the sound alone. The clean takes of the same utterances as input, against themselves, 10 ms frames: word ends are the last 100 ms before each pause of 100 ms or more, quiet syllables the other frames 20 to 35 dB under the active speech level (ITU-T P.56's, here the mean power of the frames within 35 dB of the 99th-percentile one).

| | training speakers (504): word ends | quiet syllables | SI-SDR vs input | test set (824): word ends | quiet syllables | SI-SDR vs input |
|---|---|---|---|---|---|---|
| RNNoise, unlimited | −36.7 dB | −11.4 dB | 15.0 dB | −30.8 dB | −10.2 dB | 15.5 dB |
| RNNoise, `limit: 20` (before 0.3) | −14.4 | −10.2 | 16.0 | −12.9 | −9.5 | 16.4 |
| **RNNoise, `limit: 16`**, the default | **−11.7** | **−9.4** | **16.6** | **−10.5** | **−8.8** | **17.0** |
| DeepFilterNet3, `denoise()`, unlimited | −0.9 | −0.7 | 39.0 | −0.9 | −0.8 | 39.2 |
| **DeepFilterNet3, `denoise()`, `limit: 18`**, the default | **−0.7** | **−0.6** | **40.2** | **−0.8** | **−0.7** | **40.4** |

RNNoise gates: its gains close on the ends of words and on quiet syllables even with no noise at all, which a listener hears as clipped word ends; the port is bit-exact to upstream's C, so this is the model. DeepFilterNet3 leaves clean speech within a decibel.

**Band-limited input.** The noisy utterances at 16 and at 44.1 kHz (`resample_poly` from 48 kHz), and low-passed at 16 kHz as an MP3 encoder cuts them (a 1023-tap Kaiser FIR, kept at 48 kHz); `denoise()` resamples them to 48 kHz and back; scored at 16 kHz as above. 0.2 heard the empty bands as they are; 0.3 hears the bands wholly above the edge over the floor (DeepFilterNet3, step 2). PESQ / STOI:

| DeepFilterNet3 | training speakers (504): 0.2, unlimited | 0.3, unlimited | 0.2, `limit: 18` | **0.3, `limit: 18`** | test set (824): 0.2, `limit: 18` | **0.3, `limit: 18`** |
|---|---|---|---|---|---|---|
| 48 kHz, the full band | 2.57 / 0.881 | the same, bit for bit | 2.24 / 0.876 | the same | 2.90 / 0.942 | the same |
| 44.1 kHz | 2.55 / 0.880 | the same, bit for bit | 2.23 / 0.876 | the same | 2.89 / 0.940 | the same |
| low-passed at 16 kHz | 2.42 / 0.876 | 2.51 / 0.881 | 2.19 / 0.874 | **2.22 / 0.876** | 2.76 / 0.938 | **2.86 / 0.941** |
| 16 kHz | 2.26 / 0.863 | 2.42 / 0.878 | 2.13 / 0.866 | **2.17 / 0.874** | 2.72 / 0.934 | **2.83 / 0.940** |

On the test set, paired, at the default 18 dB: PESQ +0.11 ± 0.01 at 16 kHz and +0.10 ± 0.01 low-passed, STOI +0.006 and +0.003, DNSMOS within 0.02 (BAK −0.011 at 16 kHz, +0.022 low-passed). Unlimited the gains are larger: 16 kHz from 2.68 to 2.99 PESQ (+0.31 ± 0.02; the full band scores 3.15) and 0.924 to 0.940 STOI, low-passed from 2.90 to 3.07; at 16 kHz the model then leaves a little more noise (DNSMOS BAK −0.06, OVRL −0.04) while keeping more of the voice. A 44.1 kHz input is 0.2's output bit for bit: no band lies wholly above 22.05 kHz (a floor over the top band's empty part, tried first, cost 0.012 ± 0.010 PESQ there on the training speakers). The narrations below, eight of them with empty top bands, move by 0.008 DNSMOS at most.

**Real rooms.** Ten amateur narrations from Spoken Wikipedia (volunteers at home, 2007 to 2026, the first 60 s of each; Ogg Vorbis at 44.1 and 48 kHz, decoded and resampled to 48 kHz by ffmpeg), scored without a reference: DNSMOS P.835 as above; the noise floor as the RMS of the quietest 500 ms (ACX Check's measure; one file with pauses cut to digital silence left out); speech level change over the loudest half of 50 ms frames.

| System | SIG | BAK | OVRL | noise floor, dBFS (median) | speech level |
|---|---|---|---|---|---|
| raw | 3.45 | 4.04 | 3.17 | −83 to −54 (−62) | |
| `omlsa()` | 3.46 | 4.09 | 3.19 | −98 to −69 (−75) | 0.0 dB |
| "Enhance speech" denoising | 3.45 | 4.07 | 3.17 | −101 to −71 (−85) | −0.2 dB |
| RNNoise, unlimited | 3.42 | 4.14 | 3.19 | −127 to −106 (−112) | −0.5 dB |
| RNNoise, `limit: 16`, the default | 3.47 | 4.14 | 3.23 | −98 to −70 (−78) | −0.4 dB |
| DeepFilterNet3, upstream's settings, unlimited | 3.47 | 4.12 | 3.21 | −152 to −61 (−83) | −0.4 dB |
| DeepFilterNet3, `denoise()`, unlimited | 3.47 | 4.11 | 3.21 | −165 to −60 (−78) | −0.4 dB |
| DeepFilterNet3, `denoise()` 0.2, `limit: 18` | 3.47 | 4.11 | 3.21 | −93 to −59 (−73) | −0.4 dB |
| DeepFilterNet3, `denoise()`, `limit: 18`, 0.4's default (0.5's `floor: 40` everywhere moves DNSMOS by 0.001; the current default not measured here) | 3.47 | 4.11 | 3.21 | −93 to −59 (−73) | −0.4 dB |
| DeepFilterNet3, `limit: 12` | 3.47 | 4.10 | 3.20 | −91 to −58 (−68) | −0.3 dB |

These rooms are mostly quiet (raw OVRL 3.17), so the gains are small, and 0.5's floor changes them by 0.001 DNSMOS (it takes the noisiest room's floor from −59.1 to −62.7 dBFS, under ACX's −60; bench/speech.py's scoring, Measured). OM-LSA keeps SIG and raises BAK a little less than the models (0.3.11's lowered SIG by 0.20); both models raise BAK, and at their default limits keep SIG. Unlimited, both can take pauses to dead silence (RNNoise's median floor is −112 dBFS, DeepFilterNet3's lowest −165), which ACX Check flags; the default limits keep the floors between −98 and −59 dBFS at the same OVRL or better, room tone left in. Eight of the narrations end between 16.7 and 19.5 kHz and hear 0.3's floor in their top bands (one ends at 21.1 kHz, inside the top band, one runs to 24): DNSMOS moves by 0.008 at most on any file. On lena (a film scene with music under the voice) RNNoise lowers OVRL from 2.76 to 2.47 unlimited and keeps 2.78 at 16 dB; DeepFilterNet3 raises it to 2.83 upstream, 2.89 at the default limit.

**Singing.** VocalSet (Wilkins et al., ISMIR 2018, CC BY 4.0): 40 long tones (straight, forte, pianissimo, messa di voce; 20 singers) and 20 sung phrases (straight and vibrato), alone and in living-room noise (DEMAND, from the VoiceBank+DEMAND test set) at 15 dB SNR; level change of the voice over the frames within 20 dB of the loudest. At the default 18 dB, against 0.1's processing at the same limit (heard as it is, no voice guard): held notes −1.9 dB clean and −3.4 dB in noise, 14% of frames more than 6 dB down, against −14.2 and −16.2 dB, 96% (at 10 dB SNR −4.8 against −15.6); sung phrases −0.9 and −2.0 dB against −10.2 and −14.9; the same singers' spoken phrases −0.3 and −0.8 dB against −1.7 and −3.7 (they are quiet recordings: the −20 dBFS hearing helps them). Upstream's unlimited output takes the held notes 38.7 and 37.4 dB down. The noise in the pauses drops as before (−14.1 dB against −14.3), and a 100 or 120 Hz buzz 20 dB under the voice drops in them as without the guard (−17.2 dB), in quiet and in cafe noise: it is the background, not over it.

**Music.** Both models enhance speech, and to them everything else is noise: before 0.4 music lost at every band, and the limit only capped the loss (DeepFilterNet3, songs: −13 to −17 dB per band, median; unlimited, single bands of single songs lost up to 54 dB, RNNoise's up to 68). From 0.4 the music guard passes it through ([Music](#music), with the before and after per band). The first 60 s of four pieces (`vibeace` jazz, `brahms` string orchestra, `nutcracker` celesta and strings, `trumpet` solo), five chords of three VocalSet singers each, and the 50 MUSDB18 test previews (Rafii et al. 2017, 7 s of a song each, most with vocals), at 44.1 kHz, are the sets of that table.

The default's row and `everywhere`'s: `node scripts/floor.mjs vb OUT 1`, `render OUT`, then `python3 scripts/floor.py OUT stoi full` (accuracy.py's own scoring) and `node scripts/floor.mjs table OUT everywhere`. To rerun the rest: `python scripts/accuracy.py prepare` writes the band-limited inputs and decodes the MUSDB18 previews; `python scripts/accuracy.py guard-sets` writes the music guard's labelled sets and `node scripts/accuracy.mjs guard train|test` measures it (systems `rnnoise-guard` and `dfn3-guard` are 0.4's defaults, the other rows take `music: 'enhance'`); `node scripts/accuracy.mjs SET SYSTEMS` (SET: `vbdemand`, `vbtrain`, `vbclean`, `vbtrain-clean`, either noisy set `@16000`, `@44100` or `-lp16k`, `music`) and `node scripts/accuracy.mjs rooms` write the outputs (the data sources and checksums are in its header); `python scripts/accuracy.py SET SYSTEMS`, `limits SET rnnoise inf,12,14,15,16,18,20`, `clean SET SYSTEM [LIMITS]`, `music SYSTEMS`, `presence SYSTEM` and `rooms` score them; `scripts/deepfilter-reference.py --vbdemand` writes the Python row; the classical rows come from `@audio/denoise`'s `node scripts/speech.mjs vbdemand|rooms SYSTEMS` and `python scripts/speech.py score vbdemand|rooms SYSTEMS`. Per-file scores stay in `~/.cache/audiojs/data/` (`vbdemand/scores/neural-*.csv`, `vbdemand-train/scores/neural-*.csv`, `spoken/scores.json`); the scripts regenerate outputs, skipping any already there.

## Measured: against iZotope RX 12 Dialogue Isolate

October 2026: RX 12 Advanced's Dialogue Isolate plug-in (VST3 hosted through Pedalboard 0.9, `audio`'s `bench/rx/host.py`) and `audio`'s `deepfilter()` on the same buffers. RX at its defaults with the noise off (noise gain −∞; dialogue and reverb gain 0 dB, sensitivity 5, the four band amounts 100%), and tuned: the reverb off too. Ours at its defaults, 0.4 (`floor: 0` and no speech prior: `audio`'s op bit for bit) and 0.5. Every setting, RX's and ours, was chosen on the tune split (VoiceBank's training speakers, MUSDB18's training songs, the even-numbered MIT rooms, DEMAND's first 150 s, the spoken-train narrations): RX's sensitivity 0, 2.5, 5, 7.5 and 10 and its reverb off tried, the reverb off with sensitivity 5 the best mean PESQ (1.88) and OVRL (2.71) over the speech conditions; our floor 0, 35, 40 and 45 tried (mean PESQ 1.75, 1.96, 1.98, 1.98; VoiceBank's training speakers 2.24, 2.38, 2.43, 2.46 at SIG 3.41, 3.38, 3.36, 3.35; unlimited 2.57, SIG 3.34), 40 kept: SIG is the voice, and 40 dB under it is ACX's −60 dBFS floor for a voice at −20. Then the test split once.

**Harder mixtures**, `audio`'s `bench/rx/isolate.mjs`, test split: VoiceBank's clean test takes (every 8th, 103 per row) at −26 dBFS under DEMAND noises VoiceBank+DEMAND does not use (washing machine, field, park, river, hallway) at −5, 0 and 5 dB SNR (active levels), six Spoken Wikipedia narrators at once, MUSDB18 test accompaniments (drums, bass, other), and MIT IR Survey rooms (odd-numbered, a tail to hear) with the noise at 5 dB, the reference there the direct sound and 50 ms. Scored at 16 kHz: PESQ / DNSMOS OVRL (at the reference's loudness), the last column paired against RX tuned with 95% intervals.

| | RX default | RX tuned | 0.4 | **0.5** | 0.5 − RX tuned, PESQ / OVRL |
|---|---|---|---|---|---|
| noise, −5 dB | 2.00 / 2.79 | 2.00 / 2.78 | 1.73 / 2.77 | **2.11 / 2.91** | +0.11 ± 0.05 / +0.12 ± 0.03 |
| noise, 0 dB | 2.28 / 2.95 | 2.27 / 2.95 | 2.07 / 2.96 | **2.43 / 3.04** | +0.16 ± 0.05 / +0.09 ± 0.03 |
| noise, 5 dB | 2.56 / 3.07 | 2.56 / 3.07 | 2.44 / 3.05 | **2.70 / 3.10** | +0.15 ± 0.07 / +0.03 ± 0.03 |
| babble, 0 dB | 1.18 / 2.11 | 1.18 / 2.16 | 1.24 / 2.31 | **1.31 / 2.41** | +0.12 ± 0.03 / +0.25 ± 0.06 |
| babble, 5 dB | 1.43 / 2.47 | 1.44 / 2.51 | 1.58 / 2.67 | **1.70 / 2.73** | +0.26 ± 0.04 / +0.22 ± 0.05 |
| music bed, 0 dB | 1.65 / 2.74 | 1.65 / 2.74 | 1.26 / 2.10 | 1.63 / 2.33 | −0.02 ± 0.09 / −0.41 ± 0.17 |
| music bed, 5 dB | 1.96 / 2.90 | 1.96 / 2.90 | 1.82 / 2.94 | **2.26 / 3.03** | +0.30 ± 0.05 / +0.13 ± 0.05 |
| room, noise 5 dB | 1.66 / 2.21 | 1.80 / 2.51 | 1.62 / 2.28 | 1.77 / 2.42 | −0.03 ± 0.05 / −0.09 ± 0.07 |
| mean | 1.84 / 2.66 | 1.86 / 2.70 | 1.72 / 2.63 | **1.99 / 2.75** | |

SI-SDR follows PESQ but under steady noise, where RX keeps the waveform closer (13.2, 15.8, 17.8 dB against 12.7, 15.3, 17.2); in babble ours is the closer (3.5 and 8.9 against 1.7 and 8.4). DNSMOS SIG, the voice: ours 3.21 to 3.39 under noise, RX's 3.07 to 3.33.

**VoiceBank+DEMAND** test set (824, `audio`'s `bench/speech.mjs` and `bench/speech.py`, scored as in Accuracy): RX default PESQ 2.73, STOI 0.929, SI-SDR 18.9 dB, SIG 3.30, BAK 3.82, OVRL 2.94; tuned 2.72, 0.929, 18.8, 3.30, 3.83, 2.94; ours 0.4 2.90, 0.942, 17.9, 3.48, 3.84, 3.10; **0.5 3.05, 0.944, 18.9, 3.45, 4.01, 3.15**. **The ten narrations** (Accuracy, Real rooms): raw OVRL 3.18; RX 3.23 (tuned 3.23), every room's floor (its quietest 500 ms) taken under −79.6 dBFS (tuned −88.2), room tone gone; ours 3.23 at 0.4 and 0.5, room tone kept, the noisiest room's floor at −62.7 dBFS (0.4: −59.1).

**Nothing to isolate.** The clean test takes: RX raises them 1.4 dB and leaves PESQ 3.98 and SI-SDR 23.6 dB against the take (tuned the same); ours −0.1 dB, PESQ 4.60, SI-SDR 41.2 dB (OVRL 3.24 against RX's 3.30: RX also takes the studio's own faint noise). Music alone, 24 MUSDB18 test previews, songs and accompaniments in turn: RX takes it 37.6 dB down (every band 33 to 57 dB), its median SI-SDR to the input −16.8 dB; ours passes all 24 bit for bit.

What 0.5 changed, and where RX still wins:

- **The floor.** 18 dB of attenuation leaves noise as loud as the voice 18 dB under it, and RX takes it all: before 0.5 RX won every noise row. Unlimited, the model beat RX already (tune split, −5 dB: PESQ 2.12 against 1.98); the floor keeps that where the noise stands near the voice and the 18 dB limit's room tone where it is quiet. +0.38, +0.36, +0.26 PESQ at −5, 0, 5 dB over 0.4. That is the floor everywhere, `everywhere: true` now; the default keeps the limit under the voice and was not measured against RX ([DeepFilterNet3's floor](#deepfilternet3s-floor)).
- **The guard's prior.** Speech under a bed as loud as itself reached ina as music, so 0.4 passed half of those takes untouched (PESQ 1.26). With speech favoured by 1 nat a patch, as the stream already decided, 37% of their frames pass (Music): PESQ 1.63 against RX's 1.65, OVRL 2.33 against 2.74. What still passes is a voice ina cannot tell from a singer at that level: the model keeps the voice of a song as it keeps this one, 4 to 7 dB under the mix, and ina hears speech in both, so passing less would gut songs (tune split: a voice under a bed at its level kept −3.5 to −5.9 dB and heard as speech at 0.81 or more; the training songs, at most −4.6 dB and 1.00). `music: 'enhance'` enhances it all: PESQ 1.87, OVRL 2.91, SI-SDR 11.0 dB, ahead of RX on all three, and music alone then loses 28.5 dB (RX 37.6).
- **Rooms.** RX's reverb gain takes the late reverberation (SI-SDR 7.2 dB to the direct sound and 50 ms, against ours 5.7; at its default, reverb kept, 5.2). DeepFilterNet3 trained with reverberant speech in 10% of its mixtures (its `config.ini`, `p_reverb`), and takes part of it. `audio`'s `deepfilter()` now runs it again where its output still carries a room (`@audio/denoise-dereverb` 0.6's checks, asked of the output and of the take), on the take with the room's linear prediction taken off (that package's WPE alone): 81 of the 103 room takes, PESQ 1.81, OVRL 2.44, SI-SDR 6.2 dB (RX tuned 1.80, 2.51, 7.2; paired +0.01 ± 0.05 PESQ, −0.08 ± 0.07 OVRL); the music bed at 0 dB 1.67 / 2.44, 13 takes heard again; the other rows, clean speech, music alone, VoiceBank+DEMAND (2 of 824) and the narrations (2 of 10) as above. Chosen on the tune split, reverb5: WPE before the model PESQ 1.88 → 1.96, SI-SDR 7.3 → 7.9; `dereverb()` whole after the model 1.89, before it 1.87 (OVRL −0.09); MossFormer2_SE_48K (ClearerVoice-Studio, Apache-2.0, trained with reverberant speech in 30% of its mixtures; its 221 MB PyTorch checkpoint) 1.80, OVRL 2.35, SI-SDR 6.3. Still behind RX: OVRL and SI-SDR in rooms, the late tail the model and the linear prediction leave.

To rerun: in `audio`, `node bench/rx/isolate.mjs test all rx-di rx-di-tuned dfn` (renders kept in `~/.cache/audiojs/data/rx/isolate/`), `node bench/speech.mjs test rx-di,rx-di-tuned,dfn-18` then `python bench/speech.py test rx-di,rx-di-tuned,dfn-18`; tune with `tune` in place of `test`.

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

A 128-sample render quantum lasts 2.67 ms and a frame is due every 3.75 quanta: the slowest SIMD frame measured (0.94 ms) fits a quantum 2.8 times over, the mean 8 times; the JS path's slowest (up to 4.0 ms) does not, and relies on device buffers of 256 samples or more, the browsers' usual size. The live underruns came with the machine oversubscribed, the control underran too; an idle machine is untested. Memory: 3.5 MB of weights (plus a 2.8 MB copy in the kernel's memory), about 70 KB of state per channel (the contract atom's, with the music guard and the queues: 0.5 to 1.0 MB, by block size and rate); the Node evaluation processes peaked at 83 MB RSS. The contract atom's first processor in a process takes 0.63 to 0.73 s of CPU to make over all threads, 0.41 to 0.47 s of it on the calling thread, its warm-up (In `audio`); later ones take none of it.

**The contract atom's allocation in a process's first stream**: 20 s of stereo speech, silence, a chord, noise and clipped speech, `limit` to no limit (its share 0), fractions and back, `music` switched, then 10 s of a second instance; each half second what V8's young and old generations and the typed arrays' stores take between two forced minor collections, as @audio/compile's validator counts it, under the validator's flags and under Node's own; at 16, 32, 48 and 88.2 kHz in blocks of 64, 1, 128 and 2048 frames, once each (on Node 25 at 48 kHz three times, on Node 20 under its own flags five). Beside it, a processor that only copies its input, through the same harness, gives the measure's own floor:

| Node, flags | 0 to 0.5 s | 0.5 to 1 s | after 1 s | a second instance | the copying processor, after 1 s |
|---|---|---|---|---|---|
| 25, the validator's | 121 to 123 KB | 1.86 to 1.87 MB | up to 0.8 KB | up to 0.5 KB | up to 0.4 KB |
| 25, Node's | 87 to 141 KB | 69 KB to 7.5 MB | up to 1.4 KB | up to 1.8 KB | up to 0.5 KB |
| 22, either | 0.5 to 11.5 KB (one 319 KB) | 224 bytes | up to 4.3 KB, in at most 4 half seconds of 38; 8.6 to 9.3 KB at the first `music` switch in calls of 1 frame | up to 6.4 KB | up to 1.7 KB; 2.2 KB in a second instance |
| 20, the validator's | 0.2 to 1.2 KB | 104 bytes | up to 1.4 KB | up to 0.8 KB (one first half second of 250 KB, 32 kHz in calls of 1) | up to 0.2 KB |
| 20, Node's | 0.2 to 61 KB | 0.1 to 17 KB | in 4 of 8 runs (2 of 5 at 48 kHz in 128, and 16 kHz in 64, 32 kHz in 1) 12 to 17 KB every half second, through all 20 s, single ones up to 240 KB; in the other 4, one or two half seconds of 34 to 183 KB | in the same 4, 12 to 15 KB every half second (one 269 KB); else up to 1.0 KB | up to 0.1 KB (one half second of 33 KB) |

On Node 25 the first second is V8 compiling the music guard's decision, which a stream makes once a patch. What allocates on Node 22 and 20 is not known.

**DeepFilterNet3**, onnxruntime-node 1.30, 4 intra-op threads: real-time factor 0.047 in CPU on a 180 s file (8.4 s), 0.087 over the 824 short test utterances (2072 s of audio in 180 s; per-run overhead); Python DeepFilterNet (PyTorch 2.1, 4 threads) 0.067 on the same utterances. Peak RSS 436 MB over the test set, 563 MB for 180 s in 10 s chunks. The voice guard's pitch analysis adds 0.16 s of wall time per minute of audio to the model's 0.6 (one channel at 48 kHz, 4 threads); the band edge, 0.06 s of CPU. The browser path (onnxruntime-web, wasm and WebGPU) is untested here.

**Music guard**: 1.1 s of CPU per minute of audio per channel (real-time factor 0.019; WebAssembly SIMD, Node 25, the machine as above), 11 s on the JS path. RNNoise through `denoise()` goes from 3.0 to 4.2 s of CPU per minute; DeepFilterNet3 skips the model where a whole input is music. The guard's state: 0.3 MB per channel, the network 3.1 MB once unpacked to float32.

## Licenses

Audited 2026-09-27 against what each project states, DeepFilterNet3 and inaSpeechSegmenter again 2026-11-17; a summary, not legal advice.

| Model | Code | Weights | Weight files and their terms | Here |
|---|---|---|---|---|
| **RNNoise**, model 0a8755f8 (January 2025) | BSD-3-Clause (`COPYING`: Valin, Amazon, Mozilla, Xiph.Org, Borgerding) | BSD-3-Clause | `rnnoise_data-0a8755f8….tar.gz` (media.xiph.org, sha256 pinned in upstream's `model_version`): `src/rnnoise_data.c` (default), `src/rnnoise_data_little.c`, two `.pth` checkpoints. No header of their own; upstream's `autogen.sh` downloads them into `src/` and compiles them into the BSD library, and the README calls them "the models distributed with RNNoise". Trained on the corpora in `datasets.txt`: OpenSLR 30–86 (CC BY-SA 4.0) and Hi-Fi TTS (OpenSLR 109, CC BY 4.0) for speech, Xiph's noise collections (donated noise "freely available", terms in that archive, not checked) | bundled: `rnnoise.bin`, the default model through upstream's `dump_weights_blob`; the little model (1.55 MB) loads through `weights` |
| **DeepFilterNet3**, d375b2d | MIT OR Apache-2.0 | MIT OR Apache-2.0: "All code and model weights in this repository are dual-licensed" (upstream's README since [7b14b89](https://github.com/Rikorose/DeepFilterNet/commit/7b14b895fe7fc42cab72b4a4c1e49c982bd6404b), 2026-11-06) | `models/DeepFilterNet3_onnx.tar.gz`, `DeepFilterNet3.zip`, `DeepFilterNet3_ll_onnx.tar.gz`, in the same repository, the same files at d375b2d and after it. Until 2026-11-06 the README licensed "All code in this repository"; issues asked whether that covered them ([#697](https://github.com/Rikorose/DeepFilterNet/issues/697), [#700](https://github.com/Rikorose/DeepFilterNet/issues/700), [#709](https://github.com/Rikorose/DeepFilterNet/issues/709), [#712](https://github.com/Rikorose/DeepFilterNet/issues/712), opened July to September 2026), and the commit that named the weights closed the first three (#712 is still open). The DeepFilterNet3 paper's abstract says "the framework as well as pretrained weights have been published under an open source license", naming none; it trained "on the full multi-lingual DNS4 dataset", PTDB and VCTK oversampled, whose sets the DNS Challenge provides "under the original terms that Microsoft received such datasets", each its own. Third parties republish them as MIT (Intel/deepfilternet-openvino on Hugging Face) or Apache-2.0 | fetched from upstream at the pinned commit, cached, never bundled or re-hosted |
| **inaSpeechSegmenter**, speech/music/noise CNN (`smn`) | MIT (`LICENSE`, © 2018 Ina) | MIT: the repository's license; the weight files are its release assets (`models`), no terms of their own. Training data: not stated (its README, the `models` release, the ICASSP 2018 and MMEDIA 2017 papers searched) | `keras_speech_music_noise_cnn.hdf5` (sha256 `f04b5e3c…`) | bundled: `guard.bin`, its weights folded and in float16 (`scripts/guard.py`); the feature recipe follows ina's `sidekit_mfcc.py` (from SIDEKIT, LGPL), written anew from its definitions |
| YAMNet (Google) | Apache-2.0 | Apache-2.0 | TF Hub `yamnet/1` | not used: compared, passes more speech over music beds ([Music](#music)) |
| Silero VAD | MIT | MIT | `silero_vad.onnx` | not used: speech against everything else, noise included |
| DeepFilterNet 1 and 2 | same | same as DeepFilterNet3 | `models/DeepFilterNet{,2}*.{zip,tar.gz}` | not used; libDF deprecates DeepFilterNet2 |
| GTCRN (Rong et al., ICASSP 2024) | MIT | MIT (checkpoints in the repository) | 48.2K parameters, trained on DNS3 or VCTK-DEMAND | not used: 16 kHz only, narration would lose everything above 8 kHz |
| Demucs speech denoiser (facebookresearch/denoiser) | CC BY-NC 4.0 | CC BY-NC 4.0 | | excluded: non-commercial; archived |
| DNSMOS P.835 (Microsoft) | CC BY 4.0 (microsoft/DNS-Challenge) | CC BY 4.0 | `DNSMOS/DNSMOS/sig_bak_ovr.onnx` at 82f1b17e77 | evaluation only |
| VoiceBank+DEMAND (Valentini-Botinhao 2017) | | CC BY 4.0 | 824 test utterances, DataShare checksums verified | evaluation only |
| Spoken Wikipedia narrations | | CC BY-SA 3.0/4.0, CC BY 4.0, CC0 | ten files from Wikimedia Commons | evaluation only, not redistributed |

In the browser, [shiguredo/rnnoise-wasm](https://github.com/shiguredo/rnnoise-wasm) (Apache-2.0) builds the same RNNoise commit with emscripten, and [sapphi-red/web-noise-suppressor](https://github.com/sapphi-red/web-noise-suppressor) (MIT) wraps its 2022 build next to Speex and GTCRN. This package needs no WebAssembly (it takes 419 bytes of it for speed where it can), is checked bit for bit against the C, and adds DeepFilterNet3.

The package is BSD-3-Clause (the RNNoise translation and weights); `deepfilter.js` carries DeepFilterNet's MIT notice, `guard.bin` inaSpeechSegmenter's ([NOTICE](./NOTICE)).

## Reference

J.-M. Valin, "A Hybrid DSP/Deep Learning Approach to Real-Time Full-Band Speech Enhancement", MMSP 2018, [arXiv:1709.08243](https://arxiv.org/abs/1709.08243). · H. Schröter, T. Rosenkranz, A. N. Escalante-B., A. Maier, "DeepFilterNet: Perceptually Motivated Real-Time Speech Enhancement", Interspeech 2023, [arXiv:2305.08227](https://arxiv.org/abs/2305.08227). · C. Valentini-Botinhao, "Noisy speech database for training speech enhancement algorithms and TTS models", University of Edinburgh, 2017, [doi:10.7488/ds/2117](https://doi.org/10.7488/ds/2117). · ITU-T P.862.2 (2007), wideband PESQ. · C. H. Taal, R. C. Hendriks, R. Heusdens, J. Jensen, "An Algorithm for Intelligibility Prediction of Time-Frequency Weighted Noisy Speech", IEEE TASLP 19(7), 2011. · J. Le Roux, S. Wisdom, H. Erdogan, J. R. Hershey, "SDR: Half-baked or Well Done?", ICASSP 2019. · C. K. A. Reddy, V. Gopal, R. Cutler, "DNSMOS P.835: A Non-Intrusive Perceptual Objective Speech Quality Metric to Evaluate Noise Suppressors", ICASSP 2022. · S. A. Figueroa, "When is double rounding innocuous?", SIGNUM Newsletter 30(3), 1995. · [xiph/rnnoise](https://gitlab.xiph.org/xiph/rnnoise) at 70f1d25 · [Rikorose/DeepFilterNet](https://github.com/Rikorose/DeepFilterNet) at d375b2d. · D. Doukhan, J. Carrive, F. Vallet, A. Larcher, S. Meignier, "An Open-Source Speaker Gender Detection Framework for Monitoring Gender Equality", ICASSP 2018 · [ina-foss/inaSpeechSegmenter](https://github.com/ina-foss/inaSpeechSegmenter), models release.

**Use when:** speech with noise a statistical denoiser leaves or smears (moving noise, a busy room); live voice in a browser (RNNoise, 30 ms); files where quality is worth an 8 MB download (DeepFilterNet3); programmes that mix speech and music (the music passes through untouched).<br>
**Not for:** denoising music itself, a noisy song or concert (music passes untouched; with `music: 'enhance'` both models take 8 to 16 dB from every band); singing alone (about half of it passes, the rest is enhanced); clean speech with nothing to remove through RNNoise (it gates word ends and quiet syllables, 9 to 12 dB at its default limit); installs that must stay model-free ([`@audio/denoise`](https://github.com/audiojs/denoise)); reverb as such (`dereverb`); live DeepFilterNet3 (upstream's graphs take no state).

---

Part of the [@audio/neural](https://github.com/audiojs/neural) lane.

BSD-3-Clause © 2026 audiojs (port), Jean-Marc Valin, Xiph.Org Foundation, Mozilla, Amazon (RNNoise); DeepFilterNet parts MIT © 2021 Hendrik Schröter
