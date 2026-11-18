# @audio/neural-quality

> How listeners would rate a recording of speech, with no clean reference to compare it to: Microsoft's DNSMOS P.835, computed in JS as its own script computes it.

DNSMOS P.835 (Reddy, Gopal, Cutler, ICASSP 2022) is a network trained on listeners' ITU-T P.835 ratings of noisy and
noise-suppressed speech. It gives three numbers on P.835's scales, 1 (bad) to 5 (excellent):

- **SIG**, the speech: how natural and undistorted the voice itself sounds.
- **BAK**, the background: how little and how unobtrusive the noise behind it is.
- **OVRL**, the whole: the rating a listener gives the recording overall.

This package runs Microsoft's network (`sig_bak_ovr.onnx`, microsoft/DNS-Challenge at `82f1b17e77`) in Node and in a
browser through `@audio/neural-runtime`, with the method of Microsoft's `dnsmos_local.py` at the same commit around it:
the same resampling, windows, polynomials and averages. On 72 files at 8 to 48 kHz it gives the script's numbers to
1.8e-15 (Verification).

```
npm install @audio/neural-quality onnxruntime-node   # onnxruntime-web in the browser
```

```js
import dnsmos, { load } from '@audio/neural-quality'

let { sig, bak, ovrl } = await dnsmos(samples, { sampleRate: 44100 })     // lena: { sig: 3.58, bak: 3.07, ovrl: 2.76, segments }
let model = await load()                                                   // held across calls
let before = await dnsmos([left, right], { sampleRate: 48000, model })     // channels mixed to their mean
let after = await dnsmos(edited, { sampleRate: 48000, model, windows: 'all' })
model.free()
```

`segments` holds each window the mean is taken over: `{ time, sig, bak, ovrl, raw }`, `time` its start in seconds,
`raw` the network's three outputs before the polynomials. The model file is fetched once from GitHub at the pinned
commit (1.1 MB), refused unless of its SHA-256, and in Node kept at `~/.cache/audiojs/neural/dnsmos/sig_bak_ovr.onnx`,
the path `audio`'s `bench/speech.py` reads it from too; `opts.weights` reads another copy (bytes, a URL, a path in Node).

## What it is not

- **Not a judge of music or other sound.** It is trained on speech in noise and reads anything else as bad speech:
  piano, an orchestra, chiptune, a jazz combo, rain and birdsong all score OVRL 1.07 to 1.10 here, a 1 kHz tone 1.19,
  digital silence 1.84 (with BAK 3.47).
- **Deaf above 8 kHz.** It hears 16 kHz audio. Speech low-passed at 4 kHz scores OVRL 3.43 against the full band's
  3.45, and the same speech at 8 kHz 3.44: it barely tells a narrowband recording from a wideband one.
- **Not a codec meter.** Its training material is noise suppressors' output (the paper's section 3), not codecs; no
  claim is made for MP3 or Opus artefacts, and none was measured here.
- **Level-dependent on purpose.** The paper avoids "any kind of feature normalization" because "Humans tend to give
  lower ratings to clips with low amplitudes" (p. 3). The same speech 50 dB down scores OVRL 2.03 against 3.45:
  compare an edit at the level it will be heard at, or at the input's loudness, as `audio`'s bench does. A gain of
  6 dB alone moves BAK by 0.3 to 0.45 on noisy speech (four VoiceBank takes end to end: 2.57 at -6 dB, 2.27, 1.82 at
  +6 dB): an edit that changes the level is scored for that too.
- **Unsteady on a clip shorter than its window.** A clip under 9.01 s is repeated to fill the window, so where it
  ends changes what the network hears: 50 ms of silence added after a 1.6 s take moves SIG by 0.69, and 5 ms fades
  on a 4 s clip by 0.16, though neither is heard. Compare a short edit with its input at the same length.
- **Rate-dependent at 8 kHz.** The same speech in white noise scores BAK 0.60 higher at 8 kHz than at 16 kHz: half
  the noise band is missing (clean speech moves by 0.013). From 22.05 to 48 kHz it moves by 0.013 at most. Compare
  an edit at the rate of its input.
- **Shaky on one clip.** The paper's correlation with listeners' P.835 ratings (arXiv:2110.01763v4, p. 4, Table 3, on
  850 unseen clips from 17 noise suppressors): averaged per model, PCC 0.94 SIG, 0.98 BAK, 0.98 OVRL (SRCC 0.95, 0.99,
  0.98); per clip, PCC 0.71, 0.83, 0.82 (SRCC 0.72, 0.82, 0.81), measured on two of the models with 30 ratings a clip.
  It ranks systems over many clips far better than it rates one recording. The weights at `82f1b17e77` came in a
  commit titled "Add support for personalized and improved models" (2022-05-24), after the paper's last version
  (4 February 2022); the repository states no correlation for them.
- **Not yet checked against this project's own listening.** No verdict of a listener on these recordings has been
  compared with its numbers.
- **Blind to part of a longer clip, as upstream computes it.** Below.

## The windows upstream drops

`dnsmos_local.py` cuts window `i` as `audio[int(i*fs) : int((i+INPUT_LENGTH)*fs)]` and skips it when shorter than
144160 samples (lines 83-85). `(i + 9.01) * 16000` is a float product, and for `i` = 7 to 23 and 119 to 122 (and in
longer runs past 4.5 hours) it truncates one sample short, so those windows are skipped. A clip from 17 s to under 34 s
is scored on its first 15.01 s alone, and in any longer clip the stretch from 15.01 to 24 s is never heard. Noise
added over 16 to 30 s of a 30 s clip leaves its scores unchanged to the last digit (Verification). torchmetrics 1.9.0
copies the same arithmetic (`functional/audio/dnsmos.py`, lines 254-265).

This package drops the same windows by default, so its numbers are upstream's. `windows: 'all'` keeps every 1 s hop:
on 60 s narrations OVRL moves by up to 0.06; on the 30 s clip above, from 3.43 to 2.79. Use it where what follows
15 s matters, for an edit there above all; its numbers are then this package's, not upstream's.

## How it works

As `DNSMOS/dnsmos_local.py` at `82f1b17e77` computes it (`ComputeScore.__call__`, lines 61-106):

- **Resampling** (lines 63-67): to 16 kHz by `librosa.resample(aud, input_fs, fs)`. That positional call runs on
  librosa 0.9 and older (0.10 and later refuse it: `resample() takes 1 positional argument but 3 were given`), whose
  default is resampy's `kaiser_best`; then librosa cuts or pads it to ceil(n · 16000 / rate). On the day of the
  commit (2022-05-24) the releases were librosa 0.9.1 and resampy 0.2.2, whose `kaiser_best` is a Kaiser-windowed
  sinc of 64 zero crossings with its rolloff at 0.948 of Nyquist. `resampy.js` translates both and gives their 16 kHz
  signal sample for sample, as float32, on all 41 files at 8, 22.05, 44.1 and 48 kHz below. resampy 0.3.0
  (2022-06-29) replaced the filter (50 zero crossings, rolloff 0.917), so the script run today on librosa 0.9.2 gives
  other numbers at those rates: up to 0.056 BAK (Verification).
- **Channels**: the script passes a multichannel file's 2-D array on, and at a rate other than 16 kHz librosa
  resamples it along its last axis, the channels. Above 32 kHz resampy refuses it (`Input signal length=2 is too
  small`); at 32 kHz and below the network refuses a file of 144,160 frames or more (`Invalid rank for input: input_1
  Got: 3 Expected: 2`), and a shorter one `np.append` flattens into one signal of its interleaved values, which the
  script scores without an error (a narration in two channels at 8 to 24 kHz: OVRL 1.56 to 1.84, against 3.18 to 3.36
  for the channels' mean). Here channels are mixed to their mean, as `audio`'s bench and this family's other packages
  do; no path of the script gives that mean.
- **Short clips** (lines 70-71): doubled (`np.append(audio, audio)`) until they hold a window; 0.1 s becomes 12.8 s of
  the same 0.1 s, scored over 3 windows (OVRL 1.65; 2 s of the same speech, 3.09).
- **Windows** (lines 73-85): 9.01 s (144160 samples) every 1 s, `int(floor(len / 16000) − 9.01) + 1` of them, less
  those the float product drops (above).
- **The network**: the waveform in (`input_1`, [N, 144160] float32), the spectrogram computed inside (a 320-sample
  DFT as two convolutions, 161 bins, log power), then 7 convolutions and 3 dense layers: three raw outputs.
- **Polynomials** (`get_polyfit_val`, lines 51-53, non-personalized), on each window's raw outputs s, b, o:
  SIG = −0.08397278 s² + 1.22083953 s + 0.0052439, BAK = −0.13166888 b² + 1.60915514 b − 0.39604546,
  OVRL = −0.06766283 o² + 1.11546468 o + 0.04602535; the scores are their means over the windows (lines 103-105).

Refused, where the script would not give a number: an empty input (its doubling loop never ends), a sample that is NaN
or infinite. The personalized variant (`pDNSMOS`) and P.808 (`model_v8.onnx`, not in the repository at this commit)
are left out.

## Verification

`node test.js`, against `fixtures/reference.json`, which `scripts/reference.py` writes from Microsoft's own script
(imported and called as its `main()` calls it; librosa 0.9.1, resampy 0.2.2, onnxruntime 1.30.0, numpy 1.26.4) on the
waveforms `scripts/inputs.mjs` makes, their SHA-256 checked first: the windows of ten ramps whose sample values name
their positions, `get_polyfit_val` on six inputs, 0.5 s of librosa's 16 kHz signal for lena at 44.1 and 8 kHz, every
window's raw outputs and the scores of six inputs. Every check is exact on this machine (raw outputs to the last bit).

Measured once, by scripts outside the package, on 73 files: VoiceBank+DEMAND clean and noisy takes (48 kHz), Spoken
Wikipedia narrations (60 s, 48 kHz), recordings of 1860 and 1890, speech, music, rain and birdsong from
audiojs.github.io/samples (22.05 to 48 kHz), lena; at 16 kHz the same speech in white and park noise at 20 to −5 dB SNR,
low-passed, telephone-band, 0.1 to 99 s long, 50 dB down, clipped, silence, a tone, white noise; and five recordings
each at 8, 16, 22.05, 44.1 and 48 kHz. Microsoft's script against this package, the model the same file, onnxruntime
1.30.0 on both sides:

| Files | SIG, BAK, OVRL: largest difference | Raw outputs |
|---|---|---|
| 31 at 16 kHz | 1.8e-15 | equal |
| 41 at 8, 22.05, 44.1, 48 kHz | 8.9e-16 | equal |
| 1 with two channels | the script refuses it; here the mean | |

What the other resamplers give, on the same 41 files, against the script as it ran in May 2022:

| Resampler | Largest difference | Over 0.01 |
|---|---|---|
| resampy 0.3 and later (`kaiser_best` redesigned; the script today on librosa 0.9.2) | 0.056 BAK, 0.044 OVRL | 6 of 41 |
| librosa 0.10's default `soxr_hq` (the script with its call made keyword, as torchmetrics 1.9.0 writes it) | 0.022 BAK | 3 of 41 |
| `@audio/resample-sinc` (this family's) | 0.137 BAK, 0.083 OVRL | 7 of 41 |

Each largest difference is speech in white noise 5 dB down, at 22.05 or 44.1 kHz. There the resamplers differ in how
much of the noise between 7.6 and 8 kHz they keep: after resample-sinc, which passes the band to 8 kHz, it stands 12
to 32 dB higher than after resampy 0.2.2's filter (measured on the 48 kHz copy).

## Speed

60 s of a narration at 48 kHz, onnxruntime-node 1.30, Node 25, an Apple M4 Max (14 cores) running other jobs (load
averages 155 to 184, so the wall times are long), the session made first, three runs each:

| Session | Windows | CPU per minute of audio | Wall |
|---|---|---|---|
| one thread (`sessionOptions: { intraOpNumThreads: 1 }`) | 34 | 5.2 to 5.4 s | 21 to 22 s |
| one thread, `windows: 'all'` | 51 | 7.7 to 7.8 s | 22 to 28 s |
| onnxruntime's default | 34 | 13.7 to 14.3 s | 6.3 to 6.7 s |

Of that, resampling to 16 kHz takes 0.5 s from 48 kHz, 0.6 s from 44.1 kHz; the rest is the network, 0.15 s of CPU a
window on one thread. Each second of audio is in nine windows, so the cost grows with the length, about one window a
second. The Python script took 12.3 to 13.5 s of CPU per minute under the same load (onnxruntime's default). Memory:
36 MB for the session; while scoring, a peak 255 MB over Node's own (arena on), 199 MB with `sessionOptions:
{ enableCpuMemArena: false, enableMemPattern: false }`. Untested in a browser.

## The model

| File | Size | SHA-256 | Source |
|---|---|---|---|
| `sig_bak_ovr.onnx` | 1,157,965 bytes | `269fbebdb513aa23cddfbb593542ecc540284a91849ac50516870e1ac78f6edd` | [microsoft/DNS-Challenge](https://github.com/microsoft/DNS-Challenge/tree/82f1b17e7776a43eee395d0f45bae8abb700ad00/DNSMOS) `DNSMOS/DNSMOS/` at `82f1b17e7776a43eee395d0f45bae8abb700ad00` |

keras2onnx 1.8.1, opset 12, graph `SIG_BAK_OVR_mse_cosine`: 184,227 trained values and 103,040 in the DFT kernels.
Fetched, never bundled; used as published.

## API

| Export | |
|---|---|
| `dnsmos(audio, opts)` (default) | → `Promise<{ sig, bak, ovrl, segments }>` |
| `load(opts?)` | → `Promise<{ session, free() }>`: the network, for `opts.model` |
| `polyfit({ sig, bak, ovrl })` | the polynomials alone: raw outputs to the P.835 scale |
| `windows(n, { windows }?)` | → `{ length, starts }`: the doubled length of a 16 kHz signal of `n` samples and each window's first sample |
| `prepare(audio, { sampleRate })` | → `Float32Array`: the channels' mean at 16 kHz, as librosa 0.9 resamples it |
| `SAMPLE_RATE`, `INPUT_LENGTH`, `WINDOW` | 16000, 9.01, 144160 |
| `MODEL`, `REVISION`, `SHA256` | the pinned file's URL, its commit, its SHA-256 |

`audio`: a `Float32Array` (mono), an array of channels, or `{ channelData, sampleRate }`. `opts`: `{ sampleRate,
model, windows = 'upstream', weights, backend, sessionOptions, progress, fetch }`; without `model`, the network is
loaded and freed within the call. `@audio/neural-quality/audio` is the `audio` stat manifest: `{ stat: 'dnsmos',
compute(channels, { sampleRate, ...opts }) }`, returning the same Promise. The stat scores every window (`windows: 'all'`), so a clip of 17 s or more is heard whole; pass `windows: 'upstream'` for the reference script's numbers.

## Licences

Checked 2026-11-18 against what each project states; a summary, not legal advice.

| Part | Licence | Source |
|---|---|---|
| `dnsmos_local.py`'s method, translated in `quality.js` | MIT, Copyright (c) Microsoft Corporation | the repository's README, Legal Notices: "grant you a license to any code in the repository under the MIT License, see the LICENSE-CODE file" |
| `sig_bak_ovr.onnx` | CC BY 4.0 | the same notice: "a license to the Microsoft documentation and other content in this repository under the Creative Commons Attribution 4.0 International Public License, see the LICENSE file". Attribution as its Section 3(a) asks is in [NOTICE](./NOTICE) |
| resampy 0.2.2's `kaiser_best` and librosa 0.9.1's `resample`, translated in `resampy.js` | ISC, Copyright (c) 2016 Brian McFee; ISC, Copyright (c) 2013--2017 the librosa development team | their LICENSE files |

The package is MIT; [NOTICE](./NOTICE) carries the attributions.

## Reference

C. K. A. Reddy, V. Gopal, R. Cutler, "DNSMOS P.835: A Non-Intrusive Perceptual Objective Speech Quality Metric to
Evaluate Noise Suppressors", ICASSP 2022, [arXiv:2110.01763](https://arxiv.org/abs/2110.01763). · ITU-T P.835 (2003),
Subjective test methodology for evaluating speech communication systems that include noise suppression algorithm. · [microsoft/DNS-Challenge](https://github.com/microsoft/DNS-Challenge/tree/82f1b17e7776a43eee395d0f45bae8abb700ad00/DNSMOS)
at `82f1b17e77`. · B. McFee, [resampy](https://github.com/bmcfee/resampy) 0.2.2. · [librosa](https://github.com/librosa/librosa) 0.9.1.

**Use when:** asking whether an edit made a voice cleaner, its background quieter, the whole better, on speech, in
Node or a browser, from a 1.1 MB model, the numbers Microsoft's script gives.<br>
**Not for:** music, band-limited or codec damage, a verdict on one recording without a listener's, or what lies past
15 s in a clip of 17 to 34 s (without `windows: 'all'`).

---

Part of the [@audio/neural](https://github.com/audiojs/neural) lane.

MIT © 2026 audiojs; DNSMOS P.835 © Microsoft Corporation, CC BY 4.0
