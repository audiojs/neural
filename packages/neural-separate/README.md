# @audio/neural-separate

> Source separation (stems): Open-Unmix and Hybrid Transformer Demucs through ONNX, matching the original PyTorch implementations.

The ML upgrade to [`@audio/vocals`](https://github.com/audiojs/vocals)'s classical center-cancel: instead of one M/S trick, a trained model estimates each target (vocals, drums, bass, other), and the pipeline around it (STFT, Wiener refinement, chunking, overlap-add) is ported from the reference implementations and checked against them.

```js
import separate from '@audio/neural-separate'

let { stems, residual } = await separate([left, right], { sampleRate: 44100, model: 'umxhq' })
stems.vocals   // Float32Array[2]
```

## Install

```
npm install @audio/neural-separate onnxruntime-node
```

In the browser `@audio/neural-runtime` runs `onnxruntime-web` instead (`device: 'wasm'` or `'webgpu'`).

## Models

| `model` | Architecture | Files | Weights | SDR, MUSDB18 previews: vocals · drums · bass · other | Real-time factor: Node CPU · browser wasm · WebGPU |
|---|---|---|---|---|---|
| `'umxhq'` | Open-Unmix: bi-LSTM on magnitude spectrograms, multichannel Wiener EM | 4 × 35.6 MB (fp16: 4 × 17.8 MB) | MIT | 6.75 · 6.11 · 5.00 · 3.36 | 0.19 (vocals only: 0.11) · 0.12 · 0.29 |
| `'htdemucs'` | Hybrid Transformer Demucs: waveform and spectrogram U-Nets joined by a cross-domain transformer | 174 MB | research only | 8.86 · 9.55 · 9.30 · 5.69 | 1.4 (vocals only: 1.2) · 0.70 · 0.21 |
| `'htdemucs_ft'` | four fine-tuned HTDemucs, one per source | 4 × 174 MB | research only | not measured | vocals only: 1.1 · not measured · not measured |

SDR as measured below (Verification), identical to the Python originals'. The previews are 7 s excerpts, so these numbers sit apart from full-track results in scale, not in order: the Demucs README reports overall SDR 9.0 for fine-tuned HT Demucs against 5.3 for Open-Unmix on the MUSDB18-HQ test set.

Real-time factor: processing time over duration, lower is faster. 27 s of music; onnxruntime-node and onnxruntime-web 1.30, the web one in headless Chromium (wasm, Metal WebGPU); the best of two to four runs spread over a day on a 14-core M4 Max shared with other jobs (load averages 45 to 180), so upper bounds. The Python originals' best there: 0.43 (open-unmix) and 2.8 (demucs). umxhq's time is mostly JS (ONNX Runtime takes 6 to 12 %: STFT, Wiener EM, iSTFT around it), and it runs faster on wasm than on WebGPU; htdemucs's is about 80 % inference, where WebGPU pays. Memory: umxhq 1.3 GB, hybrid models 3 GB at peak in Node.

`targets` picks a subset: `targets: ['vocals']` runs only the vocals graph of `umxhq` (against the residual, see Algorithm: 2.4× faster, vocals SDR 6.50 instead of 6.75) and of `htdemucs_ft`, and skips the other sources' iSTFT for `htdemucs`. Presets resample to their 44.1 kHz and back.

## Weights

Nothing ships in this package and nothing is fetched by default. Export the weights once from the original checkpoints:

```sh
pip install torch openunmix onnx onnxruntime onnxscript
python3 node_modules/@audio/neural-separate/scripts/export-openunmix.py --model umxhq --verify

pip install demucs
python3 node_modules/@audio/neural-separate/scripts/export-htdemucs.py --model htdemucs --verify
```

They land in `$AUDIO_NEURAL_CACHE/<model>/`, default `~/.cache/audiojs/neural/<model>/`, where a preset looks in Node; a missing file throws, naming itself and the script. Elsewhere, serve that directory and pass its URL: `separate(audio, { model: 'umxhq', weights: 'https://…/' })`; `@audio/neural-runtime` fetches each file once and caches it (Cache API in the browser). `--verify` compares the ONNX graphs with the PyTorch modules and writes the reference separation `test.js` checks the JS pipeline against. `export-openunmix.py --fp16` also writes float16 weights with float32 I/O, half the size.

- **`export-openunmix.py`** exports each target's `OpenUnmix` through a wrapper that takes the frame count from the traced tensor (upstream reads it off `x.data.shape`, which export freezes), with the TorchScript exporter (the `torch.export` one bakes the LSTM's sequence length into a reshape). Magnitude `[1, C, F, T]` in and out, T dynamic. `umxhq`'s bandwidth restriction (`max_bin`, 1487 of 2049 bins for `n_fft=4096`@44.1kHz) crops the network's *input* below 16 kHz for efficiency; the final dense layer regresses the **full** bin range from that reduced representation (a learned extrapolation, not a literal zero-fill).
- **`export-htdemucs.py`** follows [sevagh/demucs.onnx](https://github.com/sevagh/demucs.onnx): the STFT and iSTFT, which ONNX export cannot carry, move out of the graph. Rather than a vendored copy of `htdemucs.py`, it runs upstream's own `forward` with its four transform methods swapped on the instance. One graph per 7.8 s segment: `mix [1, 2, 343980]` and `mix_spec [1, 4, 2048, 336]` (complex as channels) in, `stems_spec [1, 4, 4, 2048, 336]` and `stems_wave [1, 4, 2, 343980]` out; a source is the iSTFT of the first plus the second.

## Algorithm

**`'openunmix'`, `'mask'`** (spectrogram models):

1. **STFT** each channel — `n_fft` 4096, hop 1024, periodic Hann, `center=True` (torch.stft-compatible reflect padding) — own implementation on top of [`fourier-transform`](https://github.com/scijs/fourier-transform)'s raw FFT (not its bundled `stft` submodule, which zero-pads instead of centering — Open-Unmix's own filterbank is `torch.stft(..., center=True, pad_mode="reflect")`, so this package matches that framing exactly rather than reusing a differently-conventioned STFT).
2. **Magnitude** per channel → **one ONNX run per target** (or one multi-target graph) → estimated magnitude (`modelType: 'openunmix'`) or a `[0,1]` mask multiplied by the mixture magnitude (`modelType: 'mask'`).
3. **Multichannel Wiener EM** (`wiener` option, default 1 iteration) refines the per-target magnitude estimates into full complex spectra, using the mixture's spatial (inter-channel) structure — see below. It runs in windows of 300 frames, as open-unmix's `Separator` does (`wiener_win_len=300`), each window estimating its own spatial covariances; a single target runs against a residual (open-unmix's `residual=True`), since EM needs two sources.
4. **iSTFT** each target's channel spectra back to waveforms.
5. Long files are **chunked** (`chunk` seconds, default 30, with `overlap` seconds crossfade, default 2) so a bi-LSTM-class model's memory stays bounded; chunk stems are stitched back with a linear crossfade. Trade-off: the model only sees `chunk` seconds of context at once, same reasoning as Open-Unmix's own reference `Separator` batching its Wiener step into `wiener_win_len=300`-frame windows (`openunmix/model.py`) — coarser here (whole pipeline per chunk, not just the Wiener step), chosen for a simple, uniform memory bound across both spectral and waveform model families.

**`'hybrid'`** (Hybrid Transformer Demucs): `demucs.apply.apply_model` with `split=True, overlap=0.25, shifts=0`, and `demucs.api`'s normalization by the whole input's mean and standard deviation. Segments of 7.8 s start every 5.85 s; each is centered in its window with the neighbouring input as context, zeros past the ends. Per segment: HTDemucs's framing (reflect re-padding by ¾ hop, `normalized=True`, the Nyquist bin and two edge frames dropped) → ONNX → frequency branch through the iSTFT plus time branch. Segments overlap-add under a triangular window. `shifts` (averaging randomly time-shifted runs, "up to 0.2 points" of SDR per demucs's docstring) is not implemented; the CLI's default single shift averages nothing.

`modelType: 'waveform'` (Demucs v2-class) skips steps 1–4 entirely: chunked raw audio in `[1, C, N]`, stacked stems out `[1, S, C, N]`, no STFT or Wiener step — Demucs v2 operates in the time domain by design.

## Verification

| Check | Result |
|---|---|
| ONNX vs PyTorch module, random input (`--verify`) | umxhq max \|diff\| ≤ 5.7e-6 of max \|y\| (fp16: ≤ 5.3e-3); stems of htdemucs ≤ 1.6e-4, of htdemucs_ft ≤ 2.5e-4 |
| Pipeline vs `openunmix.Separator` in float64, 9 s reference mix (`test.js`) | 112–134 dB SNR per stem |
| Pipeline vs `demucs.apply.apply_model`, same mix (`test.js`) | htdemucs 80–84 dB, htdemucs_ft 78–89 dB SNR per stem |
| `wienerFilter` vs open-unmix's `wiener()`, float64 fixture (`test.js`) | ratio mask: 3e-12; mixture phase: 1.7e-7, upstream's `atan2` adds a float32 π |

The 50 test tracks of the MUSDB18 7 s previews (`musdb.DB(download=True)`; its terms are educational and non-commercial, so it serves measurement only), SDR by [museval](https://github.com/sigsep/sigsep-mus-eval) (BSSEval v4, 1 s windows, median over windows, then over tracks: the SiSEC 2018 aggregation):

| | vocals | drums | bass | other | this package vs Python, waveform SNR per track (median / lowest) |
|---|---|---|---|---|---|
| `umxhq`, open-unmix `Separator` | 6.75 | 6.11 | 5.00 | 3.36 | |
| `umxhq`, this package | 6.75 | 6.11 | 5.00 | 3.36 | 116–125 / 72–83 dB |
| `htdemucs`, demucs `apply_model` | 8.86 | 9.55 | 9.30 | 5.69 | |
| `htdemucs`, this package | 8.86 | 9.55 | 9.30 | 5.69 | 78–88 / 56–75 dB |

Per track, the SDRs differ by at most 0.003 dB.

Upstream runs its Wiener EM in float32. On the reference mix that alone moves its stems by 71 to 93 dB SNR against a float64 run of the same code, which this port matches to 112 to 134 dB; the float64 run is the reference.

## Wiener filter

`wienerFilter(mixStft, estimates, opts)` is a from-scratch JS port of the algorithm in [norbert](https://github.com/sigsep/norbert) (Liutkus & Stöter), with [open-unmix-pytorch](https://github.com/sigsep/open-unmix-pytorch)'s `openunmix/filtering.py` defaults (`eps=1e-10`, `softmask=False`, `scale_factor=10`) — softmask off by default, matching upstream's own recommendation ("`softmask=False` is recommended... once the model estimates are themselves good"). The core update — re-estimate each source's power spectral density and spatial covariance matrix, rebuild the modelled mixture covariance, apply the resulting multichannel Wiener gain, iterate — is the local Gaussian model from **Duong, Vincent, Gribonval, "Under-determined reverberant audio source separation using a full-rank spatial covariance model," IEEE TASLP 18(7), 2010**. The gain applies as `v_j R_j (Cxx⁻¹ x)`: one inverse times the mixture per bin, then one C×C product per source. The covariances are Hermitian, so stereo takes the closed-form 2×2 inverse, other channel counts a Gauss-Jordan solver.

```js
import { stft, wienerFilter, istft } from '@audio/neural-separate'

let mixStft = [stft(left, {}), stft(right, {})]                    // per channel
let estimates = { vocals: vocalsMag, drums: drumsMag }             // per target: magnitude[channel][frame] (Float64Array(bins))
let complex = wienerFilter(mixStft, estimates, { iterations: 1 })  // per target: complex STFT per channel
let vocals = complex.vocals.map(ch => istft(ch, { length: left.length }))
```

`iterations: 0` returns the initial estimate untouched ("raw masks"). `softmask: true` uses a ratio mask that sums to the mixture exactly, by construction; `residual: true` appends a `'residual'` target (mixture minus the other targets) computed before EM. **Known limitation**: multichannel Wiener EM assumes genuine inter-channel diversity — mono content duplicated to stereo (identical L≡R) gives a rank-1 spatial covariance matrix per bin, a degenerate case where EM iterations can occasionally underperform `iterations: 0` on an already-good separation (real stereo mixes, with any actual left/right difference, don't hit this — verified in `test.js`).

## `opts`

| Option | Default | |
|---|---|---|
| `sampleRate` | — | required unless `audio` is `{ channelData, sampleRate }` |
| `model` | — | required. Preset `'umxhq' \| 'htdemucs' \| 'htdemucs_ft'` · `url \| bytes` (single target, named `'stem'`) · `{ target: url \| { url, targets }, ... }` (one graph per target, Open-Unmix's own layout; a multi-source graph contributes its own target) · `{ url, targets: [...] }` (one multi-target graph, stacks a target axis) |
| `targets` | all | the targets to return |
| `weights` | `$AUDIO_NEURAL_CACHE` or `~/.cache/audiojs/neural` (Node) | where a preset's files are: URL, or a directory in Node |
| `modelType` | `'openunmix'` | `'openunmix'` (magnitude out) · `'mask'` (`[0,1]` mask out, multiplied by mixture magnitude) · `'hybrid'` (demucs.onnx contract) · `'waveform'` (Demucs v2-class); presets know theirs |
| `wiener` | `1` | EM iterations; `0` = raw masks. Ignored for `'hybrid'` and `'waveform'` |
| `wienerWindow` | `300` | frames per EM window |
| `chunk` / `overlap` | `30` / `2` (seconds) | `overlap` must be `< chunk`; not used by `'hybrid'`, which segments as demucs does |
| `segment` | `343980` | `'hybrid'`: segment length in samples when the graph does not declare its input length |
| `targetRate` | the preset's rate, else the input rate | resample to the model's rate for inference; stems are resampled back to the input rate |
| `device` | — | passed through to `@audio/neural-runtime`'s `load()` as `backend` |
| `dtype` | `'float32'` | only `'float32'` tensor marshalling is implemented; anything else throws |
| `progress` | — | `({ chunk, totalChunks }) => {}` |
| `session` | — | overrides `@audio/neural-runtime`'s `load()` — for tests, or a custom ORT setup |

Mono input is duplicated to stereo internally (matching `openunmix.utils.preprocess`'s own "if we have mono, we duplicate it to get stereo"), so stems always come back stereo. `separate()` returns `{ stems, sampleRate, residual }` — `residual` is the mixture minus the sum of all stems, per channel, at the input rate.

## Precedence and licenses

Open-source stem separation has three lineages; this package runs the first two.

| Project | Code | Weights | |
|---|---|---|---|
| **Open-Unmix** (`umx`, `umxhq`) | MIT | MIT ([Zenodo](https://zenodo.org/records/3370489)-declared) | trained on MUSDB18(-HQ); this package's primary target |
| **Open-Unmix** (`umxl`) | MIT | **CC BY-NC-SA 4.0 — non-commercial only** | despite being the `openunmix` package's own default variant name; trained on a private stems dataset (see the project [README](https://github.com/sigsep/open-unmix-pytorch#pre-trained-models)) |
| **Demucs** (Meta; `htdemucs`, `htdemucs_ft`) | MIT | **research only**: "The model weights are not covered by the MIT license, and are provided only for scientific purposes" (the author in [#327](https://github.com/facebookresearch/demucs/issues/327), 2022; again in [#508](https://github.com/facebookresearch/demucs/issues/508), 2023) | trained on MUSDB18-HQ plus 800 songs; the repository is archived, maintained at [adefossez/demucs](https://github.com/adefossez/demucs) |
| **Spleeter** (Deezer) | MIT | **undocumented** | the README licenses only "the code of Spleeter"; the pretrained weights' license is an open, unresolved question ([deezer/spleeter#898](https://github.com/deezer/spleeter/issues/898)) — do not assume MIT |

Audit any weight source yourself before shipping it — this table reflects what each project states as of this writing, not a guarantee. `scripts/export-openunmix.py` defaults to `umxhq` (not `umxl`) for exactly this reason.

## Reference

Stöter, Uhlich, Liutkus, Mitsufuji, "Open-Unmix - A Reference Implementation for Music Source Separation," *JOSS* 4(41), 2019. · Rouard, Massa, Défossez, "Hybrid Transformers for Music Source Separation," *ICASSP* 2023. · Duong, Vincent, Gribonval, "Under-determined reverberant audio source separation using a full-rank spatial covariance model," *IEEE TASLP* 18(7), 2010. · Rafii, Liutkus, Stöter, Mimilakis, Bittner, "The MUSDB18 corpus for music separation," 2017. · [norbert](https://github.com/sigsep/norbert) (Liutkus & Stöter) · [open-unmix-pytorch](https://github.com/sigsep/open-unmix-pytorch) · [Demucs](https://github.com/facebookresearch/demucs) · [demucs.onnx](https://github.com/sevagh/demucs.onnx) · [museval](https://github.com/sigsep/sigsep-mus-eval) · [Spleeter](https://github.com/deezer/spleeter).

**Use when:** you have (or can license) an ONNX-exported spectrogram-mask, Hybrid-Demucs or waveform separation model and want to run it — with proper multichannel Wiener refinement, not just the raw mask — dependency-free, in Node or the browser; `umxhq` where the weights must be MIT, `htdemucs` for the higher SDR under research-only terms.<br>
**Not for:** the classical, model-free case — reach for [`@audio/vocals`](https://github.com/audiojs/vocals) when a center-panned M/S trick is all you need; training a model (this is inference-only); real-time streaming (neither the bi-LSTM Open-Unmix nor Hybrid Transformer Demucs is causal — offline/chunked only, same as upstream).

---

Part of the [@audio/neural](https://github.com/audiojs/neural) lane.

MIT © [audiojs](https://github.com/audiojs)
