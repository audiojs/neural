# @audio/neural-separate

> Source separation (stems): spectrogram-mask models of the Open-Unmix class, run through `@audio/neural-runtime`'s model-agnostic ONNX adapter.

The ML upgrade to [`@audio/vocals`](https://github.com/audiojs/vocals)'s classical center-cancel: instead of one M/S trick, a trained model estimates each target's magnitude spectrogram, and multichannel Wiener refinement turns those estimates into a full stereo separation — vocals, drums, bass, other, or whatever targets the model was trained on.

```js
import separate, { wienerFilter, stft, istft } from '@audio/neural-separate'

let { stems, residual } = await separate([left, right], {
  sampleRate: 44100,
  model: { vocals: 'vocals.onnx', drums: 'drums.onnx', bass: 'bass.onnx', other: 'other.onnx' },
  modelType: 'openunmix',
})
stems.vocals   // Float32Array[2]
```

**Status — honest**: the DSP pipeline (STFT/iSTFT, chunking + crossfade, multichannel Wiener EM, ONNX I/O marshalling for both spectral and waveform model families) is complete and tested — against oracle magnitude masks (no weights needed) and a real ONNX identity-mask model run through `onnxruntime-node`. No pretrained weights ship in this package or are fetched by its tests. `scripts/export-openunmix.py` converts official Open-Unmix checkpoints to this package's ONNX contract; it is reviewed carefully against the upstream source but **not executed** — this environment has no `torch` (installing it here would be 500+ MB for a package that otherwise has none) and pretrained weights require a network fetch. Run `--verify` before trusting an export.

## Install

```
npm install @audio/neural-separate
```

## Algorithm

1. **STFT** each channel — `n_fft` 4096, hop 1024, periodic Hann, `center=True` (torch.stft-compatible reflect padding) — own implementation on top of [`fourier-transform`](https://github.com/scijs/fourier-transform)'s raw FFT (not its bundled `stft` submodule, which zero-pads instead of centering — Open-Unmix's own filterbank is `torch.stft(..., center=True, pad_mode="reflect")`, so this package matches that framing exactly rather than reusing a differently-conventioned STFT).
2. **Magnitude** per channel → **one ONNX run per target** (or one multi-target graph) → estimated magnitude (`modelType: 'openunmix'`) or a `[0,1]` mask multiplied by the mixture magnitude (`modelType: 'mask'`).
3. **Multichannel Wiener EM** (`wiener` option, default 1 iteration) refines the per-target magnitude estimates into full complex spectra, using the mixture's spatial (inter-channel) structure — see below.
4. **iSTFT** each target's channel spectra back to waveforms.
5. Long files are **chunked** (`chunk` seconds, default 30, with `overlap` seconds crossfade, default 2) so a bi-LSTM-class model's memory stays bounded; chunk stems are stitched back with a linear crossfade. Trade-off: the model only sees `chunk` seconds of context at once, same reasoning as Open-Unmix's own reference `Separator` batching its Wiener step into `wiener_win_len=300`-frame windows (`openunmix/model.py`) — coarser here (whole pipeline per chunk, not just the Wiener step), chosen for a simple, uniform memory bound across both spectral and waveform model families.

`modelType: 'waveform'` (Demucs-class) skips steps 1–4 entirely: chunked raw audio in `[1, C, N]`, stacked stems out `[1, S, C, N]`, no STFT or Wiener step — Demucs operates in the time domain by design.

## Wiener filter

`wienerFilter(mixStft, estimates, opts)` is a from-scratch JS port of the algorithm in [norbert](https://github.com/sigsep/norbert) (Liutkus & Stöter), with [open-unmix-pytorch](https://github.com/sigsep/open-unmix-pytorch)'s `openunmix/filtering.py` defaults (`eps=1e-10`, `softmask=False`, `scale_factor=10`) — softmask off by default, matching upstream's own recommendation ("`softmask=False` is recommended... once the model estimates are themselves good"). The core update — re-estimate each source's power spectral density and spatial covariance matrix, rebuild the modelled mixture covariance, apply the resulting multichannel Wiener gain, iterate — is the local Gaussian model from **Duong, Vincent, Gribonval, "Under-determined reverberant audio source separation using a full-rank spatial covariance model," IEEE TASLP 18(7), 2010**. Complex linear algebra (matrix inversion per time-frequency bin) uses a generic Gauss-Jordan solver rather than norbert/open-unmix's hand-specialized 1- and 2-channel closed forms — correct for any channel count, negligible cost difference at the channel counts (1–2) this package actually sees.

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
| `model` | — | required. `url \| bytes` (single target, named `'stem'`) · `{ target: url, ... }` (one graph per target, Open-Unmix's own layout) · `{ url, targets: [...] }` (one multi-target graph, stacks a target axis) |
| `modelType` | `'openunmix'` | `'openunmix'` (magnitude out) · `'mask'` (`[0,1]` mask out, multiplied by mixture magnitude) · `'waveform'` (Demucs-class) |
| `wiener` | `1` | EM iterations; `0` = raw masks. Ignored for `modelType: 'waveform'` |
| `chunk` / `overlap` | `30` / `2` (seconds) | `overlap` must be `< chunk` |
| `targetRate` | input rate | resample to the model's rate for inference; stems are resampled back to the input rate |
| `device` | — | passed through to `@audio/neural-runtime`'s `load()` as `backend` |
| `dtype` | `'float32'` | only `'float32'` tensor marshalling is implemented; anything else throws |
| `progress` | — | `({ chunk, totalChunks }) => {}` |
| `session` | — | overrides `@audio/neural-runtime`'s `load()` — for tests, or a custom ORT setup |

Mono input is duplicated to stereo internally (matching `openunmix.utils.preprocess`'s own "if we have mono, we duplicate it to get stereo"), so stems always come back stereo. `separate()` returns `{ stems, sampleRate, residual }` — `residual` is the mixture minus the sum of all stems, per channel, at the input rate.

## Getting weights: `scripts/export-openunmix.py`

```
pip install torch openunmix onnx onnxruntime
python3 scripts/export-openunmix.py --model umxhq --targets vocals,drums,bass,other --out-dir ./onnx --verify
```

Wraps each target's `OpenUnmix` module (its `forward()` already takes and returns `(nb_samples, nb_channels, nb_bins, nb_frames)` — exactly this package's `[1, C, F, T]` contract, no permute needed) and calls `torch.onnx.export` per target, opset 17, dynamic frame axis, `--fp16` for a float16 variant (post-converted via `onnxconverter-common`, not exported directly — LSTM's fp16 ONNX export has historically been unreliable), `--combined` for one multi-target graph, `--verify` to run the export through `onnxruntime` on random input and assert `max|Δ| < 1e-4` against the PyTorch model directly. **Not run in this environment** — reviewed against open-unmix-pytorch's actual source (`model.py`, `utils.py`, `filtering.py`), not executed.

`umxhq`'s bandwidth restriction (`max_bin`, ~1487 of 2049 bins for `n_fft=4096`@44.1kHz) crops the network's *input* below 16kHz for efficiency — contrary to the common "zero-pads above max_bin" description, the final dense layer regresses the **full** bin range from that reduced representation (a learned extrapolation, not a literal zero-fill; see the script's header comment for the exact source citation).

## Precedence and licenses

Open-source stem separation has three lineages; this package targets the first (spectrogram-mask, Open-Unmix class) with a JS-side design that also supports the second (waveform, Demucs class) as `modelType: 'waveform'`.

| Project | Code | Weights | |
|---|---|---|---|
| **Open-Unmix** (`umx`, `umxhq`) | MIT | MIT ([Zenodo](https://zenodo.org/records/3370489)-declared) | trained on MUSDB18(-HQ); this package's primary target |
| **Open-Unmix** (`umxl`) | MIT | **CC BY-NC-SA 4.0 — non-commercial only** | despite being the `openunmix` package's own default variant name; trained on a private stems dataset (see the project [README](https://github.com/sigsep/open-unmix-pytorch#pre-trained-models)) |
| **Demucs** (Meta) | MIT | MIT | no non-commercial carve-out found in-repo |
| **Spleeter** (Deezer) | MIT | **undocumented** | the README licenses only "the code of Spleeter"; the pretrained weights' license is an open, unresolved question ([deezer/spleeter#898](https://github.com/deezer/spleeter/issues/898)) — do not assume MIT |

Audit any weight source yourself before shipping it — this table reflects what each project states as of this writing, not a guarantee. `scripts/export-openunmix.py` defaults to `umxhq` (not `umxl`) for exactly this reason.

## Reference

Stöter, Uhlich, Liutkus, Mitsufuji, "Open-Unmix - A Reference Implementation for Music Source Separation," *JOSS* 4(41), 2019. · Duong, Vincent, Gribonval, "Under-determined reverberant audio source separation using a full-rank spatial covariance model," *IEEE TASLP* 18(7), 2010. · [norbert](https://github.com/sigsep/norbert) (Liutkus & Stöter) · [open-unmix-pytorch](https://github.com/sigsep/open-unmix-pytorch) · [Demucs](https://github.com/facebookresearch/demucs) · [Spleeter](https://github.com/deezer/spleeter).

**Use when:** you have (or can license) an ONNX-exported spectrogram-mask or waveform separation model and want to run it — with proper multichannel Wiener refinement, not just the raw mask — dependency-free, in Node or the browser.<br>
**Not for:** the classical, model-free case — reach for [`@audio/vocals`](https://github.com/audiojs/vocals) when a center-panned M/S trick is all you need; training a model (this is inference-only); real-time streaming (the bi-LSTM Open-Unmix architecture is not causal — offline/chunked only, same as upstream).

---

Part of the [@audio/neural](https://github.com/audiojs/neural) lane.

MIT © [audiojs](https://github.com/audiojs)
