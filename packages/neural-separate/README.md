# @audio/neural-separate

> Source separation (stems): a song's by SCNet, Hybrid Transformer Demucs and Open-Unmix, a soundtrack's dialogue, music and effects by MRX and TIGER, through ONNX, matching the original PyTorch implementations.

The ML upgrade to [`@audio/vocals`](https://github.com/audiojs/vocals)'s classical center-cancel: instead of one M/S trick, a trained model estimates each target (vocals, drums, bass, other), and the pipeline around it (STFT, Wiener refinement, chunking, overlap-add) is ported from the reference implementations and checked against them.

```js
import separate from '@audio/neural-separate'

let { stems, residual } = await separate([left, right], { sampleRate: 44100, model: 'scnet-large' })
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
| `'htdemucs_ft'` | four fine-tuned HTDemucs, one per source | 4 × 174 MB | research only | 8.67 · 9.45 · 9.68 · 5.58 | vocals only: 1.1 · not measured · not measured |
| `'scnet-large'` | SCNet-large: band-split convolutions around dual-path LSTMs, on the complex spectrogram | 45 MB (export 169 MB) | MIT | 11.00 · 10.27 · 8.21 · 6.87 | 2.1 · 7.8 · 5.0 |
| `'scnet'` | SCNet: the same at half the width | 13 MB (export 43 MB) | MIT | 9.88 · 9.43 · 8.35 · 6.15 | 0.70 · 1.7 · 0.91 |
| `'mel-roformer'` | Mel-Band RoFormer (Kimberley Jensen's vocals model): axial transformers over time and mel bands, on the complex spectrogram; written for the GPU | 464 MB, float16 weights (export 928 MB) | MIT | vocals only: 12.08 | 1.4 · not measured · 0.56 (Node's WebGPU; the export 0.44) |

SDR as measured below (Verification), identical to the Python originals'. The previews are 7 s excerpts, so these numbers sit apart from full-track results in scale, not in order: the Demucs README reports overall SDR 9.0 for fine-tuned HT Demucs against 5.3 for Open-Unmix on the MUSDB18-HQ test set.

`scnet-large` is ahead of iZotope RX 12 Music Rebalance at its Best quality on drums and other and on most songs for every stem, behind on bass by the median (Measured). Its time is the segments' overlap: each sample is heard in four 11 s segments (`step` 0.25); `step: 0.5` takes half the time (1.0) for 0.07 to 0.17 dB less (27 s of four MUSDB18 training previews end to end, six of them), and a song shorter than a segment takes one. The SCNet times are of one run, a busy machine's (load averages 53 to 93), beside htdemucs's 0.24 and RX 12 Music Rebalance's 1.22 at Best in the same run. Real-time factor: processing time over duration, lower is faster. 27 s of music; onnxruntime-node and onnxruntime-web 1.30, the web one in headless Chromium (wasm, Metal WebGPU); the best of two to four runs spread over a day on a 14-core M4 Max shared with other jobs (load averages 45 to 180), so upper bounds. The Python originals' best there: 0.43 (open-unmix) and 2.8 (demucs). umxhq's time is mostly JS (ONNX Runtime takes 6 to 12 %: STFT, Wiener EM, iSTFT around it), and it runs faster on wasm than on WebGPU; htdemucs's is about 80 % inference, where WebGPU pays. Memory: umxhq 1.3 GB, hybrid models 3 GB at peak in Node.

A soundtrack's stems, Divide and Remaster's three (the cocktail fork problem: Petermann, Wichern, Wang, Le Roux, ICASSP 2022):

| `model` | Architecture | Files | Weights | Targets | Real-time factor: Node CPU · browser wasm · WebGPU |
|---|---|---|---|---|---|
| `'mrx'` | MRX: magnitudes at three STFT resolutions (Hann 1024, 2048, 8192, hop 256) into one hidden layer, a BLSTM per source, their mean, a real mask per source and resolution | 31 MB (export 122 MB) | MIT | `dialogue`, `music`, `effects` | 0.21 · 0.31 · 0.75 |
| `'tiger'` | TIGER: three band-split models (57 bands, multi-scale convolutions and frame and frequency attention, 1.4 M parameters each), a complex mask each, one source kept from each | 7.3 MB (export 29 MB) | Apache 2.0 | `dialogue`, `music`, `effects` | 7.5–23 · 9.2 · 3.1 |

Measured on Divide and Remaster v3's English test set (Watcharasupat, Wu, Orife, 2024; CC BY-SA 4.0, built from sources that allow commercial use), 150 of its 1200 clips of 60 s, every 8th ([`audio`](https://github.com/audiojs/audio)'s `bench/rx/scene.mjs`): each stem against its reference over the whole clip, SNR the median over clips (the measure Bandit v2's paper reports on this set) and SI-SDR the mean (MRX's on v2); remixes, a stem 6 dB up or down as the input plus (g − 1) times it, against the true remix, SNR, the median.

| | dialogue | music | effects | dialogue +6 dB | music −6 dB | effects −6 dB |
|---|---|---|---|---|---|---|
| the mixture as it is | | | | 7.17 | 13.35 | 12.84 |
| `mrx`, SNR · SI-SDR | 11.82 · 10.13 | 5.32 · 2.54 | 6.03 · 3.66 | 18.91 | 18.66 | 17.93 |
| Bandit v2, its paper (all 1200 clips; weights CC BY-SA 4.0) | 15.6 | 10.4 | 9.9 | | | |
| 30 of the clips (every 40th): the mixture as it is | | | | 7.75 | 13.38 | 11.93 |
| `mrx` | 10.92 · 9.81 | 5.17 · 2.80 | 5.72 · 3.27 | 18.65 | 18.29 | 17.72 |
| `tiger` | 12.68 · 12.35 | 10.24 · 8.06 | 8.06 · 7.50 | 20.98 | 21.46 | 21.15 |

TIGER, at about 50 times MRX's time (RTF 10.6 against 0.21 here), is ahead on 29, 28 and 29 of the 30 clips (dialogue, music, effects), by a median 2.3, 3.7 and 3.4 dB. MRX's own README reports 12.5 · 4.2 · 5.7 dB SI-SDR on DnR v2, the set it trained on, whose music holds singing; v3's music holds none, and its loudness and languages differ (v3's paper). DnR v2's test split comes only inside a 116 GB gzip of the whole set; v3's clips are fetched one by one. The real-time factors are of one run on a 14-core M4 Max shared with other jobs (load averages 25 to 60), so upper bounds: `mrx` a 60 s clip in 13 s; `tiger` three models of 15,000 small operators each, every sample in three 12 s segments.

The browser's, the shipped files (Size, below) on 30 s (a song; a Divide and Remaster clip), model load included:
onnxruntime-web 1.30 in headless Chromium 153, wasm on its default 4 threads, WebGPU on Metal, the faster of two runs
on a 14-core M4 Max shared with other jobs (load averages 110 to 300; the other run, at 9 to 140 with another job
training on the GPU, took 1.5 to 2 times as long), so upper bounds. Peak memory, the browser's processes
together (0.9 GB of it the browser idle): wasm `scnet` 3.0 GB, `scnet-large` 4.6, `mrx` 3.0, `tiger` 3.5; WebGPU 2.1,
2.4, 3.4, 1.5. On WebGPU `tiger`'s session takes 13 s to make (38,817 nodes), a 60 s clip about 3 minutes, MRX's 45 s;
on wasm 9 and 0.3 minutes. Float16 compute does not pay: `tiger`'s on WebGPU is 13 % faster and its music 4.7 dB from
the float32 graph's.

`targets` picks a subset: `targets: ['vocals']` runs only the vocals graph of `umxhq` (against the residual, see Algorithm: 2.4× faster, vocals SDR 6.50 instead of 6.75) and of `htdemucs_ft`, and skips the other sources' iSTFT for `htdemucs`. Presets resample to their 44.1 kHz and back.

## Weights

Nothing ships in this package. `scnet-large`, `scnet`, `mrx` and `tiger` read a compact file each, `<model>.int8.onnx`
(Size, below), and `mel-roformer` its float16 weights, `mel-roformer.fp16.onnx` (or `roformer.js`'s `mel-roformer.engine.bin`), hosted on Hugging Face
(`audiojs/scnet-large`, `audiojs/scnet`, `audiojs/mrx`, `audiojs/tiger-dnr`, `audiojs/mel-roformer`, each with its model
card and licence): without `weights`, `https://huggingface.co/<repo>/resolve/<revision>/<file>` at the
revision `REVISIONS` pins, fetched once and checked against its SHA-256 (the browser keeps it in the Cache API, Node in
the neural cache, below). Every model can also be exported from its original checkpoint:

```sh
pip install torch openunmix onnx onnxruntime onnxscript
python3 node_modules/@audio/neural-separate/scripts/export-openunmix.py --model umxhq --verify

pip install demucs
python3 node_modules/@audio/neural-separate/scripts/export-htdemucs.py --model htdemucs --verify

pip install pyyaml
python3 node_modules/@audio/neural-separate/scripts/export-scnet.py --model scnet-large --verify

pip install pyloudnorm
python3 node_modules/@audio/neural-separate/scripts/export-mrx.py --verify

pip install safetensors
python3 node_modules/@audio/neural-separate/scripts/export-tiger.py --verify

pip install einops beartype rotary-embedding-torch librosa
python3 node_modules/@audio/neural-separate/scripts/export-roformer.py --verify --fp16 --engine
node node_modules/@audio/neural-separate/scripts/engine.mjs       # roformer.js in Chromium against onnxruntime (playwright)

python3 node_modules/@audio/neural-separate/scripts/compact.py --model tiger --verify   # tiger.int8.onnx from tiger.onnx
```

They land in `$AUDIO_NEURAL_CACHE/<model>/`, default `~/.cache/audiojs/neural/<model>/`, where a preset looks in Node: its compact file when it is the hosted one (its SHA-256), else its export (float32, the numbers the Python original gives), else the hosted file, fetched there once and written whole (offline, a compact file of another SHA-256, `compact.py`'s own build, is read as it is); a missing file throws, naming itself and the script. `weights` a directory: its files as they are, the compact one first. Elsewhere, serve that directory and pass its URL: `separate(audio, { model: 'umxhq', weights: 'https://…/' })`; `@audio/neural-runtime` fetches each file once and caches it (Cache API in the browser). `--verify` compares the ONNX graphs with the PyTorch modules and writes the reference separation `test.js` checks the JS pipeline against. `export-openunmix.py --fp16` also writes float16 weights with float32 I/O, half the size; `export-scnet.py --fp16` float16 weights with float32 compute (`compact.py --as fp16`).

- **`export-openunmix.py`** exports each target's `OpenUnmix` through a wrapper that takes the frame count from the traced tensor (upstream reads it off `x.data.shape`, which export freezes), with the TorchScript exporter (the `torch.export` one bakes the LSTM's sequence length into a reshape). Magnitude `[1, C, F, T]` in and out, T dynamic. `umxhq`'s bandwidth restriction (`max_bin`, 1487 of 2049 bins for `n_fft=4096`@44.1kHz) crops the network's *input* below 16 kHz for efficiency; the final dense layer regresses the **full** bin range from that reduced representation (a learned extrapolation, not a literal zero-fill).
- **`export-scnet.py`** fetches SCNet's code from ZFTurbo/Music-Source-Separation-Training at a pinned commit and the author's MUSDB18-HQ checkpoint from that repository's releases, and exports the network between its STFT and iSTFT (the same split as demucs.onnx's): `mix_spec [1, 4, 2049, 476]` in, `stems_spec [1, 16, 2049, 476]` out, one 11 s segment. Two changes to the graph, neither to its function: the rFFT and irFFT over time in its separation network become products with cosine and sine matrices (torch 2.14 exports no `fft_rfft`), and each `GroupNorm(1, C)` reduces its mean and variance one axis at a time: exported whole, onnxruntime summed a 10-million-value group in float32 and the stems came out 52 to 55 dB from PyTorch's, against 110 dB now (PyTorch's own float32 against float64: 110 dB). `--fp16` also writes float16 weights.
- **`export-mrx.py`** fetches MRX's code and MERL's checkpoint (`default_mrx_pre_trained_weights.pth`, trained with the SNR loss so its stems keep the mixture's level; the three others are scale-invariant or trained on loudness- or EQ-adapted stems, and their stems come out at another level: on Divide and Remaster v3 their SNR is negative where their SI-SDR is up to 1 dB higher) from merlresearch/cocktail-fork-separation at a pinned commit, and exports the network between its STFTs and iSTFTs: each channel's three magnitude spectrograms `mag_1024 [B, 513, T]`, `mag_2048 [B, 1025, T]`, `mag_8192 [B, 4097, T]` in (scaled 1/√n, torch.stft's `normalized=True`), the masks `mask_<n> [B, 3, F, T]` out, music, speech and sfx; B and T free.
- **`export-tiger.py`** fetches TIGER's model code (JusperLee/TIGER, `look2hear/models/tiger_dnr.py` and the two layer modules it reads) at a pinned commit and its Divide and Remaster weights (huggingface.co/JusperLee/TIGER-DnR) at a pinned revision, and exports TIGERDNR's three models between their STFT and iSTFT as one graph, each its kept source (the first's dialogue, the second's effects, the third's music): `mix_spec [1, 2, 1025, 1034]` (one channel's STFT, n 2048, hop 512, Hann, unnormalized) in, `stems_spec [1, 6, 1025, 1034]` (dialogue, music, effects, re and im) out, one 12 s segment. Three changes to the graph, none to its function: `F.adaptive_avg_pool1d`, which the exporter takes only for output sizes dividing the input's, becomes a cumulative sum read at torch's windows; UConvBlock's global sum starts at its first term, not at a zeros tensor exported as a constant the size of the features; each `GroupNorm(1, C)` reduces axis by axis, as SCNet's. ONNX Runtime runs the graph in 12 s segments at about twice real time on a CPU: three models of 15,000 small operators each, every sample in three segments.
- **`export-roformer.py`** fetches Mel-Band RoFormer's code and Kim's config from ZFTurbo/Music-Source-Separation-Training at a pinned commit and Kim's checkpoint (huggingface.co/KimberleyJSN/melbandroformer, MIT) at a pinned revision, checked by its SHA-256, and exports the network between its STFT and iSTFT: `mix_spec [1, 4, 1025, 801]` in, `stems_spec [1, 4, 1025, 801]` (the vocals) out, one 8 s segment every 4 s. It is written for the GPU, its function unchanged (`--verify`: 1e-5 to 2e-4 of the peak from MelBandRoformer.forward, whose own float32 stands 5e-3 from its float64 on tones): the 60 bands' projections and mask MLPs as batched products, bands of near widths grouped and zero-padded; each RMSNorm's scale folded into the weights it feeds, the attention's scale into the queries, its gates' projection joined to q, k, v; the rotary embedding's interleaved pairs turned to halves (the same permutation of q's and k's features), so it is two products with cos and sin tables and one with a fixed matrix; the overlapping mel bands' masks averaged by two gathers, not a scatter. 826 operators where an export of BS-RoFormer as written has about 3,000, and SCNet's LSTMs none: on Node's WebGPU (Metal, M4 Max) 2.4 s a segment against 9 s on its CPU; SCNet-large's LSTMs take 19 of its 26 s a segment there. And no tensor passes 100 MB, each transformer run on slices of its sequences: onnxruntime 1.30's WebGPU Softmax over a tensor past WebGPU's default 128 MiB storage binding came out wrong one run in two (36 dB from the CPU's at 307 MB, every run right at 123 MB), and the stems with it. `--fp16`: float16 weights, float32 compute, half the download, 3.0 s a segment and 48 dB from the float32 file's stems.
- **`roformer.js`**: in a browser whose GPU offers subgroup matrices (`chromium-experimental-subgroup-matrix`: Chromium on Metal), `mel-roformer` runs on kernels of its own instead of onnxruntime, from `mel-roformer.engine.bin` and `.json` (`export-roformer.py --engine`: the same tensors, hosted beside the ONNX file; a browser downloads one or the other). Matrix products and attention on the GPU's matrix units, float32: band split, norms, the q, k, v projection with the rotation and gates, attention in two passes over the keys (the rows' maxima, then the output in registers), feed-forward with GELU, the mask MLP with tanh and GLU, the masks averaged and applied; 163 dispatches a segment, recorded once; kept on the GPU a minute after a separation, so the next starts at once (its weights and 2 GB of buffers made in about a second the first time). Against onnxruntime's CPU on the same weights 101.5 dB (`scripts/engine.mjs`); in Chromium on an M4 Max under the same load as onnxruntime-web's WebGPU there, 1.5 to 1.8 s a segment against 2.6 to 3.3; GPU time by kernel: the time attention a third, the matrix products half (about 3 TFLOPS each, against onnxruntime's 1.8 for its float32 products and 0.5 to 0.75 for its attention). Elsewhere (Node, Firefox, Safari, a GPU without them) onnxruntime runs the ONNX file as before.
- **`export-htdemucs.py`** follows [sevagh/demucs.onnx](https://github.com/sevagh/demucs.onnx): the STFT and iSTFT, which ONNX export cannot carry, move out of the graph. Rather than a vendored copy of `htdemucs.py`, it runs upstream's own `forward` with its four transform methods swapped on the instance. One graph per 7.8 s segment: `mix [1, 2, 343980]` and `mix_spec [1, 4, 2048, 336]` (complex as channels) in, `stems_spec [1, 4, 4, 2048, 336]` and `stems_wave [1, 4, 2, 343980]` out; a source is the iSTFT of the first plus the second.

## Size

`scnet-large`, `scnet`, `mrx` and `tiger` each read one compact file, `<model>.int8.onnx`, that `scripts/compact.py`
makes from the export: the weights stored in 8 bits (symmetric, a scale per output channel; an LSTM's per gate row and
direction), the few whose rounding costs most in 16, computed in float32: onnxruntime folds each weight's Cast and scale
at load, on every backend, so the session holds float32 weights and runs as fast as the export. A variant is kept when
its cost is negligible: every stem's median within 0.05 dB of the export's (SDR on the 50 MUSDB18 test previews, SNR on
Divide and Remaster v3 test clips) and no song or clip worse by 0.5 dB or more; the smallest such is shipped.

| `model` | export (float32) | float16 weights | int8, every weight | **shipped**: int8, the costliest float16 | its gzip · brotli |
|---|---|---|---|---|---|
| `'scnet-large'` | 169.2 MB | 86.5 MB | 44.7 MB | **45.1 MB**: 83 of 88 weights int8 | 39.8 · 39.0 MB |
| `'scnet'` | 42.8 MB | 23.2 MB | 12.8 MB | **12.9 MB**: 77 of 82 | 10.7 · 10.2 MB |
| `'mrx'` | 122.3 MB | 61.5 MB | 31.3 MB | **31.3 MB**: all 39 | 27.5 · 26.6 MB |
| `'tiger'` | 28.6 MB (18.8 folded) | 10.9 MB | 7.3 MB | **7.3 MB**: all 444 | 5.2 · 2.6 MB |

Against the export, on the same songs or clips: the median per stem, then the largest loss on one song or clip, dB; and
the graph's own output against the export's on a calibration excerpt (two MUSDB18 training previews; a Divide and
Remaster v3 tuning clip), SNR.

| `model`, measure | export | float16 weights | int8, every weight | shipped |
|---|---|---|---|---|
| `'scnet-large'`, SDR: vocals · drums · bass · other | 11.00 · 10.27 · 8.21 · 6.87 | 10.99 · 10.27 · 8.21 · 6.87 | | **10.96 · 10.26 · 8.20 · 6.92** |
| … the song that lost most | | −0.004 | | −0.43 |
| … output, calibration | | 62.3 dB | 24.0 dB | 43.7 dB |
| `'scnet'`, SDR | 9.88 · 9.43 · 8.35 · 6.15 | 9.88 · 9.43 · 8.35 · 6.15 | 9.87 · 9.44 · 8.32 · 6.13 | **9.88 · 9.44 · 8.35 · 6.14** |
| … the song that lost most | | −0.004 | −2.61 | −0.29 |
| … output, calibration | | 60.6 dB | 23.4 dB | 42.4 dB |
| `'mrx'`, SNR, 30 clips: dialogue · music · effects | 10.92 · 5.17 · 5.72 | 10.92 · 5.17 · 5.72 | = shipped | **10.92 · 5.17 · 5.70** |
| … the clip that lost most | | −0.002 | | −0.04 |
| … output (masked spectra), calibration | | 73.8 dB | 40.4 dB | 40.4 dB |
| `'tiger'`, SNR, the same 30 clips | 12.68 · 10.24 · 8.06 | | = shipped | **12.66 · 10.23 · 8.03** |
| … the clip that lost most | | | | −0.10 |
| … output, calibration | | 73.9 dB | 41.4 dB | 41.4 dB |

Per song or clip the median change is within 0.015 dB everywhere; the largest losses fall where a stem is near silence:
SCNet's and SCNet-large's on PR - Happy Daze, whose vocals they separate at −2 dB SDR as exported (`scnet` with all 82
weights in int8: −2.6 dB there).
The remixes (`rebalance`'s, `scene`'s) move as little: vocals +6 dB 20.21 → 20.22 (`scnet-large`), dialogue +6 dB
18.29 → 18.30 (`mrx`).

Which weights stay float16: each weight rounded alone to int8 on a calibration excerpt (two MUSDB18 training previews;
a Divide and Remaster v3 tuning clip), the others float32, its cost the error it adds over the output's power; the
weights go to int8 in order of that cost per byte saved while the costs' sum stays 40 dB under the output (separate
weights' costs add: the sum predicted the whole to 0.2 dB). SCNet keeps the five layers ending its decoder in float16,
1 % of its values (`decoder.2.0`'s convolution, `decoder.1.1`'s three transposed convolutions, `decoder.2.1`'s first:
each alone at int8 26.8 to 39.0 dB); MRX (40.4 dB) and TIGER (41.4 dB) keep none.

Tried and left out: 4-bit weights (blocks of 32: SCNet's stems 7 dB from float32's; its LSTMs and linear layers alone,
18 dB); int8 scales per block of 32 values rather than per channel (31.5 dB against 26.3: still the same decoder
layers); DequantizeLinear for the int8 weights (onnxruntime 1.30 fuses it with the MatMul it feeds into MatMulNBits,
which quantizes the activations to int8 on a CPU: MRX 38.4 dB from float32 so, 40.4 as stored).

Hosts send these files as they are: Hugging Face's CDN and raw.githubusercontent.com set no Content-Encoding on a binary
(both answer any origin, `Access-Control-Allow-Origin: *`). gzip would save 12 to 17 % of SCNet's and MRX's, 29 % of
TIGER's, brotli 14 to 21 % and 65 % (TIGER's graph, 38,817 nodes, is most of its file): not worth a decoder in the page.

## Algorithm

**`'openunmix'`, `'mask'`** (spectrogram models):

1. **STFT** each channel — `n_fft` 4096, hop 1024, periodic Hann, `center=True` (torch.stft-compatible reflect padding) — own implementation on top of [`fourier-transform`](https://github.com/scijs/fourier-transform)'s raw FFT (not its bundled `stft` submodule, which zero-pads instead of centering — Open-Unmix's own filterbank is `torch.stft(..., center=True, pad_mode="reflect")`, so this package matches that framing exactly rather than reusing a differently-conventioned STFT).
2. **Magnitude** per channel → **one ONNX run per target** (or one multi-target graph) → estimated magnitude (`modelType: 'openunmix'`) or a `[0,1]` mask multiplied by the mixture magnitude (`modelType: 'mask'`).
3. **Multichannel Wiener EM** (`wiener` option, default 1 iteration) refines the per-target magnitude estimates into full complex spectra, using the mixture's spatial (inter-channel) structure — see below. It runs in windows of 300 frames, as open-unmix's `Separator` does (`wiener_win_len=300`), each window estimating its own spatial covariances; a single target runs against a residual (open-unmix's `residual=True`), since EM needs two sources.
4. **iSTFT** each target's channel spectra back to waveforms.
5. Long files are **chunked** (`chunk` seconds, default 30, with `overlap` seconds crossfade, default 2) so a bi-LSTM-class model's memory stays bounded; chunk stems are stitched back with a linear crossfade. Trade-off: the model only sees `chunk` seconds of context at once, same reasoning as Open-Unmix's own reference `Separator` batching its Wiener step into `wiener_win_len=300`-frame windows (`openunmix/model.py`) — coarser here (whole pipeline per chunk, not just the Wiener step), chosen for a simple, uniform memory bound across both spectral and waveform model families.

**`'hybrid'`** (Hybrid Transformer Demucs): `demucs.apply.apply_model` with `split=True, overlap=0.25, shifts=0`, and `demucs.api`'s normalization by the whole input's mean and standard deviation. Segments of 7.8 s start every 5.85 s; each is centered in its window with the neighbouring input as context, zeros past the ends. Per segment: HTDemucs's framing (reflect re-padding by ¾ hop, `normalized=True`, the Nyquist bin and two edge frames dropped) → ONNX → frequency branch through the iSTFT plus time branch. Segments overlap-add under a triangular window. `shifts` (averaging randomly time-shifted runs, "up to 0.2 points" of SDR per demucs's docstring) is not implemented; the CLI's default single shift averages nothing.

**`'complex'`** (SCNet; any model taking each channel's STFT, re and im as channels, and giving each source's): Music-Source-Separation-Training's `demix()`, its generic mode. The input is normalized by its mean and deviation; segments of 11 s start every `step` of a segment (0.25: every sample in four), each weighted by a window that fades in and out linearly over a tenth of it; an input longer than two segments less a step is first reflected out by that much on each side, and a segment running past the end is reflected out when more than half of it is input, zero-padded when not. Per segment: zero-padded to 476 frames as `SCNet.forward` pads, the model's STFT (n 4096, hop 1024, no window, normalized, centered), ONNX, iSTFT. Departures from `demix()`: the segments stop at the first to reach the end (it runs on while one starts before the end: a 7 s song took three segments, now one), the first and last fades are set per segment (it sets them per batch of 8, so its first segment fades in), and the mean goes back to no source (its callers add it to every one), so the stems sum to the input less its DC.

TIGER's DnR model runs as `'complex'` with its own segments, those of `TIGERDNR.wav_chunk_inference`: 12 s every 4 s (`step` 1/3), unweighted (`fade` 0), the input zero-padded by 8 s at each end and a segment past the end with zeros (`pad: 'zero'`), nothing normalized (`standardize: false`), each channel apart (`mono`); every sample is heard in three segments, their sum divided by three.

**`'multires'`** (MRX): upstream's `separate.separate_soundtrack`. The input is set to -27 LUFS (BS.1770-4 integrated loudness as pyloudnorm measures it, its K-weighting designed at the model's rate) and its stems scaled back by the same gain. Each channel is heard apart, so mono stays mono: its STFT at each window (Hann, hop 256, centered, reflect padding), the magnitudes into the graph, and each source the sum over the three resolutions of the iSTFT of its mask times that resolution's spectrogram. MRX's LSTMs hear the whole input at once; here it runs in chunks of 20 s (`chunk`) crossfaded over 2 s (`overlap`), which hold Node's peak at 1.9 GB on a 60 s clip (4.4 GB whole, 1.3 GB in 10 s chunks): on the 18 tune clips of Divide and Remaster v3 that costs 0.09, 0.02 and 0.24 dB of SNR (median, dialogue, music, effects) against the whole clip at once (`chunk: 3600`), and the remixes stay within 0.15 dB; 10 s chunks cost 0.47 dB of the dialogue.

`modelType: 'waveform'` (Demucs v2-class) skips steps 1–4 entirely: chunked raw audio in `[1, C, N]`, stacked stems out `[1, S, C, N]`, no STFT or Wiener step — Demucs v2 operates in the time domain by design.

## Verification

| Check | Result |
|---|---|
| ONNX vs PyTorch module, random input (`--verify`) | umxhq max \|diff\| ≤ 5.7e-6 of max \|y\| (fp16: ≤ 5.3e-3); stems of htdemucs ≤ 1.6e-4, of htdemucs_ft ≤ 2.5e-4 |
| Pipeline vs `openunmix.Separator` in float64, 9 s reference mix (`test.js`) | 112–134 dB SNR per stem |
| Pipeline vs `demucs.apply.apply_model`, same mix (`test.js`) | htdemucs 80–84 dB, htdemucs_ft 78–89 dB SNR per stem |
| ONNX vs `SCNet.forward`, noise and tones (`export-scnet.py --verify`) | scnet-large max \|diff\| ≤ 2.8e-6 of max \|y\|, scnet ≤ 2.2e-6 |
| Pipeline vs `SCNet.forward` on `reference.py`'s segments, a 20 s mix (`test.js`) | scnet-large 114–134 dB, scnet 123–133 dB SNR per stem |
| ONNX vs `MRX.forward`, noise and tones (`export-mrx.py --verify`) | max \|diff\| ≤ 1.2e-6 of max \|y\| |
| Pipeline vs `separate.separate_soundtrack` (pyloudnorm -27 LUFS in and back), a 20 s mix (`test.js`) | 101–134 dB SNR per stem, in one chunk; in 8 s chunks 12–50 dB |
| ONNX vs the three `TIGER.forward`, one 12 s segment of tones and noise (`export-tiger.py --verify`) | max \|diff\| ≤ 4.2e-7 of max \|y\| |
| Pipeline vs `TIGERDNR.wav_chunk_inference`, its segments through the verified graph, a 20 s mix (`test.js`) | 95–140 dB SNR per stem |
| Compact file vs export, a calibration excerpt (`compact.py --verify`) | scnet-large 43.7 dB, scnet 42.4, mrx 40.4, tiger 41.4 SNR (the worst output); folding alone: bit for bit |
| Pipeline, compact file vs `reference.py`'s stems, the same 20 s mix (`test.js`) | each stem's error under the mixture: scnet-large 25.7 dB (its other and vocals, which trade the synthetic voice; drums and bass 51), scnet 47.0, mrx 45.6, tiger 58.9 |
| Hosted files: the pinned revisions' SHA-256 (`test.js`, online); a fetched file of another refused | as `models[name].sha256` |
| Browser (onnxruntime-web, Chromium) vs Node, the same compact file, 30 s | `rebalance` and `scene` through `audio`, 8 s, the files fetched from Hugging Face: the output 135 and 145 dB from Node's on wasm, 134 and 148 dB on WebGPU |
| `wienerFilter` vs open-unmix's `wiener()`, float64 fixture (`test.js`) | ratio mask: 3e-12; mixture phase: 1.7e-7, upstream's `atan2` adds a float32 π |

The 50 test tracks of the MUSDB18 7 s previews (`musdb.DB(download=True)`; its terms are educational and non-commercial, so it serves measurement only), SDR by [museval](https://github.com/sigsep/sigsep-mus-eval) (BSSEval v4, 1 s windows, median over windows, then over tracks: the SiSEC 2018 aggregation):

| | vocals | drums | bass | other | this package vs Python, waveform SNR per track (median / lowest) |
|---|---|---|---|---|---|
| `umxhq`, open-unmix `Separator` | 6.75 | 6.11 | 5.00 | 3.36 | |
| `umxhq`, this package | 6.75 | 6.11 | 5.00 | 3.36 | 116–125 / 72–83 dB |
| `htdemucs`, demucs `apply_model` | 8.86 | 9.55 | 9.30 | 5.69 | |
| `htdemucs`, this package | 8.86 | 9.55 | 9.30 | 5.69 | 78–88 / 56–75 dB |

Per track, the SDRs differ by at most 0.003 dB.

## Measured against iZotope RX 12

RX 12 Advanced Music Rebalance (VST3 hosted by Pedalboard; [`audio`](https://github.com/audiojs/audio)'s `bench/rx/separate.mjs`, October 2026) beside this package on the same 50 MUSDB18 test previews, the mixture stream in, the stems as references. RX at each quality from its defaults (sensitivity 50 %), a stem soloed (`vocal_solo` and the like); ours at its defaults. Stems: BSSEval v4 SDR as above (museval 0.4.1), the median over tracks, then the songs where each is ahead (paired). Remixes, RX Music Rebalance's own use: vocals 6 dB up, vocals 6 dB down, drums 6 dB down, against the true remix (the mixture with that stem's own part scaled, so the coding difference between MUSDB18's mixture and its stems, 22 to 31 dB, stays as it is), the same SDR of one source; RX's remix renders equal the input plus (g − 1) times its solo to 126 dB and more, so its remixes are made from its solos. Harm: every gain 0 dB, the output against the input. RTF: one render over the preview's 6.8 s.

| 50 songs | vocals | drums | bass | other | vocals +6 | vocals −6 | drums −6 | harm | RTF |
|---|---|---|---|---|---|---|---|---|---|
| the mixture as it is | | | | | 8.39 | 10.99 | 10.67 | | |
| RX 12, Good / Real-time | 6.68 | 6.52 | 4.43 | 3.15 | 15.58 | 18.60 | 17.50 | 128.7 dB | 0.08 |
| RX 12, Better / Offline | 9.29 | 8.93 | 6.89 | 5.44 | 18.53 | 20.96 | 19.98 | 130.2 dB | 0.53 |
| RX 12, Best / Offline | **10.89** | 9.85 | **9.66** | 6.45 | 19.46 | **22.26** | 20.85 | 130.2 dB | 2.48 |
| `umxhq` | 6.74 | 6.11 | 5.00 | 3.36 | 15.09 | 18.35 | 17.32 | bit for bit | 0.16 |
| `htdemucs` | 8.86 | 9.55 | 9.30 | 5.69 | 17.99 | 20.76 | 20.76 | bit for bit | 1.74 |
| `htdemucs_ft` | 8.67 | 9.45 | 9.68 | 5.58 | 17.59 | 20.33 | 21.08 | bit for bit | 4.70 |
| `scnet` | 9.88 | 9.43 | 8.35 | 6.15 | 18.23 | 21.05 | 20.91 | bit for bit | 0.45 |
| `scnet-large` | **11.00** | **10.27** | 8.21 | **6.87** | **20.20** | 22.11 | **21.88** | bit for bit | 0.96 |
| … ahead of RX Best on | 33 of 50 | 40 | 34 | 32 | 34 | 30 | 41 | | |
| … paired median over RX Best | +0.31 | +0.68 | +0.97 | +0.33 | +0.31 | +0.18 | +0.68 | | |

Our remixes are the input plus (g − 1) times the stems, as `audio`'s `rebalance` makes them, so every gain at 0 dB returns the input (RX's, to 128 dB); `rebalance` itself (its solo is the input less the other three stems) scores 10.75 · 10.30 · 8.17 · 6.94 and the same remixes. Against `htdemucs`, the best model here before, RX at Best was ahead on 46 of the 50 songs for vocals (1.62 dB, the paired median). On vocals `scnet-large`'s median is 0.11 dB over RX's and `rebalance`'s 0.14 under it (ahead on 33 songs either way, the means 10.29 and 10.16 against 9.99). On bass it is 1.5 dB under RX's median while ahead on 34 songs, the means level (8.68 against 8.67): its bass falls under 3 dB on 8 songs (RX's on 6), mostly electronic, where it hears the synth bass as other (PR - Happy Daze: none of it, 0.0 dB, against RX's 18.3); SCNet trained on MUSDB18's 100 songs alone, RX on iZotope's own. Averaging `scnet` in (vocals 10.56, bass 8.21), test-time augmentation (the channels swapped, the polarity inverted: 3× the time, +0.01 to +0.02 dB paired) and multichannel Wiener EM over its stems (−0.5 to −2.4 dB on 13 of the songs: it takes the mixture's phase where SCNet estimated its own) were tried and left out. The previews are 7 s, one 11 s segment each: on whole songs each sample is heard in four.

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
| `model` | — | required. Preset `'scnet-large' \| 'scnet' \| 'umxhq' \| 'htdemucs' \| 'htdemucs_ft' \| 'mrx' \| 'tiger'` · `url \| bytes` (single target, named `'stem'`) · `{ target: url \| { url, targets }, ... }` (one graph per target, Open-Unmix's own layout; a multi-source graph contributes its own target) · `{ url, targets: [...] }` (one multi-target graph, stacks a target axis) |
| `targets` | all | the targets to return |
| `weights` | `$AUDIO_NEURAL_CACHE` or `~/.cache/audiojs/neural` (Node); a compact preset's hosted file (`REVISIONS`) | where a preset's files are: URL, or a directory in Node |
| `modelType` | `'openunmix'` | `'openunmix'` (magnitude out) · `'mask'` (`[0,1]` mask out, multiplied by mixture magnitude) · `'hybrid'` (demucs.onnx contract) · `'complex'` (complex spectrogram in and out, SCNet's) · `'multires'` (magnitudes at several resolutions in, masks out, MRX's) · `'waveform'` (Demucs v2-class); presets know theirs |
| `wiener` | `1` | EM iterations; `0` = raw masks. Ignored for `'hybrid'` and `'waveform'` |
| `wienerWindow` | `300` | frames per EM window |
| `chunk` / `overlap` | `30` / `2` (seconds; `mrx` 20) | `overlap` must be `< chunk`; not used by `'hybrid'`, which segments as demucs does |
| `segment` | `343980` | `'hybrid'`: segment length in samples when the graph does not declare its input length; `'complex'`: the preset's (485100) |
| `step` | `0.25` | `'complex'`: segments start every `step` of a segment (each sample heard in 1/`step` of them; `tiger` 1/3) |
| `fade` / `pad` / `standardize` | `0.1` / `'reflect'` / `true` | `'complex'`: each segment's linear fade, a fraction of it (`tiger` 0); the ends reflected or zero-padded (`tiger` `'zero'`); the input normalized by its mean and deviation (`tiger` not) |
| `window` / `normalized` / `frames` | the preset's | `'complex'`: the model's STFT window (`'ones'`, torch.stft's window=None, or `'hann'`), its 1/√n scaling, its frames per segment when the graph does not declare them |
| `targetRate` | the preset's rate, else the input rate | resample to the model's rate for inference; stems are resampled back to the input rate |
| `device` | — | passed through to `@audio/neural-runtime`'s `load()` as `backend` |
| `dtype` | `'float32'` | only `'float32'` tensor marshalling is implemented; anything else throws |
| `progress` | — | `({ chunk, totalChunks }) => {}` |
| `session` | — | overrides `@audio/neural-runtime`'s `load()` — for tests, or a custom ORT setup |

Mono input is duplicated to stereo internally (matching `openunmix.utils.preprocess`'s own "if we have mono, we duplicate it to get stereo"), so stems come back stereo; `mrx` and `tiger` hear each channel apart and give mono back mono. `separate()` returns `{ stems, sampleRate, residual }` — `residual` is the mixture minus the sum of all stems, per channel, at the input rate.

## Precedence and licenses

Open-source stem separation has these lineages; this package runs SCNet, Open-Unmix and Demucs.

| Project | Code | Weights | |
|---|---|---|---|
| **Open-Unmix** (`umx`, `umxhq`) | MIT | MIT ([Zenodo](https://zenodo.org/records/3370489)-declared) | trained on MUSDB18(-HQ); this package's primary target |
| **Open-Unmix** (`umxl`) | MIT | **CC BY-NC-SA 4.0 — non-commercial only** | despite being the `openunmix` package's own default variant name; trained on a private stems dataset (see the project [README](https://github.com/sigsep/open-unmix-pytorch#pre-trained-models)) |
| **Demucs** (Meta; `htdemucs`, `htdemucs_ft`) | MIT | **research only**: "The model weights are not covered by the MIT license, and are provided only for scientific purposes" (the author in [#327](https://github.com/facebookresearch/demucs/issues/327), 2022; again in [#508](https://github.com/facebookresearch/demucs/issues/508), 2023; the Hugging Face copies he published in 2026 had their `license: mit` tag removed) | trained on MUSDB18-HQ plus 800 songs; the repository is archived, maintained at [adefossez/demucs](https://github.com/adefossez/demucs) |
| **SCNet** (`scnet`, `scnet-large`; starrytong) | MIT | MIT: "I confirm that the released SCNet and SCNet-large pretrained weights are distributed under the MIT License, consistent with the source code. You are welcome to redistribute the original checkpoints and format-converted versions, including ONNX exports, as part of your MIT-licensed tool, with appropriate attribution." (the author in [starrytong/SCNet#35](https://github.com/starrytong/SCNet/issues/35), 2026) | trained on MUSDB18-HQ; hosted with its configs by [ZFTurbo/Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training) (MIT), whose own MUSDB18-HQ checkpoints (SCNet XL, BS-RoFormer) carry its author's MIT statement only for some files ([#254](https://github.com/ZFTurbo/Music-Source-Separation-Training/issues/254)) |
| **MRX** (MERL; `mrx`) | MIT | MIT: the repository's [`.reuse/dep5`](https://github.com/merlresearch/cocktail-fork-separation/blob/main/.reuse/dep5) names the four checkpoints, "Copyright: 2023 Mitsubishi Electric Research Laboratories (MERL) / License: MIT" | trained on Divide and Remaster v2, whose music and effects come from FMA and FSD50K clips each under its own licence, some non-commercial (the DnR v3 paper's account) |
| **TIGER** (Tsinghua; `tiger`) | MIT (repository LICENSE; its README badge says Apache 2.0) | Apache 2.0: the [model card](https://huggingface.co/JusperLee/TIGER-DnR) states `license: apache-2.0` | trained on Divide and Remaster v1 (652 test mixtures in its paper), sourced as v2 is |
| **Bandit v2** (DnR v3) | Apache 2.0 | **CC BY-SA 4.0** ([Zenodo](https://zenodo.org/records/12701995)): a converted copy is an adaptation under the same terms | not shipped; BandIt v1's DnR checkpoints are CC BY-NC 4.0 ([Zenodo](https://zenodo.org/records/10160698)) |
| **Spleeter** (Deezer) | MIT | **undocumented** | the README licenses only "the code of Spleeter"; the pretrained weights' license is an open, unresolved question ([deezer/spleeter#898](https://github.com/deezer/spleeter/issues/898)) — do not assume MIT |

Audit any weight source yourself before shipping it — this table reflects what each project states as of this writing, not a guarantee. `scripts/export-openunmix.py` defaults to `umxhq` (not `umxl`) for exactly this reason. MUSDB18, which every model here trained on, is licensed for educational use; whether that reaches the weights no project has settled (Défossez in [demucs#384](https://github.com/facebookresearch/demucs/issues/384): "a grey area").

## Reference

Stöter, Uhlich, Liutkus, Mitsufuji, "Open-Unmix - A Reference Implementation for Music Source Separation," *JOSS* 4(41), 2019. · Rouard, Massa, Défossez, "Hybrid Transformers for Music Source Separation," *ICASSP* 2023. · Xu, Li, Chen, Hu, "TIGER: Time-frequency Interleaved Gain Extraction and Reconstruction for Efficient Speech Separation," *ICLR* 2025. · Petermann, Wichern, Wang, Le Roux, "The Cocktail Fork Problem: Three-Stem Audio Separation for Real-World Soundtracks," *ICASSP* 2022. · Watcharasupat, Wu, Orife, "Remastering Divide and Remaster: A Cinematic Audio Source Separation Dataset with Multilingual Support," 2024. · Tong, Zhu, Chen, Kang, Jiang, Li, Wu, Meng, "SCNet: Sparse Compression Network for Music Source Separation," *ICASSP* 2024 (arXiv:2401.13276). · Duong, Vincent, Gribonval, "Under-determined reverberant audio source separation using a full-rank spatial covariance model," *IEEE TASLP* 18(7), 2010. · Rafii, Liutkus, Stöter, Mimilakis, Bittner, "The MUSDB18 corpus for music separation," 2017. · [norbert](https://github.com/sigsep/norbert) (Liutkus & Stöter) · [open-unmix-pytorch](https://github.com/sigsep/open-unmix-pytorch) · [Demucs](https://github.com/facebookresearch/demucs) · [demucs.onnx](https://github.com/sevagh/demucs.onnx) · [SCNet](https://github.com/starrytong/SCNet) · [cocktail-fork-separation](https://github.com/merlresearch/cocktail-fork-separation) · [TIGER](https://github.com/JusperLee/TIGER) · [pyloudnorm](https://github.com/csteinmetz1/pyloudnorm) · [Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training) · [museval](https://github.com/sigsep/sigsep-mus-eval) · [Spleeter](https://github.com/deezer/spleeter).

**Use when:** you have (or can license) an ONNX-exported spectrogram-mask, Hybrid-Demucs or waveform separation model and want to run it — with proper multichannel Wiener refinement, not just the raw mask — dependency-free, in Node or the browser; `scnet-large` for the highest SDR here under MIT weights, `scnet` for a quarter of its size and a third of its time, `htdemucs` under research-only terms.<br>
**Not for:** the classical, model-free case — reach for [`@audio/vocals`](https://github.com/audiojs/vocals) when a center-panned M/S trick is all you need; training a model (this is inference-only); real-time streaming (none of Open-Unmix's bi-LSTM, Hybrid Transformer Demucs or SCNet's dual-path LSTMs is causal: offline and chunked only, as upstream).

---

Part of the [@audio/neural](https://github.com/audiojs/neural) lane.

MIT © [audiojs](https://github.com/audiojs)
