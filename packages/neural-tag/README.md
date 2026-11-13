# @audio/neural-tag

> What a sound is, in AudioSet's 521 classes: Google's YAMNet scores each 0.96 s of it, its features computed in JS as upstream computes them.

YAMNet (tensorflow/models, `research/audioset/yamnet`, Apache-2.0) is a MobileNetV1 trained on AudioSet: speech, music
and its genres and instruments, animals, vehicles, tools, rooms, silence, noise. It scores every 0.96 s patch of a sound
for each of 521 classes, independently (sigmoids, not a softmax: a patch can be Speech and Music at once). This package
runs it in Node and in a browser through `@audio/neural-runtime`: the log-mel patches computed here, the network as
ONNX. It names what is heard, in AudioSet's words; it does not decide what kind of recording that makes, and its scores
are not probabilities you can threshold alike across classes (below).

```
npm install @audio/neural-tag onnxruntime-node   # onnxruntime-web in the browser
```

```js
import { tag, scores, CLASSES } from '@audio/neural-tag'

let tags = await tag(samples, 44100)           // [{ name: 'Speech', id: '/m/09x0r', index: 0, score: 0.60 }, …] 5 best
let all = await tag([left, right], 48000, { top: Infinity })   // channels mixed to their mean; all 521, ranked
let { scores: s, hop, times } = await scores(samples, 44100)   // a Float32Array(521) per patch, every 0.48 s
let speech = s.map(row => row[0])                               // CLASSES[0] is Speech: one class over time
```

`tag` ranks the classes by their mean score over the sound's patches, as upstream's `inference.py` does. `scores`
gives every patch, for a caller that segments or thresholds itself: patch `i` hears 0.975 s from `times[i]`.

The model file is hosted at [audiojs/yamnet](https://huggingface.co/audiojs/yamnet) (`REVISION` `32997c9`): without
`opts.weights` it is fetched once (7.5 MB), refused unless of the SHA-256 `SHA256`, and in Node kept in the neural cache
(`~/.cache/audiojs/neural/yamnet/`, where `scripts/export.py` writes it too) and read from there after, offline as
well. `opts.weights` reads another copy: a URL, a path in Node, or the bytes of `yamnet.fp16.onnx`.

## What it knows, and how well

The vocabulary is AudioSet's ontology less six classes upstream dropped "on the recommendation of our Fairness
reviewers": the gendered Speech and Singing, Battle cry, Funny music. Upstream's figures ([README](https://github.com/tensorflow/models/blob/c14bf9ad91962cf189f9f58db2132c06247fcd53/research/audioset/yamnet/README.md),
Performance): on AudioSet's 20,366-segment evaluation set, over the 521 classes, balanced average d-prime 2.318,
balanced mAP 0.306, balanced average lwlrap 0.393. Its model card ([TF Hub google/yamnet/1](https://github.com/tensorflow/tfhub.dev/blob/master/assets/docs/google/models/yamnet/1.md),
Limitations): "YAMNet's classifier outputs have not been calibrated across classes, so you cannot directly treat the
outputs as probabilities", and its training on YouTube may not match a task's audio; "you should expect to do some
amount of fine-tuning and calibration".

What the original Keras model says of this package's test signals (`fixtures/reference.json`; this package says the
same, to the scores below):

| Signal (16 kHz) | Patches | Keras's top 3, mean score |
|---|---|---|
| lena, 12.3 s ([audio-lena](https://github.com/audiojs/audio-lena): Lena Stolze, *Das schreckliche Mädchen*) | 25 | Speech 0.602, Crying, sobbing 0.062, Whimper 0.057 |
| eight plucked notes (Karplus–Strong), 3 s | 6 | Music 0.507, Cowbell 0.144, Musical instrument 0.106 |
| a tone swept 600–1200 Hz and back, with its third harmonic, 3 s | 6 | Synthesizer 0.667, Keyboard (musical) 0.549, Music 0.322 |
| white noise, 3 s | 6 | White noise 0.694, Noise 0.528, Static 0.362 |
| a 1 kHz beep, 0.5 s | 1 | Beep, bleep 0.973, Sound effect 0.052, Sine wave 0.019 |
| digital silence, 2 s; and nothing at all | 4; 1 | Silence 1.000 |

As a frame-by-frame speech/music judge, `@audio/neural-denoise` measured it against inaSpeechSegmenter (its README,
Music): YAMNet passed songs and accompaniment as music every time, and called speech over a music bed 10, 15, 20 dB
down music 29, 9.6, 0.9% of the time (inaSpeechSegmenter 2.2, 1.2, 0.4%).

## How it works

As `features.py` and `params.py` define it, at 16 kHz:

- **Padding** (`pad_waveform`): a sound shorter than 0.975 s (15,600 samples, one patch's windows) is padded with zeros
  to one patch; a longer one with zeros to end on a whole hop. Patches: 1 + ⌈max(0, L − 15600) / 7680⌉. So a 0.5 s
  sound is one patch, half of it zeros, and an empty input is one patch of silence, which YAMNet scores Silence 1.000.
- **Spectrogram** (`tf.signal.stft`): frames of 400 samples (25 ms) every 160 (10 ms), not centered, a periodic Hann
  window, zero-padded to 512, magnitudes of 257 bins.
- **Mel** (`tf.signal.linear_to_mel_weight_matrix`): 64 bands from 125 to 7500 Hz, HTK mels (1127 ln(1 + f/700)),
  each band a triangle in mels over the bins above DC, unnormalized; then ln(mel + 0.001).
- **Patches** (`tf.signal.frame`): 96 frames × 64 bands every 48 frames (0.96 s every 0.48 s); `opts.hop` sets the step
  (params.py: "The patch hop can be changed arbitrarily").
- **Network** (`yamnet.py`): a 3 × 3 convolution with stride 2 and 32 filters, then 13 depthwise-separable convolutions
  (64 to 1024 filters, strides 1 and 2), each with a batch norm and a ReLU, 'same' padding; a 1024 × 3 × 2 map averaged
  to 1024 values; a dense layer and a sigmoid: 521 scores. 3,729,481 parameters once the batch norms are folded in
  (upstream: "3.7M weights", "69.2M multiplies for each 960ms input frame").

What this package does differently, on purpose:

- The features are computed in float64 (upstream: float32). They are within 2.4e-7 of upstream's own ops run in
  float64; upstream's float32 run strays from that by up to 6.1e-5 on lena and up to 3.3e-3 on the synthetic tones,
  where a band near the log's floor magnifies float32's rounding (Verification).
- Other rates are resampled to 16 kHz by [`@audio/resample-sinc`](https://github.com/audiojs/resample) (`inference.py`
  uses resampy); channels are mixed to their mean first, as `inference.py` does.
- The patches are counted in integers. Upstream's `pad_waveform` divides in float32 and fails on some sounds longer
  than 2^24 samples (17.5 min at 16 kHz): 23,055,601 samples raise `InvalidArgumentError: Paddings must be
  non-negative`.
- `opts.hop` is taken in whole 10 ms frames, for the padding and the framing alike (upstream frames by round(hop / 10 ms)
  but pads by ⌊hop · 16000⌋ samples, which agree only on whole frames).
- The network's batch norms (center, no scale, epsilon 1e-4) are folded into the convolutions, and its weights stored
  as float16 (below).

## Verification

`node test.js`, against the fixtures `scripts/export.py --reference` writes from upstream's own code (tensorflow/models
at `c14bf9a`, TensorFlow 2.21, tf-keras 2.21, `yamnet.h5`) on the waveforms `scripts/inputs.mjs` makes, their SHA-256
checked first:

| Check | Result |
|---|---|
| Log-mel patches against `features.py` (float32), the patches the fixtures keep | lena 2.6e-5 max \|diff\| (first and last patch), white noise 2.2e-5, plucked notes 6.5e-4, sweep 2.9e-3, beep 1.1e-3 (last or only patch); silence and nothing: every cell ln(0.001) |
| The same, every patch, against upstream's ops in float64 (`tf.signal` with float64 tensors; measured once, by a script outside the package) | within 2.4e-7 on all five, the float32 storage of the result; upstream's float32 against its float64: lena 6.1e-5, noise 2.1e-5, plucked notes 6.5e-4, beep 1.1e-3, sweep 3.3e-3 |
| The ONNX network against the Keras network (`yamnet.yamnet`, `yamnet.h5` loaded) on the same 49 patches (`export.py --verify`) | float32 export: max 1.6e-6 (mean 1.5e-8); float16 weights: max 3.6e-3 (mean 3.0e-6); the top class the same in all 49 |
| Scores end to end (these features, the float16 file) against `yamnet_frames_model` on the same waveform | lena 2.3e-3, plucked notes 3.6e-3, sweep 1.3e-3, noise 7.0e-4, beep 1.2e-4, silence 6.3e-8, nothing 3.1e-9, lena with a 2 s hop 7.4e-4; Keras's top 3 in its order for each (its classes over 0.01) |
| The same with the float32 export | 6.5e-5 at most (the sweep); lena 5.4e-6 |
| 48 kHz in (lena resampled from 44.1) against 16 kHz | 1.4e-3 at most per patch; the same top 5 |
| 44.1 and 48 kHz against 16 kHz, signals made at each rate | sweep: the same top 5, mean scores within 2.6e-3 and 4.7e-5 (single patches up to 2.5e-2: its empty bands sit at the log's floor, where the resampler's 83 dB SNR shows); beep: within 1.3e-4 |
| Channels | two channels score as their mean; two equal channels as one |
| A browser: headless Chromium 151, onnxruntime-web 1.31.1 wasm (`opts.weights` a URL; measured once, by a script outside the package) | lena at 44.1 kHz: max \|score − Keras\| 2.3e-3, as in Node; the top 3 the same; without weights, the error |

The float16 file's 3.6e-3 is its weights' rounding spread over the 14 layers: rounded alone, no weight moves a score by
more than 1.3e-3, and keeping all the depthwise kernels in float32 still leaves 2.7e-3 (measured once on the 49 patches). The float32 export
(`yamnet.onnx`, 14.9 MB) is there for `opts.weights` where that matters.

## Speed

A minute of lena (repeated) at 44.1 kHz through `tag()`, onnxruntime-node 1.30, Node 25, an Apple M4 Max (14 cores)
running other jobs (load averages 16 to 17), the session already made, three runs each:

| Session | CPU per minute of audio | Wall |
|---|---|---|
| one thread (`sessionOptions: { intraOpNumThreads: 1 }`) | 0.49–0.54 s | 0.60–0.73 s |
| four threads | 0.72–0.75 s | 0.29–0.32 s |
| onnxruntime's default | 1.20–1.60 s | 0.36–0.50 s |

Of that, resampling 44.1 to 16 kHz takes 0.08–0.18 s and the features 0.03–0.05 s; the rest is the network, 124
patches. The float32 file runs as fast (1.43–1.51 s at the default). In the browser above, 12.3 s of lena took 0.17 s
once the session was made (2.6 s for the first call, onnxruntime-web's wasm and the model loaded). WebGPU is untested.

## The model

`scripts/export.py` fetches upstream's `yamnet.py`, `features.py`, `params.py`, `yamnet_class_map.csv` at
`c14bf9ad91962cf189f9f58db2132c06247fcd53` and `yamnet.h5` (SHA-256 `13c33089…1744`, 15,296,092 bytes), each checked,
and builds the ONNX graph from the weights' arrays (h5py and onnx, no TensorFlow in the export): `patches` [n, 96, 64]
float32 in, `scores` [n, 521] out, opset 17. In `yamnet.fp16.onnx` the weights of over 1024 values are stored as float16
and Cast to float32 in the graph, computed in float32 on every backend (`@audio/neural-separate`'s `compact.py` rule);
biases and the first three kernels (288, 288 and 576 values) stay float32.

| File | Size | SHA-256 |
|---|---|---|
| `yamnet.fp16.onnx` | 7,495,175 bytes | `0a1f3b050ebfdfa651e78fb12714c3c1fe750ceebd58529818a1ddddec515688` |
| `yamnet.onnx` (float32) | 14,927,064 bytes | `0cad16ae1e90e1d84722d1ea451f5118d1bd54b137b4096638728af93c045e83` |

`classes.js` is `yamnet_class_map.csv` in output order. To rebuild everything (from this directory, after `npm install`
at the repository's root):

```
uv run --python 3.12 --with tensorflow==2.21.0 --with tf-keras==2.21.0 --with h5py --with onnx --with onnxruntime \
    python -I scripts/export.py --verify --reference
```

## API

| Export | |
|---|---|
| `tag(x, sampleRate?, opts?)` | → `Promise<[{ name, id, index, score }]>`: the classes by mean score, best first, `opts.top` of them (5; `Infinity` for all) |
| `scores(x, sampleRate?, opts?)` | → `Promise<{ scores, hop, times }>`: a `Float32Array(521)` per patch, seconds between patches, each patch's start |
| `features(x, opts?)` | → `{ data, patches }`: the log-mel patches of 16 kHz mono samples, `Float32Array(patches × 96 × 64)` |
| `CLASSES` | the 521 `{ id, name }` in output order |
| `MODEL`, `REVISION`, `SHA256` | the hosted file's URL, its commit, its SHA-256 |
| `SAMPLE_RATE`, `FRAMES`, `BANDS` | 16000, 96, 64 |

`x`: a `Float32Array` (mono) or an array of channels; `sampleRate` defaults to 16000. `opts`: `{ hop = 0.48, weights,
backend, sessionOptions, progress, fetch }`, and `top` for `tag`. A session is made once per weights, backend and
session options, and reused.

## Licences

Checked 2026-11-12 against what each project states; a summary, not legal advice.

| Part | Licence | Source |
|---|---|---|
| YAMNet's code: `features.py` and `params.py` translated in `tag.js`, `yamnet.py` read by `export.py` | Apache-2.0 | tensorflow/models' [LICENSE](https://github.com/tensorflow/models/blob/c14bf9ad91962cf189f9f58db2132c06247fcd53/LICENSE) covers `/research`; the files' headers: "Copyright 2019 The TensorFlow Authors" |
| YAMNet's weights, `yamnet.h5`, and their conversion | Apache-2.0 | linked from the same repository's README; the same model on Kaggle (formerly TF Hub `google/yamnet/1`), whose card says it was "originally released in the TensorFlow Model Garden", states Apache 2.0 for every format. Fetched, never bundled |
| The class names and ids (`classes.js`) | the file: Apache-2.0; the names and ids: CC BY-SA 4.0 | `yamnet_class_map.csv` is in the same repository; the [AudioSet ontology](https://github.com/audioset/ontology) "is made available by Google Inc. under a Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0) license" |
| AudioSet, YAMNet's training data | CC BY 4.0 (the labels) | [research.google.com/audioset](https://research.google.com/audioset/download.html): the dataset under CC BY 4.0, the ontology CC BY-SA 4.0; the sounds are YouTube videos', not distributed |

The package is Apache-2.0; [NOTICE](./NOTICE) carries the attribution.

## Reference

J. F. Gemmeke, D. P. W. Ellis, D. Freedman, A. Jansen, W. Lawrence, R. C. Moore, M. Plakal, M. Ritter, "Audio Set: An
ontology and human-labeled dataset for audio events", ICASSP 2017. · A. G. Howard, M. Zhu, B. Chen, D. Kalenichenko,
W. Wang, T. Weyand, M. Andreetto, H. Adam, "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision
Applications", [arXiv:1704.04861](https://arxiv.org/abs/1704.04861), 2017. · [tensorflow/models](https://github.com/tensorflow/models/tree/c14bf9ad91962cf189f9f58db2132c06247fcd53/research/audioset/yamnet),
`research/audioset/yamnet` at c14bf9a (YAMNet by Manoj Plakal and Dan Ellis).

**Use when:** naming what a recording holds (speech, music, silence, noise, a dog, a siren) in Node or a browser, from
a 7.5 MB model, half a second of CPU per minute of audio on one thread.<br>
**Not for:** speech/music segmentation with tight boundaries (0.96 s patches; for speech over a music bed,
`@audio/neural-denoise`'s guard errs less); scores read as probabilities, or one threshold for all classes; sounds
outside AudioSet's vocabulary.

---

Part of the [@audio/neural](https://github.com/audiojs/neural) lane.

Apache-2.0 © 2026 audiojs; YAMNet © 2019 The TensorFlow Authors, Apache-2.0
