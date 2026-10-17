# @audio/neural-vocode

> A sound made again from its mel spectrogram by Vocos: a vocoder's artifacts give way to Vocos's own.

A text-to-speech model draws a mel spectrogram, and a vocoder turns it into sound. A cheap vocoder leaves its mark: each
frame's phase rebuilt slightly differently, faint copies of a frame 10, 20, 30 ms later, a voice that rings. Vocos
(Siuzdak, "Vocos: Closing the gap between time-domain and Fourier-based neural vocoders for high-quality audio
synthesis", ICLR 2024, [arXiv:2306.00814](https://arxiv.org/abs/2306.00814)) predicts each STFT frame's magnitude and
phase from the mel spectrogram afresh. Run over a finished voice, it replaces the faulty phase rather than filtering it.

```
npm install @audio/neural-vocode onnxruntime-node   # onnxruntime-web in the browser
```

```js
import vocode from '@audio/neural-vocode'

let y = await vocode(samples, { sampleRate: 44100 })   // as long, at 44.1 kHz; nothing above 12 kHz
```

In [`audio`](https://github.com/audiojs/audio): `a.vocos()` keeps the input above 11 kHz, where Vocos makes nothing.

## Measured

- **Against PyTorch.** On a 2 s fixture the output is 0.050 dB of log-mel distance from `Vocos.decode` (0.049 for the
  float32 graph). PyTorch is 0.047 dB from itself between float32 and float64 features, and Vocos changes the sound by
  3.7 dB. Sample by sample the two are 35 dB apart, as PyTorch's own float32 and float64 runs are: the phase Vocos
  predicts follows its features' rounding.
- **Each step.** The features are within 0.008 of torchaudio's log mel; torchaudio builds its filterbank in float32,
  7e-6 from the exact one. The iSTFT is 139 dB from `torch.istft` on the same frames. The ONNX graph is within 6e-5 of
  the PyTorch forward (`scripts/export.py --verify`). Chunked runs give the same frames as one run of all.
- **On a TTS take** (36 s, 44.1 kHz): its cepstrum's peaks at 20 and 30 ms, 0.020 and 0.010, fall to 0.0003 and 0.0005,
  the background's level. That takes 10 s on a CPU (onnxruntime-node, Apple M4 Max under load).

Vocos is a speech vocoder trained on LibriTTS. Music comes back altered. A tone of exact period comes back silent
from PyTorch's Vocos too: its spectrum's floor sits far under any recording's.

## The model

[charactr/vocos-mel-24khz](https://huggingface.co/charactr/vocos-mel-24khz) at
`0feb3fdd929bcd6649e0e7c5a688cf7dd012ef21` (MIT). Its config:
- 24 kHz;
- mel spectrogram with n_fft 1024, hop 256 and 100 HTK bands, magnitudes, centered with reflect padding, log clipped at 1e-7;
- ConvNeXt backbone with 8 layers, dim 512;
- iSTFT head.

`scripts/export.py` exports the backbone and the head's linear layer, the complex frames out (`re`, `im`: [1, 513, T]).
`compact.py` from `@audio/neural-separate` then stores the weights as float16, computed in float32: `vocos.fp16.onnx`,
27 MB (from 54), SHA-256 `54c5cb663571c9ace439e20fb43bce0b8948a992ee88eb8f7676d6ac19f376c0`.

Hosted at [audiojs/vocos-mel-24khz](https://huggingface.co/audiojs/vocos-mel-24khz) (`REVISION` `4f5fad0`): without
`opts.weights` it is fetched once, refused unless of that SHA-256, and in Node kept in the neural cache
(`~/.cache/audiojs/neural/vocos/`, where `export.py` writes it too) and read from there after, offline as well.

## API

| Export | |
|---|---|
| `vocode(x, opts?)` | → `Promise<Float32Array>`: mono `x` made again, at `opts.sampleRate` (24000), as long |
| `mel(x)` | → `{ data, frames }`: Vocos's features of 24 kHz samples, band-major |
| `frames(mel, opts?)` | → `Promise<{ re, im }>`: the frames Vocos predicts, bin-major, `opts.chunk` frames a run (1024) |
| `istft(re, im, frames, length?)` | → `Float32Array`: `torch.istft` of them, Hann of 1024, hop 256, centered |

`opts`: `{ sampleRate, weights, backend, chunk, progress, fetch }`.

## Licence and attribution

MIT. The weights by charactr (Hubert Siuzdak), MIT, trained on LibriTTS (CC BY 4.0); the architecture by Hubert
Siuzdak.

```bibtex
@inproceedings{siuzdak2024vocos,
  title     = {Vocos: Closing the gap between time-domain and Fourier-based neural vocoders for high-quality audio synthesis},
  author    = {Siuzdak, Hubert},
  booktitle = {ICLR},
  year      = {2024},
  eprint    = {2306.00814},
  archivePrefix = {arXiv}
}
```
