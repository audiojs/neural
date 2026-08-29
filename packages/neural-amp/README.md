# @audio/neural-amp

> NAM playback: run Neural Amp Modeler `.nam` captures in plain JS, no dependencies.

Parses and runs the classic standard WaveNet, the architecture behind the ToneHunt ecosystem's standard/lite/feather captures (thousands of free amp models). Offline full-buffer inference; the streaming/AudioWorklet build belongs to `@audio/neural-runtime`.

```js
import nam from '@audio/neural-amp'

let amp = nam(await (await fetch('5150.nam')).text())
amp(guitarDI)            // Float32Array, processed in place
amp.receptiveField       // 4093: warmup transient length in samples
amp.sampleRate           // capture's native rate (null on older files)
```

**Format**: `.nam` JSON, `architecture: "WaveNet"`, ungated Tanh/ReLU layers. The weight order (per layer array: rechannel, then per layer conv+bias / condition mixin / 1×1 residual, then head rechannel; `head_scale` as the final float) is verified two ways in the suite: a hand-computed micro-model pins the math and the causal alignment, and the shipped example model parses with exact weight-count consumption. Unsupported variants (gated, FiLM, custom heads, LSTM) throw rather than guess.

**Verified behavior** (real Peavey 5150 capture, fetched fixture): steady-state silence −73 dB after the warmup transient, ~24% harmonic content on a −12 dBFS sine, ~1.4× realtime in plain JS at 48 kHz.

**Weights policy**: no trained captures in the repo; the committed fixture is the MIT-licensed random-weights example from NeuralAmpModelerCore; real captures are fetched on demand (`fixtures/README.md`).

**Use when:** playing back existing `.nam` captures, generating reference wets, or black-box-validating `@audio/neural-capture` against real amps.<br>
**Not for:** real-time worklet playback yet (`neural-runtime`), or training captures (`neural-capture` is the formula lane; NAM's Python trainer is the weights lane).

---

Part of the [@audio/neural](https://github.com/audiojs/neural) lane.

MIT © [audiojs](https://github.com/audiojs)
