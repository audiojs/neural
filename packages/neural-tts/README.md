# @audio/neural-tts

> Text to speech, running locally — no API key, no network round-trip per sentence.

Wraps [`@huggingface/transformers`](https://github.com/huggingface/transformers.js) (Apache-2.0) / ONNX Runtime, the same transformers.js pattern as [`@audio/neural-asr`](https://github.com/audiojs/neural/tree/main/packages/neural-asr), [`@audio/neural-align`](https://github.com/audiojs/neural/tree/main/packages/neural-align), [`@audio/neural-diarize`](https://github.com/audiojs/neural/tree/main/packages/neural-diarize). Default model SpeechT5 (Microsoft, MIT): an autoregressive text→mel-spectrogram model paired with a HiFi-GAN vocoder and a speaker x-vector that selects the voice.

```js
import tts from '@audio/neural-tts'

let { channelData, sampleRate, duration } = await tts('Hare Krishna. The quick brown fox jumps over the lazy dog.')
// channelData: [Float32Array], sampleRate: 16000, duration: ~4.1
```

```js
import { loadModel } from '@audio/neural-tts'

let net = await loadModel()                 // keep the model warm across calls
let r = await net.speak('Some longer text.', { sampleRate: 44100, voice: myXVector })
await net.free()
```

## Model / license audit

| Model | Architecture | License | Default | Voice |
|---|---|---|---|---|
| `Xenova/speecht5_tts` | SpeechT5 + HiFi-GAN | **MIT** | **yes** | any 512-d x-vector; default CMU ARCTIC "slt" |
| `Xenova/mms-tts-eng` | VITS | **CC-BY-NC-4.0 — non-commercial only** | no — pass `model` explicitly | single built-in voice, no speaker control |
| `onnx-community/Kokoro-82M-v1.0-ONNX` | StyleTTS2-class, 82M | Apache-2.0 | **not wired** — documented alternative only | multiple |

SpeechT5: [Ao et al. 2022, "SpeechT5: Unified-Modal Encoder-Decoder Pre-Training for Spoken Language Processing"](https://arxiv.org/abs/2110.07205) (Microsoft, MIT, confirmed via the [original repo's license](https://github.com/microsoft/SpeechT5)); HiFi-GAN vocoder: [Kong et al. 2020](https://arxiv.org/abs/2010.05646). Default voice: the CMU ARCTIC "slt" speaker x-vector at `https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/speaker_embeddings.bin` — the embedding transformers.js's own SpeechT5 docs and examples default to (derived from the `Matthijs/cmu-arctic-xvectors` dataset's `cmu_us_slt_arctic` recording).

`Xenova/mms-tts-eng` (VITS, [Kim et al. 2021](https://arxiv.org/abs/2106.06103)) wraps Meta's MMS English checkpoint — **CC-BY-NC-4.0**, confirmed via the [HF model card metadata](https://huggingface.co/facebook/mms-tts-eng): non-commercial use only. Not the default; pass `model: 'Xenova/mms-tts-eng'` explicitly, and only if that license fits your use. It has one built-in voice — `opts.voice` is ignored for this model.

**Kokoro** (`onnx-community/Kokoro-82M-v1.0-ONNX`, Apache-2.0, confirmed via HF metadata) is listed but not implemented: its own `config.json` reports `model_type: "style_text_to_speech_2"`, not a task transformers.js's `AutoTokenizer` phonemizes for — the real Kokoro pipeline needs [`kokoro-js`](https://github.com/hexgrad/kokoro) 's own espeak-based grapheme-to-phoneme step, which is a separate dependency from plain `@huggingface/transformers`. Per this package's own scope (transformers.js only), Kokoro is a documented alternative, not a supported `model` value here.

## Options

| Option | Default | |
|---|---|---|
| `model` | `'Xenova/speecht5_tts'` | see the model table |
| `voice` | CMU ARCTIC "slt" | SpeechT5 only: 512-d x-vector `Float32Array`, a URL to fetch one from, or a raw `Tensor`. Ignored for VITS |
| `vocoder` | `'Xenova/speecht5_hifigan'` | SpeechT5 only |
| `sampleRate` | the model's native rate (16000) | resampled afterwards with `@audio/resample-sinc` |
| `maxChars` | `600` | max characters per model call before a further word-boundary split |
| `silenceGap` | `150` (ms) | silence inserted between synthesized sentences |
| `dtype` | `'fp32'` (Node) / `'q8'` (browser) | passed to `from_pretrained` |
| `device` | unset (library's own default) | see *Known issues* — do not pass `'auto'` |
| `cache` | see Cache below | Node only: override the model cache directory for this call |
| `progress` | — | model download/load progress, plus one call per synthesized sentence |

## Cache

The default model is ~630 MB (SpeechT5 ~576 MB + HiFi-GAN ~54 MB, fp32 ONNX); `Xenova/mms-tts-eng` is ~112 MB. In Node, `tts.js` points `@huggingface/transformers`' own cache at the lane's shared cache root:

- `$AUDIO_NEURAL_CACHE/hf` if `$AUDIO_NEURAL_CACHE` is set — the same root [`@audio/neural-runtime`](https://github.com/audiojs/neural/tree/main/packages/neural-runtime) uses for its raw ONNX bytes.
- `~/.cache/audiojs/neural/hf` otherwise.

Set once at module load, so it applies before any model is fetched. Override per call with `opts.cache`. In the browser, caching is untouched (Cache API default).

```js
import { loadModel } from '@audio/neural-tts'
await loadModel('Xenova/speecht5_tts')   // populates the cache; no text needed
```

## Known issues (this transformers.js version, this environment)

Two problems were found and worked around while building this package — both reproduced outside `tst`, with minimal repro scripts, against `@huggingface/transformers@4.2.0` + `onnxruntime-node` on this machine:

1. **`pipeline('text-to-speech', 'Xenova/speecht5_tts')` throws.** Calling the high-level convenience pipeline (the form the brief for this package started from) fails with `"Missing the following inputs: speaker_embeddings, encoder_hidden_states, output_sequence"` inside `generate_speech`'s decoder loop. The lower-level, JSDoc-documented call (`AutoTokenizer` + `SpeechT5ForTextToSpeech` + `SpeechT5HifiGan`, called directly) does not hit this — that's what this package uses.
2. **`device: 'auto'` passed explicitly to `from_pretrained()` corrupts a later call.** With `device: 'auto'` set (this lane's usual default — `@audio/neural-asr` and `@audio/neural-diarize` both use it safely), the *first* `speak()` call's *second* sentence fails deep inside ONNX Runtime: `"Non-zero status code ... Invalid dimension of 18446744073709551615 for SizeToDimension"` in a `MatMul`. Omitting `device` entirely (transformers.js's own ambient default, `cpu` in Node) does not hit this, nor does passing `device: 'cpu'` explicitly. This package therefore only forwards `device` to `from_pretrained` when the caller passes one explicitly — never `'auto'` — and defaults to nothing. Root cause not fully diagnosed (plausibly an execution-provider resolution path specific to constructing three raw sessions — SpeechT5's encoder, its merged decoder, and the vocoder — outside the `pipeline()` wrapper); if you need GPU/WebGPU in the browser pass `device: 'webgpu'` explicitly, which is unaffected.

## Accuracy / quality

Measured on `"Hare Krishna. The quick brown fox jumps over the lazy dog."` (58 chars, 2 sentences), fp32, Node CPU EP, warm model:

- Native output: 16000 Hz, ~4.1 s, peak ≈ 0.28–0.31, RMS ≈ 0.04 — non-silent, well within `[-1, 1]`. ~0.9–2.1 s wall time for the full 2-sentence utterance.
- `sampleRate: 44100` resamples correctly (`duration ≈ channelData.length / 44100` to float precision) — see *Determinism* for why this isn't compared byte-for-byte against a separate native-rate call.
- 2000+ character input (45 repetitions of a sentence, 2025 chars): splits into 45 sentence chunks (one `progress` call each), total duration 141 s vs 6.9 s for the first 100 characters alone — chunking and silence-joining both do real work at that length.
- `Xenova/mms-tts-eng` (VITS): also verified — non-silent, native 16 kHz, no speaker embedding required, single `tokenizer → model()` call (no autoregressive loop, no vocoder).

**Determinism**: the brief's claim "the SpeechT5 vocoder is deterministic" is true, and tested directly — HiFi-GAN given the *same* fixed mel-spectrogram tensor reproduces the exact same waveform, byte for byte, across repeated calls (no autoregression, no internal state). The **full pipeline is not** bit-deterministic end to end: SpeechT5's decoder feeds its own previous output back into itself at every step, so the small floating-point differences ONNX Runtime's multi-threaded CPU execution can introduce between two runs of the same matrix multiplications compound across iterations. Measured on identical repeated input: output length varied by 0–3072 samples (0–4.7%) across separate `test.js` runs, occasionally landing on 0 (bit-identical) and other times not — consistent with thread-scheduling-dependent floating-point summation order, not a bug in this package. `test.js` asserts the bound it actually measured (< 10% length delta, both runs non-silent) rather than a false claim of sample-level identity.

## Long text

Text is split into ≤`maxChars` chunks — first at sentence-ending punctuation (`.`/`!`/`?`), and any chunk still over the limit further at word boundaries (never mid-word) — synthesized sequentially, and joined with `silenceGap` ms of silence between chunks (none before the first or after the last). `progress` fires once per chunk. SpeechT5 is autoregressive and was trained on individual utterances; splitting at sentence boundaries keeps every model call in the regime it was trained on rather than letting one `generate_speech` run drift over an entire paragraph.

## Limitations

- **English only** for both wired models (SpeechT5's released checkpoint, and `mms-tts-eng`). Other MMS-TTS per-language checkpoints exist on the Hub and may work by passing a different `model` id (untested here) — check that checkpoint's own license before using it (MMS models are commonly CC-BY-NC).
- **Robotic prosody.** SpeechT5 is a 2022 academic TTS model, not a commercial-grade one — flatter intonation and less natural phrasing than e.g. ElevenLabs or Azure/Google Cloud TTS. Fine for narration/notifications, not for a voice a listener is meant to mistake for human.
- **VITS (`mms-tts-eng`) has no voice control** — `opts.voice` is silently ignored (documented, not thrown) since the checkpoint has exactly one built-in voice.
- **Kokoro is not wired** (see *Model / license audit* above) — listed for completeness, not usable via `model:`.
- Per-sentence synthesis is sequential, not batched — long texts pay the full per-chunk model latency serially. `progress` at least lets a caller show incremental completion.

**Use when:** offline/local narration, notifications, accessibility read-aloud, or any pipeline that already accepts "good enough, no API key" TTS.<br>
**Not for:** natural-sounding conversational voice, non-English text, or anything requiring bit-reproducible output byte-for-byte across runs.

---

Part of the [@audio/neural](https://github.com/audiojs/neural) lane.

MIT © [audiojs](https://github.com/audiojs)
