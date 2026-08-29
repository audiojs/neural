# @audio/neural-asr

> Speech-to-text: Whisper running locally, in Node and the browser, via transformers.js/ONNX Runtime.

```js
import transcribe from '@audio/neural-asr'

let { text, segments } = await transcribe(data, { sampleRate: 44100 })
```

[Whisper](https://cdn.openai.com/papers/whisper.pdf) (Radford, Kim, Xu, Brockman, McLeavey, Sutskever — "Robust Speech Recognition via Large-Scale Weak Supervision", OpenAI 2022) is a general-purpose speech recognizer trained on 680k hours of weakly-supervised multilingual audio. This package doesn't reimplement it: [`@huggingface/transformers`](https://github.com/huggingface/transformers.js) runs the same architecture through [ONNX Runtime](https://onnxruntime.ai/) — `onnxruntime-node` in Node, `onnxruntime-web` (wasm or WebGPU) in the browser — the path Xenova's `whisper-web` demo and transformers.js's own examples converged on. This atom mixes/resamples the caller's audio to Whisper's fixed 16 kHz mono input, drives the `automatic-speech-recognition` pipeline, and reshapes the result into `{start,end,text}` segments/words with 3-decimal-second timestamps.

```js
import transcribe, { loadModel } from '@audio/neural-asr'

let { text, language, segments } = await transcribe(pcm, { sampleRate: 44100 })

// word timestamps + subtitle cues
let { words, cues } = await transcribe(pcm, { sampleRate: 44100, timestamps: 'word', cues: true })

// reuse a warm model across many files
let model = await loadModel('onnx-community/whisper-small', { language: 'en' })
for (let file of files) console.log((await model.transcribe(file.pcm, { sampleRate: file.sr })).text)
await model.free()
```

| Option | Default | |
|---|---|---|
| `sampleRate` | — | required unless `audio` is `{channelData, sampleRate}` |
| `model` | `'onnx-community/whisper-base'` | any Hugging Face model id — see Models below |
| `language` | — | e.g. `'en'`, `'german'` — omitting it is **not** auto-detect *(see Limitations)* |
| `task` | `'transcribe'` | or `'translate'` (→ English) |
| `timestamps` | `'segment'` | `'segment'` \| `'word'` \| `false` |
| `chunk` / `stride` | `30` / `5` | long-audio windowing, seconds (Whisper's own 30 s training window) |
| `dtype` | `'fp32'` (Node) / `'q8'` (browser) | ONNX Runtime numeric precision — see Models |
| `device` | `'auto'` | `'cpu'` \| `'wasm'` \| `'webgpu'` \| … — transformers.js picks per environment |
| `progress` | — | `(info) => void`, model download + load progress |
| `cache` | — | override the on-disk model cache dir (Node) |
| `vad` | `false` | pre-cut long silences with `@audio/vad`, remap timestamps back |
| `cues` | `false` | also return `@audio/subtitle` `Cue[]` |

`audio`: `Float32Array` (mono, needs `sampleRate`), `Float32Array[]` (one per channel, needs `sampleRate`), or `{channelData: Float32Array[], sampleRate}` (a decoder atom's output shape — pass a `decode()` result straight through). Multi-channel input is downmixed to mono by equal-weight average before resampling.

## Models

```js
import { models } from '@audio/neural-asr'
```

| id | params | languages | size (q8) |
|---|---|---|---|
| `onnx-community/whisper-tiny.en` | 39M | en | 42 MB *(measured)* |
| `onnx-community/whisper-tiny` | 39M | multi | 42 MB |
| `onnx-community/whisper-base.en` | 74M | en | ~77 MB |
| `onnx-community/whisper-base` | 74M | multi | ~77 MB |
| `onnx-community/whisper-small` | 244M | multi | ~247 MB |
| `onnx-community/whisper-large-v3-turbo` | 809M | multi | ~812 MB |

All OpenAI Whisper weights, [MIT](https://github.com/openai/whisper/blob/main/LICENSE); converted to ONNX by the `onnx-community` org on the Hugging Face Hub (Apache-2.0 conversion tooling, same weights — [model card](https://huggingface.co/onnx-community/whisper-base)). Any other Whisper-architecture HF id works too (`opts.model` is not restricted to this table) — `Xenova/*` conversions are the historical predecessor of `onnx-community/*` and are still widely used; see Limitations for one place they currently behave differently (`test.js`'s real-inference suite deliberately uses `Xenova/whisper-tiny`, not this table's default org, precisely because it needs word timestamps to work).

`dtype` defaults to `'fp32'` in Node and `'q8'` in the browser, for different reasons: in Node the weights are already on local disk and `onnxruntime-node`'s CPU execution provider has full fp32 kernel coverage, so full precision costs nothing extra. In the browser the model ships over the wire on every first visit and `dtype` *is* what gets downloaded — q8's ~4× smaller transfer and its wasm/WebGPU-friendly int8 kernels matter more there than the last bit of accuracy. Pass `dtype` explicitly to override either default (`'auto' | 'fp32' | 'fp16' | 'q8' | 'int8' | 'q4' | 'q4f16' | 'bnb4'`, model-dependent).

## What runs where

- **Node**: `onnxruntime-node`, CPU execution provider. No GPU path in this package (`device` still accepts anything transformers.js supports, but WebGPU/CUDA/DirectML need their own runtime wiring this atom doesn't add).
- **Browser**: `onnxruntime-web`, `device: 'wasm'` everywhere, `'webgpu'` where available (`device: 'auto'`, the default, picks it up automatically). First run downloads and caches the model via the [Cache API](https://developer.mozilla.org/en-US/docs/Web/API/Cache); later runs are local.
- **Memory** (measured, tiny, q8, Node/CPU): `loadModel()` alone adds ~260 MB RSS; after a few `transcribe()` calls RSS plateaus around ~1.5 GB and stays flat (not a leak — `onnxruntime-node`'s CPU arena allocator grows to its working-set peak and reuses it, it doesn't shrink back down). Whisper always processes a fixed 30 s context window internally regardless of actual audio length (`chunk_length_s`), which is most of why this is larger than the weight size alone would suggest. Budget accordingly, especially for `small`/`large-v3-turbo` — this wasn't re-measured at every model size.

## Cache (Node)

Model files land under `~/.cache/audiojs/neural/hf/<repo>/…` — the `@audio/neural` lane's shared cache root (`$AUDIO_NEURAL_CACHE` overrides the base directory; also used by `@audio/neural-runtime` for raw ONNX bytes, in its own sha256-keyed layout — this package's `hf/` subdir holds the Hugging-Face-shaped tree `@huggingface/transformers` expects). Pass `opts.cache` to point a single call at a different directory. Not inside `node_modules` — it survives `npm install`/reinstall, so weights aren't re-downloaded on every dependency update. In the browser this doesn't apply; the Cache API is origin-scoped and untouched by this option.

## Limitations (read before shipping)

- **Silence and non-speech hallucinate.** Whisper has no explicit "no speech" output — fed digital silence or music it sometimes produces confident, fluent, *wrong* text instead of an empty string (a documented property of the architecture, not a bug in this wrapper). `opts.vad: true` sidesteps this for silence specifically (nothing to transcribe → empty result, no inference call at all) but not for music/noise that VAD classifies as speech-shaped.
- **Long, unfamiliar-to-the-model audio can loop.** transformers.js's ASR pipeline ships no anti-repetition generation defaults (`_default_generation_config = {}` upstream — OpenAI's own reference decoder gets this for free from temperature-fallback beam search, which transformers.js's greedy pipeline doesn't replicate). Left alone, this reliably degenerates into an infinite repeated-token loop on audio the model finds hard (wrong language forced, music, noise) instead of just failing. This package always passes `no_repeat_ngram_size: 3` to the underlying `generate()` call to block that failure mode; it does not fix low-confidence output, only the pathological loop.
- **`language` omitted is not the same as auto-detected.** transformers.js v4.2.0's public ASR pipeline does not perform language detection — when no `language` is given it silently assumes English (`generate()` logs "No language specified - defaulting to English"). For non-English audio, pass `language` explicitly; leaving it unset on a multilingual model is not neutral. `result.language` reflects `opts.language` when given, `'en'` for `.en`-suffixed (English-only) models, else `null` — never a guess.
- **Word timestamps (`timestamps: 'word'`) need cross-attention outputs the model was exported with.** `onnx-community`'s current `whisper-tiny`/`whisper-tiny.en` ONNX exports don't include them — `timestamps: 'word'` throws ("Model outputs must contain cross attentions…") regardless of `dtype`, a property of the exported graph, not a runtime setting. `Xenova/whisper-tiny(.en)` and `Xenova/whisper-tiny` do include them. Verify a model supports word timestamps before depending on it in production; this package doesn't probe for the capability ahead of time.
- **Word-timestamp accuracy is DTW-alignment-derived, not measured, ~±100 ms** in transformers.js's own documentation — occasionally worse: adjacent short function words can land with identical (zero-width) start/end. `segments` (the default) come from the model's own timestamp tokens and are more reliable.
- **`timestamps: 'word'` reconstructs `segments` heuristically.** Whisper's native segment boundaries come from timestamp tokens emitted only in the non-word decode path; with word timestamps there's nothing native to reuse, so `segments` are built by grouping words at sentence-ending punctuation (`.`, `?`, `!`). This is an approximation, not the model's own segmentation — expect it to differ from what `timestamps: 'segment'` returns for the same audio.

## `cues` — bridge to `@audio/subtitle`

```js
let { cues } = await transcribe(pcm, { sampleRate, timestamps: 'word', cues: true })
// → write(cues, 'srt') from @audio/subtitle
```

`@audio/subtitle` is an optional peer dependency (`npm install @audio/subtitle`), dynamically imported only when `opts.cues` is set — omitting it costs nothing until you ask for cues, and asking without it installed throws a clear "npm install @audio/subtitle" error rather than a bare `Cannot find module`. When word timestamps were requested, cues are built via `@audio/subtitle`'s `fromWords(words)`; otherwise (`timestamps: 'segment'` or `false`) cues are `segments` mapped straight to the `{start, end, text}` `Cue` shape, one cue per segment, no `words` sub-field — `fromWords` needs word-level data this package didn't ask for.

## Speed & size (measured)

tiny.en / tiny, q8, Node/CPU (this machine, `node test.js`): 42 MB download (confirmed on disk). Cold — empty cache, real download, full 17-test suite including `loadModel()` + every transcribe call: 27 s total. Warm — cache already populated: 3.8–5.4 s for the same full suite. A single `loadModel()` against an already-cached model completes in well under a second; a single `transcribe()` call on the 12.27 s test clip runs in well under a second once warm (`segment` and `word` timestamp modes both — see `test.js`'s printed timings). `loadModel` reuse avoids paying ONNX session-init cost per call; the default `transcribe()` export pays it every call (see API — use `loadModel` for batches).

**Use when:** transcripts/subtitles/search-indexing for audio already in Node or loaded in a browser tab, fully local (no API key, no upload, works offline once the model is cached).<br>
**Not for:** real-time/streaming transcription (this is whole-buffer, `chunk`/`stride` control long-file windowing, not live partial results); speaker diarization (not attempted); guaranteed-accurate word timings (see Limitations).

---

Part of the [@audio/neural](https://github.com/audiojs/neural) lane.

MIT © [audiojs](https://github.com/audiojs)
