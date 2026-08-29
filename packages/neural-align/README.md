# @audio/neural-align

> Forced alignment: audio + the words already known to be spoken → per-word (and per-character) timestamps.

Forced alignment is not ASR. ASR searches an open vocabulary for the most likely text — ambiguous, needs a language model, and any timestamps it produces are a side effect of that search. Forced alignment is given the correct text as a hard constraint and only has to answer *when* each part of it was said: a much narrower, much more accurate problem. Two layers:

- **`./ctc`** — the CTC forced-alignment core: a trellis/Viterbi search over blank-interleaved targets, exactly the recurrence behind [Graves 2006](https://www.cs.toronto.edu/~graves/icml_2006.pdf) ("Connectionist Temporal Classification") and torchaudio's [`forced_align`](https://pytorch.org/audio/stable/tutorials/ctc_forced_alignment_api_tutorial.html). Pure JS, no model, fully unit-tested against hand-computed and property-based cases.
- **`.` (default)** — a model adapter: a wav2vec2-class CTC acoustic model ([`@huggingface/transformers`](https://github.com/huggingface/transformers.js), Apache-2.0) supplies the per-frame log-probabilities the core aligns against.

```js
import align from '@audio/neural-align'

let { words, chars, score } = await align(audioFloat32, 'the exact words that were spoken', { sampleRate: 44100 })
// words: [{ text: 'THE', start: 0.46, end: 0.60, score: 0.79 }, …]  seconds, monotonic
```

```js
import { ctcAlign, mergeWords } from '@audio/neural-align/ctc'
// or: import align, { ctcAlign, mergeWords } from '@audio/neural-align'

let { path, score, spans } = ctcAlign(logProbs, T, V, targetTokenIds, { blank: 0 })
let words = mergeWords(spans, id => vocab[id], { delimiter: '|', frameDuration: 0.02 })
```

## Model / license audit

| Model | License | Language | Default |
|---|---|---|---|
| `Xenova/wav2vec2-base-960h` (`facebook/wav2vec2-base-960h` weights) | Apache-2.0 | English | **yes** |
| `facebook/mms-300m`-class checkpoints | **CC-BY-NC 4.0 — non-commercial only** | 1000+ languages, per-language adapter | no — pass `model` explicitly, and only if the license fits your use |
| `Xenova/wav2vec2-large-xlsr-53-*` community fine-tunes | varies per fine-tune — check that fine-tune's own model card | one language per fine-tune | no |

Pass any wav2vec2-class CTC checkpoint via `opts.model`; the adapter only needs `AutoProcessor` + `AutoTokenizer` + a CTC head it can read logits from. This build of `@huggingface/transformers` has no MMS per-language adapter switching — `opts.language` is accepted for documentation/forward-compatibility but has no effect on `wav2vec2-base-960h`; for another language pass a `model` that is already fine-tuned for it, don't rely on `language` to steer a multilingual checkpoint.

## Options

| Option | Default | |
|---|---|---|
| `sampleRate` | required | Hz of the input audio |
| `model` | `'Xenova/wav2vec2-base-960h'` | any wav2vec2-class CTC checkpoint on the HF hub |
| `language` | `'en'` | documentation only — see the license-audit note above |
| `normalize` | `true` | uppercase + drop characters the vocab has no id for (punctuation). Numbers are **not** spelled out — see Limits |
| `chunk` | `20` (s) | inference window for long audio, 1 s overlap, frames stitched into one alignment — the trellis itself is O(T·L) and is never chunked |
| `dtype` / `device` | onnxruntime defaults | passed through to `from_pretrained` |
| `cache` | see Cache below | Node only: override the model cache directory for this call |
| `cues` / `lrc` | `false` | attach [`@audio/subtitle`](https://github.com/audiojs/midi/tree/main/packages/subtitle) output — needs it installed separately (`peerDependencies`, optional) |

## Cache

The wav2vec2 acoustic model is ~360 MB (fp32 ONNX) and is fetched from the HF hub on first use. In Node, `align.js` points `@huggingface/transformers`' own cache (`env.cacheDir`, which otherwise defaults to `node_modules/@huggingface/transformers/.cache` and gets wiped by every `npm install`) at the lane's shared cache root instead:

- `$AUDIO_NEURAL_CACHE/hf` if `$AUDIO_NEURAL_CACHE` is set — the same root [`@audio/neural-runtime`](https://github.com/audiojs/neural/tree/main/packages/neural-runtime) uses for its raw ONNX bytes; the `hf/` subdirectory keeps the two caches from colliding.
- `~/.cache/audiojs/neural/hf` otherwise.

Set once at module load, so it applies before any model is fetched. Override per call with `opts.cache` (a directory path) when a process needs an isolated or ephemeral cache (tests, containers, multiple pinned model versions) — this is passed straight through as `from_pretrained`'s own `cache_dir` option, so concurrent `loadModel()` calls with different `opts.cache` never race each other's global state.

In the browser, caching is untouched: `@huggingface/transformers` uses the Cache API by default, as it already does for every other transformers.js consumer.

To pre-download the default model (e.g. in a Docker build step, before the process that needs low first-call latency starts):

```js
import { loadModel } from '@audio/neural-align'
await loadModel('Xenova/wav2vec2-base-960h')   // populates the cache; no audio needed
```

## Accuracy

Frames are 20 ms (the wav2vec2 conv-stem's fixed 320-sample stride at 16 kHz — every timestamp this atom produces is a multiple of 0.02 s). The CTC-segmentation literature (Kürzinger et al. 2020, ["CTC-Segmentation of Large Corpora for German End-to-End Speech Recognition"](https://arxiv.org/abs/2007.09127), and the forced-alignment literature it surveys) reports word-boundary error typically ≤ 50 ms on clean speech for this class of frame-synchronous CTC aligner — an expectation from the literature, not a number independently benchmarked against a labeled corpus here. `score` (mean per-word probability, geometric-mean over the word's frames) is a real confidence signal, not decoration: `test.js` shows it collapses to well under half its normal value when the supplied text is wrong.

## Karaoke / LRC

```js
import isolate from '@audio/vocals'   // wav2vec2 is trained on speech, not mixed music
import align from '@audio/neural-align'

let vocals = await isolate(mix, { sampleRate })
let { lrc } = await align(vocals, lyrics, { sampleRate, lrc: true })   // enhanced (word-level) LRC via @audio/subtitle
```

## Limits

- English by default (`wav2vec2-base-960h`); other languages need a different `model` (see the license table — MMS is non-commercial).
- Numbers and abbreviations must already be spelled out in the input text ("three" not "3") — the vocab is 26 letters + apostrophe + word-delimiter, there is no digit-to-word expansion.
- Full music mixes need vocal isolation first (`@audio/vocals`) — the acoustic model is trained on speech.
- `T < target length` (more characters than frames) throws — an impossible alignment, not a best-effort guess.
- Text unrecoverable from the given audio (e.g. a completely wrong transcript) does not throw; it aligns anyway and reports a low `score` — check it.

---

Part of the [@audio/neural](https://github.com/audiojs/neural) lane.

MIT © [audiojs](https://github.com/audiojs)
