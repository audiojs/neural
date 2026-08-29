# @audio/neural-diarize

> Speaker diarization: who spoke when — no transcript needed, just speaker turns.

Four steps: (1) [`@audio/vad`](https://github.com/audiojs/denoise/tree/main/packages/vad) finds speech regions, hangover-merged so short pauses (breaths, plosives) don't fragment a turn; (2) fixed sliding windows inside each region are embedded with a speaker-verification model ([`@huggingface/transformers`](https://github.com/huggingface/transformers.js), Apache-2.0); (3) agglomerative (average-linkage) clustering on cosine distance groups windows by speaker; (4) window labels are median-filtered and collapsed into segments, split at speaker changes.

```js
import diarize from '@audio/neural-diarize'

let { segments, speakers } = await diarize(audioFloat32, { sampleRate: 44100 })
// segments: [{ start: 0, end: 5.63, speaker: 'S0', score: 0.94 }, { start: 5.63, end: 12, speaker: 'S1', score: 0.91 }]
// speakers: 2
```

```js
import { loadModel, cluster, toSegments, toSubtitles } from '@audio/neural-diarize'

let net = await loadModel()               // keep the model warm across calls
let r = await net.diarize(audio, { sampleRate })
await net.free()

let labels = cluster(embeddings, { threshold: 0.75 })       // pure JS, no model — see test.js
let cues = toSubtitles(r.segments, asrResult.segments)       // '<v S0>text</v>' VTT voice tags
```

## Model / license audit

| Model | License | Params | Default |
|---|---|---|---|
| `Xenova/wavlm-base-plus-sv` (WavLM-Base-Plus, X-Vector head) | see below | ~94M | **yes** |

WavLM's base architecture and pretrained weights are published at [microsoft/unilm](https://github.com/microsoft/unilm/tree/master/wavlm) under **MIT**. This package's default model is a speaker-verification fine-tune of WavLM-Base-Plus on VoxCeleb1 with an X-Vector head ([Snyder et al. 2018](https://www.danielpovey.com/files/2018_icassp_xvectors.pdf)), published by Microsoft at [microsoft/wavlm-base-plus-sv](https://huggingface.co/microsoft/wavlm-base-plus-sv) — that model card carries no explicit license tag of its own. The fine-tuning recipe lives in [microsoft/UniSpeech](https://github.com/microsoft/UniSpeech), whose repository `LICENSE` file is CC BY-SA 3.0; audit before any commercial redistribution of the *weights themselves*. This package only calls the published ONNX conversion ([`Xenova/wavlm-base-plus-sv`](https://huggingface.co/Xenova/wavlm-base-plus-sv)) at runtime — it does not bundle any weights. VoxCeleb1's own terms of use may separately restrict commercial use of models fine-tuned on it; not independently audited here.

Base paper: Chen et al. 2022, ["WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing"](https://arxiv.org/abs/2110.13900).

## Options

| Option | Default | |
|---|---|---|
| `sampleRate` | required (unless `audio` is `{channelData, sampleRate}`) | Hz of the input audio |
| `model` | `'Xenova/wavlm-base-plus-sv'` | any WavLM-X-Vector-class checkpoint on the HF hub |
| `speakers` | — | exact speaker count; when given, clustering cuts the dendrogram at this many clusters instead of using `threshold` |
| `minSpeakers` / `maxSpeakers` | `1` / `8` | bounds on cluster count when `speakers` is not given |
| `threshold` | `0.75` | cosine-similarity stop threshold for clustering — see *Threshold* below |
| `window` / `hop` | `1.5` / `0.75` (s) | embedding window length and slide |
| `minSegment` | `0.5` (s) | minimum output segment length — also the minimum length a hangover-merged VAD region must reach to count as speech |
| `embeddings` | `false` | include the raw per-window 512-d embeddings in the result |
| `dtype` / `device` | `'fp32'` (Node) / `'q8'` (browser), `'auto'` | passed through to `from_pretrained` |
| `cache` | see Cache below | Node only: override the model cache directory for this call |
| `progress` | — | model download/load progress, plus one call per embedded window |

### Threshold

`0.75` comes from the [`Xenova/wavlm-base-plus-sv` model card](https://huggingface.co/Xenova/wavlm-base-plus-sv)'s own published example: two same-speaker pairs score `cos_sim` ≈ 0.959 and 0.963, a different-speaker pair scores ≈ 0.618. `0.75` sits with wide margin on both sides of that gap. This package's own synthetic 2-speaker test (see *Accuracy* below) independently measured a 0.646 similarity between a real voice and a −5-semitone pitch/formant-shifted copy of itself — also comfortably on the "different speaker" side of 0.75.

## Cache

The X-Vector model is ~380 MB (fp32 ONNX) and is fetched from the HF hub on first use. In Node, `diarize.js` points `@huggingface/transformers`' own cache (`env.cacheDir`, which otherwise defaults to `node_modules/@huggingface/transformers/.cache` and gets wiped by every `npm install`) at the lane's shared cache root instead:

- `$AUDIO_NEURAL_CACHE/hf` if `$AUDIO_NEURAL_CACHE` is set — the same root [`@audio/neural-runtime`](https://github.com/audiojs/neural/tree/main/packages/neural-runtime) uses for its raw ONNX bytes; the `hf/` subdirectory keeps the two caches from colliding.
- `~/.cache/audiojs/neural/hf` otherwise.

Set once at module load, so it applies before any model is fetched. Override per call with `opts.cache` (a directory path) for an isolated or ephemeral cache (tests, containers). In the browser, caching is untouched: `@huggingface/transformers` uses the Cache API by default.

To pre-download the default model:

```js
import { loadModel } from '@audio/neural-diarize'
await loadModel('Xenova/wavlm-base-plus-sv')   // populates the cache; no audio needed
```

## Accuracy

Real-capture check (`audio-lena`, 12.27 s continuous single-speaker narration): 1 speaker, 1 segment covering 100% of the clip, ~700 ms on a warm model (≈17× realtime, fp32, Node CPU EP).

Synthetic 2-speaker check: the first 6 s of `audio-lena`, followed by that same 6 s pitch-shifted −5 semitones via [`@audio/shift-psola`](https://github.com/audiojs/shift/tree/main/packages/shift-psola) (which also shifts formants — its README documents that the final resample rescales the whole spectrum by `ratio`, so formants move with f0; `shift-formant` was **not** used because it explicitly *preserves* formants, which would keep the shifted copy sounding like the same speaker). Measured mean-embedding cosine similarity between the two halves: **0.646** (distance 0.354) — clearly below the 0.75 threshold. The pipeline detects the change: the first speaker-change boundary lands at 5.63 s, 0.37 s from the true 6.0 s splice. It also produced a third, short-lived cluster later in the clip — a single window straddling the hard splice (or landing on the pitch shifter's brief settling region) has a noisier, harder-to-classify embedding than the geometric window overlap alone predicts; median-filtering and `minSegment` merging absorb most but not all of this (see *Limitations*).

## toSubtitles

```js
import transcribe from '@audio/neural-asr'
import diarize, { toSubtitles } from '@audio/neural-diarize'

let asr = await transcribe(audio, { sampleRate })
let { segments } = await diarize(audio, { sampleRate })
let cues = toSubtitles(segments, asr.segments)   // each cue.text becomes '<v S0>…</v>' (WebVTT voice span)
```

Each cue is assigned the speaker of whichever diarization segment overlaps it most, by time overlap (https://www.w3.org/TR/webvtt1/#webvtt-cue-voice-span). A cue with no overlapping segment is returned unchanged.

## Limitations

- **No overlapping-speech handling.** Each window gets exactly one speaker label; two people talking at once are attributed to whichever voice dominates that window's embedding, or split the difference badly.
- **Window-resolution boundaries.** Speaker changes are only located to within roughly half the hop (`hop/2` = 0.375 s at the defaults) in the clean case — in practice, real embeddings near a hard splice are noisier than that geometric bound, and windows straddling a genuine boundary can spawn extra spurious clusters (measured above: ±0.37 s on the boundary itself, plus one extra 3-cluster artifact from a single noisy window).
- **Very short speech regions** (under one `window` after hangover-merge) get a single embedding from less context than the model was tuned for — lower-confidence, not filtered out.
- **`minSegment`** merges a too-short segment into whichever same-run neighbour is longer; a whole speech region shorter than `minSegment` with no same-run neighbour is left as-is (can't merge into nothing).
- Clustering is average-linkage UPGMA with an `O(n²)` distance matrix and `O(n³)` worst-case merge search (`n` = window count) — fine through several hundred windows (tens of minutes of speech at the default hop); very long recordings should pre-segment (e.g. run diarize per chapter/scene) rather than feed hours of audio through one call.

**Use when:** labeling speaker turns for transcripts, meeting notes, or subtitles (`toSubtitles`); counting distinct speakers.<br>
**Not for:** overlapping-speech separation, verifying a *specific* claimed identity (this clusters relative to other speakers in the same clip, it doesn't match against an enrolled voiceprint database), or millisecond-accurate turn boundaries.

---

Part of the [@audio/neural](https://github.com/audiojs/neural) lane.

MIT © [audiojs](https://github.com/audiojs)
