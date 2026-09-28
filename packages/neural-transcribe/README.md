# @audio/neural-transcribe

> Polyphonic notes with pitch bends: Spotify's Basic Pitch, run through `@audio/neural-runtime`'s ONNX adapter in Node and the browser.

The neural tier beside [`@audio/mir-transcribe`](https://github.com/audiojs/mir/tree/main/packages/mir-transcribe) (Klapuri multi-F0 with continuity tracking). [Basic Pitch](https://github.com/spotify/basic-pitch) (Bittner, Bosch, Rubinstein, Meseguer-Brocal, Ewert, ICASSP 2022) is instrument-agnostic and polyphonic, and its model is a 230 KB ONNX file. This package runs that file and ports the Python post-processing that turns its posteriors into notes with per-frame pitch bends.

```js
import transcribe from '@audio/neural-transcribe'

let notes = await transcribe(pcm, { sampleRate: 44100 })
// [{ time, duration, midi, freq, velocity, bends }], sorted by time, then pitch
```

Frame-level outputs, and notes from them without running the model again:

```js
import { posteriors, toNotes } from '@audio/neural-transcribe'

let p = await posteriors(pcm, { sampleRate: 44100 }) // { onset, note, contour, times }, a row per 11.6 ms frame
let sparse = toNotes(p, { onsetThreshold: 0.6, minFreq: 80 })
```

`audio`: a mono `Float32Array` (with `sampleRate`), one array per channel (averaged, as `librosa.load(mono=True)` does), or a `decode()` result `{ channelData, sampleRate }`.

## Pipeline

As `basic_pitch/inference.py` and `note_creation.py` do it, at commit [fa5997a](https://github.com/spotify/basic-pitch/tree/fa5997af0a8210982619003269994a1be25eddf3):

1. Mono at 22050 Hz, resampled by [`@audio/resample-sinc`](https://github.com/audiojs/resample).
2. 3840 zeros in front, then windows of 43844 samples (2 s less one hop) every 36164. The model (CQT, harmonic stacking and CNN, all inside the graph) gives 172 frames per window: 88 note and 88 onset posteriors (MIDI 21–108) and 264 contour bins (3 per semitone, 27.5 Hz the second). The 15 frames at each window edge are dropped, the rest concatenated and cut to `trunc(L / 36164 · 142)` frames.
3. Onsets: local maxima in time of the onset posterior at `onsetThreshold` or above, merged with onsets inferred from rises of the note posterior. Each runs forward while the note posterior stays at `frameThreshold` or above, bridging up to 10 consecutive frames below it, and clears its key and both neighbours. The melodia trick then grows notes, forwards and backwards, from the strongest energy left. Notes of `minDuration` or less are dropped (default 11 frames).
4. Bends: per frame, the contour bin within ±25 bins (±8⅓ semitones) of the note's pitch that peaks under a Gaussian weighting (σ = 5 bins), in cents from the pitch. The model puts MIDI `m` at contour bin `3(m − 21) + 1` (`constants.py` places 27.5 Hz at "the second bin"), while `get_pitch_bends` centers on `3(m − 21)`, so Basic Pitch reads an in-tune note as +33.3 cents ([#87](https://github.com/spotify/basic-pitch/issues/87)). This package centers on the note's own bin, and an in-tune note reads 0; `upstream: true` keeps Basic Pitch's center.
5. Times: frame `n` at `n · 256 / 22050` s, less 10.3 ms for each completed 172-frame block (`model_frames_to_time`: a window holds 43844 samples, not 172 · 256).

`velocity` is the mean note posterior over the note, 0–1, what Basic Pitch calls amplitude (its MIDI velocity is `round(127 · velocity)`). `freq` is the equal-tempered frequency of `midi`. `bends[k]` belongs to the note's `k`-th frame; Basic Pitch's MIDI writer spaces the bends evenly from `time` to `time + duration`.

| Option | Default | |
|---|---|---|
| `sampleRate` | | required unless `audio` carries it |
| `onsetThreshold` | `0.5` | onset posterior peak that starts a note (`DEFAULT_ONSET_THRESHOLD`) |
| `frameThreshold` | `0.3` | note posterior that sustains it (`DEFAULT_FRAME_THRESHOLD`), ≥ 0 |
| `minDuration` | `0.1277` | seconds, rounded to frames; a note must span more (`DEFAULT_MINIMUM_NOTE_LENGTH_MS`): kept notes last 12 frames, 139 ms, or longer |
| `minFreq` / `maxFreq` | | Hz; keys below the one nearest `minFreq`, and from the one nearest `maxFreq` up, are ignored |
| `inferOnsets` | `true` | add onsets where the note posterior rises |
| `melodiaTrick` | `true` | grow notes from energy without an onset |
| `upstream` | `false` | bends exactly as Basic Pitch gives them, one bin sharp: an in-tune note reads +33.3 cents |
| `batch` | `8` | 2 s windows per model run; batch 1 matches Python onnxruntime to 1e-7, batch 8 to 6e-5 and uses 40% less CPU |
| `model` | `MODEL` | URL or bytes of the ONNX model |
| `device` | `'auto'` | `@audio/neural-runtime` backend: `'node'`, `'wasm'` (the browser's `'auto'`) or `'webgpu'`, which a page must ask for |
| `session` | | replaces `@audio/neural-runtime`'s `load()` (a custom ORT setup, a test double); freed after the call |

## Model and cache

`MODEL` is `basic_pitch/saved_models/icassp_2022/nmp.onnx` at the ported commit (230,444 bytes, sha256 `2c3c1d14…`, the same file as release v0.4.0). The first call downloads it through `@audio/neural-runtime`, which caches it in `~/.cache/audiojs/neural` (`$AUDIO_NEURAL_CACHE`) in Node and in the Cache API in the browser. raw.githubusercontent.com sends `Access-Control-Allow-Origin: *`, so the default works in a page; pass `model` (a URL or the bytes) to self-host or run offline.

## Verification

Against Python basic-pitch at the same commit, with onnxruntime 1.30.0 in both languages (`test.js`):

- **Note creation**, fed basic-pitch's own posteriors: identical notes, frames and bends (with `upstream: true`), and velocity within 2e-7 (numpy averages in float32), on 6 files (vocadito_10 and the Accuracy renders) × 7 option sets and on synthetic posteriors × 8 option sets that reach the edge cases (`fixtures/notes.json`, from `fixtures/make-notes.py`).
- **Model**, on the same 22050 Hz samples: posteriors within 1.2e-7 at batch 1, within 6.2e-5 at batch 8, and the same notes either way.
- **Browser**, headless Chromium with onnxruntime-web 1.30.0 on the Apple GPU: WebGPU posteriors within 1.5e-6 of Python and the same notes on vocadito_10, the chords and the 60 s mix (765 notes); wasm within 1.9e-4 and the same 664 notes on the mix.
- **From 44.1 kHz**: basic-pitch resamples with librosa (soxr_hq), this package with `@audio/resample-sinc` 1.2 (88.6 dB from soxr on vocadito_10). Over the six files, 968 of basic-pitch's 969 notes come out identical, 961 of them with every bend (`upstream: true`); all 969 keep their pitch with onset and offset within one frame, and no other note appears. The one that differs, in the 60 s mix, moves by a frame.
- **Bends**, by default: in-tune sine, triangle and six-harmonic tones at MIDI 36–96 read 0 cents in 2,100 of 2,103 mid-note frames, one bin off in the other 3, where Basic Pitch reads +33.3 (the 47 of 48 tones whose note starts within 50 ms; a sine at MIDI 43 starts 90 ms late). On the Accuracy table's vibrato line (±30 cents at 5.5 Hz), the error against the rendered pitch is 2.8 cents on average and 11.5 cents RMS, near the 9.6 cents RMS that 33-cent steps allow; Basic Pitch's bends are off by 34.5 cents on average, 36.2 RMS.

The Python reference itself reproduces basic-pitch's test fixture (`tests/resources/vocadito_10`): posteriors within 3.5e-5, the same 28 notes and bends. To rerun: `python scripts/reference.py` on vocadito_10.wav and the files `scripts/accuracy.mjs` renders writes the reference outputs into the neural cache; `node test.js` compares against whatever is there.

## Accuracy

Note F1 by [mir_eval](https://github.com/craffel/mir_eval) (onset within 50 ms, pitch within 50 cents; offsets within 20% or 50 ms for "+ offset"), against `@audio/mir-transcribe` with its defaults. Synthetic files are rendered by audiojs synths, so the labels are the rendered notes; [vocadito](https://zenodo.org/records/5578807) (CC BY 4.0) is solo singing with two annotators (annotator 2: neural 0.50, mir-transcribe 0.18):

| Audio | Notes | neural P / R / F1 | + offset | mir-transcribe P / R / F1 | + offset |
|---|---|---|---|---|---|
| chords (FM e-piano, sawtooth voice) | 57 | 0.74 / 0.95 / **0.83** | 0.82 | 0.37 / 0.65 / 0.47 | 0.33 |
| arpeggio (Karplus-Strong, eighths at 120 bpm) | 80 | 0.90 / 1.00 / **0.95** | 0.95 | 0.29 / 0.19 / 0.23 | 0.00 |
| vibrato line (triangle, ±30 cents at 5.5 Hz) | 17 | 1.00 / 1.00 / **1.00** | 1.00 | 0.94 / 0.94 / 0.94 | 0.94 |
| 60 s mix of the three | 378 | 0.47 / 0.83 / **0.60** | 0.39 | 0.45 / 0.49 / 0.47 | 0.17 |
| vocadito, 40 sung tracks, 817 s (mean per track, annotator 1) | 2,237 | 0.52 / 0.48 / **0.49** | 0.31 | 0.14 / 0.46 / 0.21 | 0.12 |

To rerun: `node scripts/accuracy.mjs` renders the files into `~/.cache/audiojs/data/transcribe` and transcribes them with both packages, `python scripts/accuracy.py` scores them.

## Speed and memory

60 s of 44.1 kHz audio through `transcribe()`, onnxruntime-node with 4 intra-op threads, on an Apple M4 Max running other jobs (load average 160 on 14 cores): 5.6 s wall, RTF 0.09, and 3.7 s of CPU at batch 8, of which inference is 2.5 s, resampling 0.8 s (0.5 s with `@audio/resample-sinc` 1.2, timed beside 1.1) and note creation 0.08 s. Batch 1 takes 5.0 s of CPU. Peak RSS of the process 250 MB at batch 8, 180 MB at batch 1. Posteriors take 9 MB per minute of audio, and the whole buffer is analysed at once.

In headless Chromium on the same machine, once onnxruntime-web is loaded, the 60 s mix's posteriors take 0.13–0.41 s on WebGPU (RTF under 0.007) and 1.0–1.1 s on wasm with 4 threads. The first call also fetches onnxruntime-web's wasm (14–28 MB) and took 5–10 s.

## Limitations

- **Bends come in 33.3-cent steps**, the contour's bins: a ±30-cent vibrato reads as −33.3, 0 and +33.3. On triangles the steps are centered (+33.3 from 20 cents sharp, −33.3 from 20 cents flat); on pure sines they sit about 5 cents low (+33.3 from 10–15 cents sharp, −33.3 only from 20–25 cents flat).
- **Held notes restart under other instruments' attacks.** In the 60 s mix, 253 of 350 false notes restart a sounding note on the plucks' eighth-note grid.
- **Sawtooth tones under vibrato split notes, band-limited or not.** With ±30-cent vibrato, nine band-limited sawtooth notes come out in 40 pieces, and the naive `@audio/synth-sfx` sawtooth gives 97 notes for 17 (F1 0.07); sine and triangle with the same vibrato give every note whole.
- Notes shorter than 12 frames (139 ms) are dropped by default; pitches outside MIDI 21–108 are not modelled.

## License

Apache-2.0. `transcribe.js` translates Basic Pitch's Apache-2.0 Python (Copyright 2022 Spotify AB); its header lists the changes and [NOTICE](./NOTICE) carries the attribution. The model is Apache-2.0 too and is downloaded, not bundled.

## Reference

R. M. Bittner, J. J. Bosch, D. Rubinstein, G. Meseguer-Brocal, S. Ewert, "A Lightweight Instrument-Agnostic Model for Polyphonic Note Transcription and Multipitch Estimation", ICASSP 2022, [arXiv:2203.09893](https://arxiv.org/abs/2203.09893). · [spotify/basic-pitch](https://github.com/spotify/basic-pitch) at fa5997a. · C. Raffel et al., "mir_eval", ISMIR 2014.

**Use when:** notes of polyphonic or unknown instruments, with pitch bends, in Node or a browser, and a 230 KB download is acceptable.<br>
**Not for:** installs that must stay model-free ([`@audio/mir-transcribe`](https://github.com/audiojs/mir/tree/main/packages/mir-transcribe) is classical); drum transcription; notes shorter than about 140 ms at the defaults; real-time streaming (2 s windows).

---

Part of the [@audio/neural](https://github.com/audiojs/neural) lane.

Apache-2.0 © 2022 Spotify AB (Basic Pitch), audiojs (port)
