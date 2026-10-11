# @audio/neural-timbre

> Name the sound in a recording: a small encoder maps one note to a vector, a catalog of preset vectors ranks it by cosine. Yamaha DX7 first.

[gpu-font](https://github.com/dy/gpu-font)'s recipe applied to sound. A trusted renderer labels the data (here [`@audio/synth-dx7`](https://github.com/audiojs/synth/tree/main/packages/synth-dx7), bit-exact to MSFA); a small network is trained once to put renders of the same preset close together; a catalog holds one vector per preset, the mean of its reference notes, bound to the encoder's SHA-256; a query ranks against it. Catalogs are swappable and carry names and vectors only. The name then seeds a rebuild: the preset's parameters start [`@audio/neural-synth`](../neural-synth)'s `match`.

```js
import { createMatcher } from '@audio/neural-timbre'

let matcher = await createMatcher()                              // the hosted encoder and dx7-factory: fetched once, offline after
let [best] = await matcher.match(note, { sampleRate: 44100 })     // one note; leading silence is trimmed
best.name       // 'E.PIANO 1'
best.cartridge  // 'ROM1A'
best.score      // cosine, -1 to 1
matcher.free()
```

**Status**: trained and benchmarked on the Yamaha DX7. On 2,925 voices from cartridges it never saw, one clean note names the voice 75.0% of the time and puts it in the top five 92.6%; through the augmentation chain, 56.4% and 79.0% ([Benchmark](#benchmark)). ONNX output matches PyTorch within 2·10⁻⁶ in Node and in Chromium on WebGPU. The encoder and two catalogs are hosted on Hugging Face, fetched once and checked by SHA-256 ([Files](#files)).

## Install

```
npm install @audio/neural-timbre onnxruntime-node   # onnxruntime-web in the browser
```

## Files

The encoder and catalogs are not in the package: they are hosted on Hugging Face at [audiojs/dx7-timbre](https://huggingface.co/audiojs/dx7-timbre), with their model card, so a source can come out at its owner's request ([Catalogs](#catalogs)); a published npm version cannot change, and unpublishing past 72 hours is restricted.

| File | Size (gzip) | What | Made by |
|---|---:|---|---|
| `encoder.int8.onnx` | 1.87 MB (1.68) | the encoder, weights int8 per output row, SHA-256 `aaa9fae0…` | `scripts/train.py --run dx7-v1`, then `scripts/export.py` |
| `catalogs/dx7-factory.json` | 0.21 MB (0.09) | 988 sounds of Yamaha's own cartridges, 4-bit vectors | `scripts/catalogs.py --run dx7-v1 --encoding int4-base64 --dir catalogs --only factory,all` |
| `catalogs/dx7-all.json` | 7.9 MB (2.7) | 29,840 sounds of every source, 4-bit vectors | the same run |

The sounds come from the sources in [Catalogs](#catalogs). `scripts/` rebuilds every file from the sysex: see [Training](#training).

Each version reads `https://huggingface.co/audiojs/dx7-timbre/resolve/<REVISION>/<file>` at the commit `REVISION` pins and refuses a file of another SHA-256. Node fetches each file once into the neural cache, `dx7-timbre/` in `$AUDIO_NEURAL_CACHE` or `~/.cache/audiojs/neural`, writes it whole once checked, and reads it there after, offline too; the browser keeps it in the Cache API (`audio-neural`, through `@audio/neural-runtime`). The first match fetches 2.1 MB with `dx7-factory` (encoder and catalog), 9.8 MB with `dx7-all`.

Offline or self-hosted, pass a copy of the repository as `weights`: a directory in Node, or the URL serving it. Each file is still checked against its SHA-256.

```sh
git clone https://huggingface.co/audiojs/dx7-timbre    # git-lfs for encoder.int8.onnx
```

```js
let matcher = await createMatcher('dx7-all', { weights: './dx7-timbre' })    // or 'https://example.com/dx7-timbre/'
```

## Algorithm

1. **Features** (`features.js`): the note at 44.1 kHz (other rates resampled), cut at its onset (the hop before the first one within 60 dB of the loudest), as a constant-Q power spectrogram: 96 bins 12 an octave from E1 (41.2 Hz) to 9956 Hz, Q = 1/(2^(1/12) − 1) (Brown 1991), Hann windows of Q periods capped at 8192 samples, computed with one FFT a frame and a sparse spectral kernel per bin (Brown & Puckette 1992); 128 frames 512 samples apart (1.49 s); dB below the loudest cell with an 80 dB floor, so level never matters.
2. **Encoder** (`scripts/train.py`): four convolution stages of 32, 64, 128 and 256 channels (3×3, batch norm, ReLU; one convolution in the first stage, two after; 2×2 max pooling), a 1×1 convolution to 96 channels, the mean and maximum over time of each channel-band cell, then two linear layers to a 128-dimensional unit vector: 1.85M parameters, 1.9 MB as int8 ONNX (weights rounded per output row, gpu-font's scheme). Trained once on 26,206 DX7 voices, eight views each: any note 33–99, velocity 20–127, held 0.4–2.5 s or staccato 0.06–0.4 s, two thirds of them through an augmentation chain whose stages each apply by their own chance: chorus, EQ, reverb (Freeverb, Dattorro), another voice or drums underneath at 0–20 dB SNR, noise at 5–35 dB, compression, gain and MP3 or Opus at 12–64 kb/s, all from audiojs. Each step: 12 voices, each with one of its 8 nearest training voices by parameters (the near-copies it must tell apart), 4 views each; supervised contrastive loss over the views (Khosla et al. 2020, temperature 0.1), a margin softmax against one proxy per training voice (CosFace, scale 30, margin 0.15, as gpu-font's style encoder), and the voice's algorithm (1–32) read linearly from the vector. 30,000 steps took 1.7 hours on an Apple M4 Max GPU (MPS, within 3 GB, shared with other jobs); the kept checkpoint is the one with the best development top-1, the last: 86.0% clean and 72.2% augmented against the 702 development voices.
3. **Catalog**: each preset rendered at C2, C3, C4, C5, C6 (MIDI 36–84) at velocities 64 and 112, held 1 s; the unit mean of the ten embeddings, stored with one scale per row as 4-bit integers in the hosted catalogs (gpu-font's `int4-base64`) and as 8-bit in the benchmark's (`int8-base64`). `readCatalog` refuses a catalog whose `encoderSha256` or `featuresSha256` differs from the encoder's.
4. **Ranking**: cosine against every row, best first.

### Features, by measurement

Same 3,000 training voices, 8 views each, same encoder, 3,000 steps; a catalog of the 702 development voices, from cartridges never trained on; one clean and one augmented query per voice at a note and velocity no reference used. The frontend that won is the one shipped. The pilot ran with librosa's absolute floor (`amin` 10⁻¹⁰), before the floor became relative to the loudest cell; the choice rests on these figures. `scripts/logmel.js` keeps the log-mel frontend, which matched librosa to 3·10⁻⁶ dB.

| Development voices | clean top-1 | clean top-5 | augmented top-1 | augmented top-5 |
|---|---:|---:|---:|---:|
| log-mel, 96 Slaney bands, 2048-point STFT | 61.0% | 86.2% | 40.5% | 64.7% |
| **constant-Q, 96 bins** | **71.7%** | **90.0%** | **48.9%** | **71.9%** |
| constant-Q, supervised contrastive alone (no proxies) | 62.5% | 85.9% | 44.6% | 71.4% |

## Benchmark

Frozen before training: `scripts/freeze.js` pins the voice split, the render and augmentation recipe and the frontend by SHA-256, and `scripts/bench.js` refuses to report against a changed pin (the manifest records two re-pins of the frontend, both before any held-out feature existed: the feature choice, and the level-floor fix). Held-out cartridges are a fifth of the 4,158 distinct cartridge files, chosen by a hash of the file, with Yamaha's forced into training; a voice counts as held out only if no training or development cartridge holds the same sound. That leaves 2,925 audible voices from 247 cartridges, never seen in training or checkpoint selection. Each is one catalog entry and one query: a note (MIDI 38–94) and velocity (30–127) no reference uses, held 0.6–2 s or staccato 0.08–0.3 s, after 0–50 ms of silence; the same query under each condition. The hosted encoder throughout, `encoder.int8.onnx` in `@audio/neural-runtime`, with int8 catalogs; the hosted catalogs' 4-bit vectors cost about a point of clean top-1 ([Size and speed](#size-and-speed)).

| Query | held-out top-1 | top-5 | all voices top-1 | top-5 |
|---|---:|---:|---:|---:|
| clean | 75.0% | 92.6% | 45.0% | 75.8% |
| reverb (Freeverb, room 0.8, 35% wet) | 64.8% | 86.7% | 35.2% | 65.2% |
| EQ (+6 dB below 150 Hz, -6 dB at 1.5 kHz, -9 dB above 5 kHz) | 70.7% | 89.9% | 41.1% | 70.5% |
| compressor (-30 dB, 8:1) | 63.7% | 86.1% | 32.6% | 64.0% |
| chorus (3 voices, 0.8 Hz) | 67.2% | 88.9% | 36.9% | 67.4% |
| gain -30 dB | 75.0% | 92.6% | 45.0% | 75.8% |
| MP3 32 kb/s | 69.5% | 89.2% | 39.7% | 68.8% |
| Opus 12 kb/s | 63.6% | 85.3% | 33.8% | 63.2% |
| pink noise, 20 dB SNR | 62.1% | 84.5% | 32.4% | 61.8% |
| pink noise, 10 dB SNR | 50.9% | 76.9% | 22.3% | 48.3% |
| another voice under it, 10 dB SNR | 49.0% | 73.3% | 24.3% | 48.3% |
| another voice, 0 dB SNR | 28.6% | 48.0% | 11.8% | 27.9% |
| drums, 10 dB SNR | 67.4% | 87.9% | 37.2% | 67.4% |
| training chain (a random mix of the above) | 56.4% | 79.0% | 28.0% | 56.2% |

- **Held-out**: the right voice among the 2,925 held-out voices. **All voices**: among all 29,840, the 26,213 training voices included (gpu-font's "all catalogs" read): with ten times the candidates, top-1 drops to 45%, largely to training voices that are variations of the held-out one (26% of training voices lie within 3 parameters of another, 14% within 1; sampled over 3,000).
- **Level** never matters: the −30 dB copy scores exactly as the clean note, since the features are relative to the loudest cell.
- **Near copies**: 412 held-out voices lie within 3 parameters of a training voice; they score as the 2,511 novel ones on the held-out catalog (73.3% against 75.3% clean top-1) and lower on all voices (35.7% against 46.6%), where their training neighbour competes.
- **Algorithm head**: the voice's algorithm (1–32) comes out right for 37.1% of clean held-out queries (always answering the most common, algorithm 5, would score 13.1%). It is weak; the catalog vector, not the head, carries identity.
- **Fails**: two voices under one note (mix at 0 dB: 28.6%) and heavy noise; one note, one voice is the working range. Separate first (`neural-separate`) and query per note.

Retrieval figures within one catalog are exact, not twin-credited: a held-out voice whose near twin is also held out counts as a miss when the twin ranks first.

## Size and speed

| | held-out clean top-1 | top-5 | augmented top-1 | top-5 | all voices clean top-1 | top-5 |
|---|---:|---:|---:|---:|---:|---:|
| float32 encoder (7.4 MB), int8 catalog | 74.8% | 92.6% | 56.3% | 79.2% | | |
| int8 encoder (1.9 MB, 1.7 MB gzip), int8 catalog | 75.0% | 92.6% | 56.4% | 79.0% | 45.0% | 75.8% |
| **int8 encoder, int4 catalog: hosted** | **74.0%** | **92.1%** | **56.4%** | **79.0%** | **44.3%** | **75.1%** |

Frozen queries. Rounding the weights to int8 per output row costs nothing measurable. 4-bit catalog vectors cost a point of clean top-1 or less and nothing augmented (all voices: 28.3% and 55.9% against 28.0% and 56.2%); with gzip, as web servers send them, they cut `dx7-all` from 4.7 MB to 2.7 MB and `dx7-factory` from 0.15 MB to 0.09 MB. The hosted catalogs are 4-bit, as gpu-font's are.

In Node (onnxruntime-node, Apple M4 Max under other load, int8 catalogs): loading the int8 encoder and `dx7-factory` takes 0.15 s, `dx7-all` 0.63 s; a match takes 56 ms against `dx7-factory` and 85 ms against `dx7-all`, of which the constant-Q frontend is 35 ms. The encoder alone, 32 notes a batch in headless Chromium: 0.7 ms a note on WebGPU, 24 ms on single-threaded WASM.

## Rebuild

A name is a starting point: the preset's parameters seed [`@audio/neural-synth`](../neural-synth)'s `match` in the original synth. `scripts/rebuild.js` measures that on 64 clean held-out queries (a fixed sample by voice hash): `specLoss` between the query note and a preset played at the same note, velocity and length, for a random other held-out voice, the top-1 retrieved from `dx7-heldout`, the retrieved preset after seeded `match`, and blind `match` (no seed) at the same budget of 1,000 renders over the 145 sound parameters (the name bytes stay fixed: they cannot change the sound).

| specLoss, 64 held-out notes | mean | median |
|---|---:|---:|
| random held-out voice | 29.0 | 12.2 |
| retrieved preset | 5.9 | 0 |
| retrieved, then seeded `match` | **1.9** | **0** |
| blind `match`, same budget | 9.6 | 9.9 |

Seeded beats blind on 63 of 64 notes. The gain is retrieval's: 44 of the 64 retrieve the exact voice (loss 0). On the 20 misses the retrieved preset scores a median of 6.7 against blind's 10.0. At 1,000 renders refinement improved only the 3 misses whose retrieved preset was worse than blind's usual result (losses 251, 22 and 11), to about blind's level, and left the 17 near misses as retrieved. Four times the budget (`--budget 4000 --misses`) starts to refine:

| specLoss, the 20 misses | 1,000 renders: mean | median | 4,000 renders: mean | median |
|---|---:|---:|---:|---:|
| retrieved preset | 19.0 | 6.7 | 19.0 | 6.7 |
| retrieved, then seeded `match` | 6.1 | 6.7 | **5.5** | **6.2** |
| blind `match` | 9.5 | 10.0 | 7.1 | 6.9 |

At 4,000 renders seeding improves 10 of the 20 misses and still beats blind on 17. Each CMA-ES start is evaluated before it steps, so seeding never makes the result worse.

```js
import dx7, { parse, MAX } from '@audio/synth-dx7'
import match from '@audio/neural-synth'

let voice = parse(cartridgeBytes)[best.index - 1].patch            // your own copy of best.cartridge
let render = p => dx7(freq, { patch: Uint8Array.from(voice).map((v, i) => i < 145 ? Math.round(p[i]) : v), velocity, duration, release })
let r = match(render, recording, { bounds: [...MAX.subarray(0, 145)].map(m => [0, m]), seeds: [[...voice.subarray(0, 145)]], budget: 1000 })
```

## Catalogs

Two are hosted with 4-bit vectors, bound to `encoder.int8.onnx`: `createMatcher()` searches `dx7-factory`, `createMatcher('dx7-all')` every sound. `scripts/catalogs.py` writes all four into `timbre/catalogs/` in the neural cache, at 8 bits by default: the benchmark's.

| Catalog | Voices | From | Hosted, int4 (gzip) | int8 (gzip) |
|---|---:|---|---:|---:|
| `dx7-factory` | 988 | Yamaha's own cartridges: ROM1A–ROM4B (223 distinct sounds) and VRC-101 to VRC-112 (765) | 0.21 MB (0.09) | 0.29 MB (0.15) |
| `dx7-all` | 29,840 | every audible voice of every source | 7.9 MB (2.7) | 10.5 MB (4.7) |
| `dx7-heldout` | 2,925 | voices found only in held-out cartridges: the benchmark's catalog | | 1.07 MB (0.46) |
| `dx7-dev` | 702 | development cartridges, which chose the checkpoint | | 0.26 MB (0.11) |

An entry holds `name`, `cartridge` (ROM1A for Yamaha's, else the file's path in its archive), `index` (the voice's number in the file: 1–32 in a bank, counting on through a file of several banks, as 223 entries of `dx7-all` do), `source`, `algorithm` and `id`: a hash of the parameters the engine uses (`soundKey` in `scripts/voices.js`), one per sound, so voices that render the same samples share it. No voice data: to rebuild, load the cartridge with `@audio/synth-dx7`'s `parse` and take the voice by number (`scripts/voices.js` reads a file of several banks bank by bank).

| `source` | Sounds in `dx7-all` | What, from where | Terms stated |
|---|---:|---|---|
| `yamaha-factory` | 223 | Yamaha's ROM1A–ROM4B cartridges, from [yamahablackboxes.com](https://yamahablackboxes.com/collection/yamaha-dx7-synthesizer/patches/), byte-identical to rohandrape.net's copies | none; voice data © Yamaha Corporation, 1983 |
| `yamaha-vrc` | 765 | Yamaha's VRC-101 to VRC-112 cartridges, sides A and B, from the same site | none; © Yamaha and the named programmers |
| `greymatter` | 56 | Grey Matter Response E! card disks 2, 5 and 7, from the same site | none; © Grey Matter Response |
| `alltheweb` | 28,571 | [DX7 All The Web](https://bobbyblues.recup.ch/yamaha_dx7/dx7_patches.html): 13,192 sysex files gathered from the web by Bobby Blues | "public domain material" gleaned from the Net, per the collector, who removes material at its owners' request; it includes commercial banks |
| `library` | 225 | BlackWinny's [Yamaha-DX7-patch-library](https://github.com/visualizersdotnl/Yamaha-DX7-patch-library) v1.0 (2015), commit 6b17188 | CC0-1.0, marked by its uploader, who does not own every bank |

No source licenses all its voices for this use (the library's CC0 mark comes from an uploader who does not own every bank), so the catalogs follow gpu-font's policy for DaFont and Adobe Fonts: names and vectors only, published by the maintainer's decision, and a source comes out at its owner's request: open a discussion on [audiojs/dx7-timbre](https://huggingface.co/audiojs/dx7-timbre) or an issue here, naming the source or cartridges; a new revision drops their entries and the next version of the package pins it. The sysex stays in `~/.cache/audiojs/data/dx7`, whose `sources.json` records the same.

Your own presets make a catalog the same way: render each at a few notes, `createEncoder().project` the `features`, average, and pass the unit means to `catalog()` (`scripts/catalog.js` does it for DX7 voices through `@audio/neural-runtime`).

## Training

Everything rebuilds from the sysex. Data stays in `~/.cache/audiojs/data/dx7` (`$TIMBRE_DATA`): the cartridges, `sources.json` with each source's terms, the voice table and float16 features read with `pread`, never loaded whole. Models, catalogs and reports go to `timbre/` in the neural cache (`$AUDIO_NEURAL_CACHE` or `~/.cache/audiojs/neural`). Beyond the package's dependencies, the scripts need:

- `@audio/synth-dx7` wherever voices are read or rendered; `@audio/neural-synth` for `rebuild.js`; for the augmentations of `factory.js`, `@audio/effect-chorus`, `eq-parametric`, `reverb-freeverb`, `reverb-dattorro`, `dynamics-compressor`, `encode-mp3`, `decode-mp3`, `encode-opus`, `decode-opus` and `synth-drum`. All resolve from `scripts/`: in the lane, `@audio/neural-synth` as a workspace and the rest as symlinks in the package's own `node_modules/@audio` (`ln -s <package dir> node_modules/@audio/<name>`).
- `playwright-core` for `webgpu.js`.
- Python with torch (MPS), numpy, onnx and onnxruntime for `train.py`, `export.py` and `catalogs.py`.

```sh
node scripts/voices.js                 # sysex → voices.json, patches.u8: distinct sounds, split by cartridge
node scripts/freeze.js                 # pin the benchmark before any training
node scripts/factory.js dev            # constant-Q features of development references and queries
node scripts/factory.js train          # 8 augmented views of each training voice
python scripts/train.py --run NAME     # MPS, within 3 GB
python scripts/export.py --run NAME    # encoder.onnx, encoder.int8.onnx, parity.json
node scripts/factory.js heldout        # the benchmark's references and queries, 14 conditions
node scripts/factory.js refs           # references of training voices
python scripts/catalogs.py --run NAME  # catalogs/dx7-{factory,dev,heldout,all}.json
node scripts/bench.js                  # the frozen benchmark through @audio/neural-runtime
node scripts/rebuild.js                # retrieval-seeded match against blind match
node scripts/webgpu.js --chrome PATH   # the encoder in headless Chromium, WebGPU and WASM, against PyTorch
```

The hosted files: `encoder.int8.onnx` is `export.py`'s, copied from the cache into the package directory, and `python scripts/catalogs.py --run dx7-v1 --encoding int4-base64 --dir catalogs --only factory,all` writes the two catalogs in place, laid out as audiojs/dx7-timbre. A new build goes up as a new revision: its SHA-256s into `FILES` in `timbre.js`, its commit into `REVISION`.

`npm test` needs none of this: it runs on the hosted files (the copies in the package directory where they are, else fetched) and `fixtures/rom1a.f16`, the features of ROM1A's 32 voices at F4, velocity 90 (`fixtures/make-rom1a.js` renders them, with `@audio/synth-dx7` and the cartridge). With those two at hand it also checks the fixture against a live render, and with `export.py`'s `parity.json` in the cache, both encoders against PyTorch.

The feature pilot: `factory.js train --name pilot --limit 3000 --views 8 --kinds logmel,cqt` and `factory.js dev --kinds logmel,cqt`, then `train.py --train pilot --kind logmel` (or `cqt`, or `cqt --proxy 0`) `--steps 3000`.

`scripts/catalog.js <voices>` builds a catalog through `@audio/neural-runtime` alone, rendering on the fly: the path for a small catalog of your own presets.

## API

| Export | |
|---|---|
| `createMatcher(catalog = 'dx7-factory', { model?, backend?, weights? })` | → `Promise<Matcher>`: `catalog` is a hosted catalog's name (`'dx7-factory'`, `'dx7-all'`), a URL, a path (Node) or parsed JSON, bound to the encoder; `match(audio, { sampleRate, top = 5 })` → entries best first with `score` |
| `identify(audio, { catalog?, sampleRate, top, model?, backend?, weights? })` (default) | one-shot `createMatcher` → `match` → `free` |
| `createEncoder({ model?, backend?, weights? })` | → `{ embed(audio, { sampleRate }), project(patches), binding, free() }` |
| `MODEL`, `catalogs`, `REVISION` | URLs of the hosted encoder and catalogs, and the commit of audiojs/dx7-timbre they are read at |
| `features(audio, { sampleRate, trim = true })` | → `Float32Array(96 × 128)`, band-major, in [−1, 1] |
| `catalog(entries, vectors, binding, { encoding })` | → catalog JSON from unit vectors; `encoding` `'int8-base64'` (default) or `'int4-base64'` |
| `readCatalog(json, binding?)`, `rank(embedding, catalog, top)` | read and search a catalog |
| `FEATURES`, `cqt(samples)`, `featuresSha256()`, `sha256(bytes)` | the input contract, the transform on samples already at the onset, and the hashes a catalog binds |

`model` is a URL, a path (Node) or bytes; default the hosted encoder, the graph the hosted catalogs bind. `weights` is a URL, or a directory in Node, holding a copy of audiojs/dx7-timbre, read instead of the hosted revision ([Files](#files)). `backend` passes to `@audio/neural-runtime`. Catalogs from `scripts/` or your own presets load by path or URL: `createMatcher(path.join(cache, 'timbre/catalogs/dx7-heldout.json'))`.

## Prior art

| | Le Vaillant & Molle, 2025 | Braun & Finkelstein, 2026 | this package |
|---|---|---|---|
| Retrieves | an instrument from a gallery of audio embeddings, from a single note or a mixture of three | a DX7 preset's parameters from its audio, through a shared audio-parameter embedding | a preset from a catalog of audio embeddings, from one note |
| Data | NSynth (1,000 instruments) and Surge patches (2,884) | 31,443 DX7 presets from DX7AllTheWeb, deduplicated by parameters, loudness-filtered | 29,886 DX7 sounds from Yamaha's ROM and VRC cartridges, DX7 All The Web and BlackWinny's library, deduplicated by rendered samples |
| Positives | other notes of the same virtual instrument, not crops of one sound | a preset's own parameters (SLAP objective) | other notes, velocities, lengths and augmentations of the same preset |
| Queries | notes and three-instrument mixtures | C4 at velocity 85, as rendered for training | a note and velocity the catalog never rendered, held or staccato, 14 conditions |
| Gallery | 3,864 instruments | 3,144 presets; 4,096 with 8 of 32 algorithms unseen | 2,925 unseen voices; 29,840 in all |
| Encoder | Audio Spectrogram Transformer, AudioSet-pretrained | CNN14 variant, 82.8M parameters | CNN, 1.85M parameters, 1.9 MB at int8 |
| Reported | 80.4% top-1, 93.1% top-5 single (InfoNCE); 81.7%, 95.7% in mixtures of three | R@1 86.1% all seen; R@1 52.2%, R@10 88.5% with algorithms unseen | [Benchmark](#benchmark) |

The columns measure different tasks on different galleries, so their figures do not compare. Both confirm the premise: a synth's own renders are labels enough. Braun & Finkelstein's topology-aware parameter encoder is the next step for presets no catalog holds.

- G. Le Vaillant, Y. Molle, "Contrastive timbre representations for musical instrument and synthesizer retrieval," [arXiv 2509.13285](https://arxiv.org/abs/2509.13285), 2025.
- D. Braun, A. Finkelstein, "FM synthesizer audio-parameter shared embeddings," [arXiv 2608.18226](https://arxiv.org/abs/2608.18226), 2026.

## Reference

J. C. Brown, "Calculation of a constant Q spectral transform," *JASA* 89(1), 1991. · J. C. Brown, M. S. Puckette, "An efficient algorithm for the calculation of a constant Q transform," *JASA* 92(5), 1992. · P. Khosla et al., "Supervised contrastive learning," *NeurIPS* 2020. · H. Wang et al., "CosFace: large margin cosine loss for deep face recognition," *CVPR* 2018. · R. Levien, [Music Synthesizer for Android](https://github.com/google/music-synthesizer-for-android) (the DX7 engine, Apache-2.0). · [gpu-font](https://github.com/dy/gpu-font) (the recipe).

**Use when:** naming which preset of a known synth plays a recorded note (a DX7 factory sound, a cartridge voice), as the seed of a rebuild in the original synth; indexing your own presets into a catalog and searching them by sound.<br>
**Not for:** chords, layered or heavily processed parts without separating them first (one note, one voice: see the mixture and reverb rows above); synths no catalog covers (the closest preset comes back with a lower score, not a refusal: there is no calibrated threshold yet); real-time use.

---

Part of the [@audio/neural](https://github.com/audiojs/neural) lane.

MIT © [audiojs](https://github.com/audiojs)
