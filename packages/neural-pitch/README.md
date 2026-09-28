# @audio/neural-pitch

> Monophonic pitch from a small transposition-equivariant network with our own MIT weights, in plain JavaScript: a pitch posterior and a voicing probability per frame, and a stage 1 for [`@audio/pitch-pyin`](https://github.com/audiojs/pitch/tree/main/packages/pitch-pyin)'s pitch HMM and note model.

pYIN (Mauch & Dixon 2014) has three stages: YIN candidates per frame, a pitch HMM, and Tony's note model. The first is the weak one on noisy and reverberant recordings. [PESTO](https://github.com/SonyCSLParis/pesto) (Riou et al., ISMIR 2023) showed that a network of under 30k parameters, equivariant to transposition, matches far larger pitch models; its code and weights are LGPL-3.0. This package trains its own weights, MIT, on license-clean data only, and hands the network's output to pYIN's HMM as its observations. The HMM and the note model stay deterministic and unchanged: the network predicts the uncertain part.

```js
import { candidates } from '@audio/neural-pitch'
import { track, notes } from '@audio/pitch-pyin'

let { times, f0, voiced } = track(samples, { fs: 44100, candidates })   // pYIN's HMM on the network
let events = notes(samples, { fs: 44100, candidates })                  // Tony's notes on it

let write = notes({ fs: 44100, candidates })                            // streaming, as pYIN streams
```

In [`audio`](https://github.com/audiojs/audio), `a.stat('notes', { robust: true })` is this `notes`, times from the call's `at`; YIN stays its default.

The network alone, every frame, no HMM:

```js
import pitch from '@audio/neural-pitch'

let { times, f0, voicing } = pitch(samples, { fs: 44100 })   // f0 on every frame, voicing 0…1
```

**Status**: experimental. 6,066 parameters, trained in 55 min on Apple MPS; JavaScript equals PyTorch within 6·10⁻⁵. Through pYIN's HMM it beats YIN under heavy noise and in rooms (Vocadito at 0 dB SNR: raw pitch 0.918 against 0.499, note onsets F 0.758 against 0.526) and trails it slightly on clean audio (0.988 against 0.991, note offsets 0.608 against 0.648). Tables below.

## Model

1. **Input** (`features.js`): a variable-Q spectrum per frame, centred on the frame's time: 288 bins, 3 a semitone from MIDI 23 (30.9 Hz) to MIDI 118⅔ (7.6 kHz), bandwidth α·f + γ with α = 2^(1/36) − 1 and γ = 12 Hz (Schörkhuber et al. 2014, window lengths as `librosa.vqt`), Hann windows by one FFT a frame and a sparse spectral kernel a bin (Brown & Puckette 1992). Windows are set in seconds, so any sample rate from 8 kHz up gives the same spectrum below its Nyquist frequency; the longest window, 84 ms, fits a 4096-point FFT at 48 kHz. dB relative to the frame's loudest bin, 80 dB deep, scaled to 0…1; the loudest bin's level in dBFS goes to the voicing head.
2. **Network** (`model.js`, `scripts/model.py`): a 9-tap convolution to 24 channels; 4 residual blocks of a 5-tap depthwise convolution (dilations 1, 2, 4, 8) and a pointwise one, LeakyReLU; a pointwise projection to 8 maps; then a Toeplitz layer as PESTO's, banded: per map one kernel over offsets from 2 octaves below to 6 above, so the harmonics of a fundamental vote for it and its subharmonics against it. Every layer commutes with a shift along frequency: a transposed input moves the posterior by as many bins. Softmax over the 288 bins is the pitch posterior; per map maximum and mean, the logits' maximum and log-sum-exp, and the level feed a 16-unit voicing head. 6,066 parameters, stored at half precision (12 KB; `weights.js`, 16.6 KB as base64). 0.92 million multiply-adds a frame in the convolutions; the Toeplitz layer runs as one FFT convolution (eight 1024-point real FFTs and one inverse).
3. **Output**: the posterior's peak refined as the mean over ±2 bins (±67 cents), weighted by the posterior (CREPE's decoding, Kim et al. 2018), and the voicing probability.
4. **Stage 1 for pYIN** (`candidates`): each frame's posterior peaks within `minFreq`…`maxFreq`, up to five, each with its mass over ±2 bins. pYIN's HMM decodes a frame voiced once a candidate's probability exceeds about 2/n, n its 0.1-semitone states (639 for 50–2000 Hz): YIN's candidate mass is near zero on noise, a network's voicing probability is not. So each candidate is weighted by the voicing odds v / (1 − v), times 2/(n + 1): the HMM then adds up the network's voicing log-odds frame by frame against its switching cost, and chooses among the peaks by their mass and its pitch continuity. The frame's `rms` is YIN's window's, which Tony's note onsets read.

## Training

Recipe and code in `scripts/` (gpu-font's: trustworthy examples, offline training, held-out evaluation, export, only the inference needed). Everything the weights saw is ours or CC BY 4.0:

- **Synthetic, exact f0** (`scripts/data.js`, `scripts/factory.js`): 8,000 phrases of 6 s (13.3 h) by audiojs synths: a singing voice (`@audio/voice-glottis` LF or Rosenberg pulses through `@audio/voice-tract`, a vowel a note, 33% of phrases), `synth-osc`, `synth-voice`, `synth-pluck`, `synth-fm` (integer ratios), `synth-modal` (string, tubes, bar), `synth-dx7` (random patches, integer ratios, OP1 at 1) and `synth-tonewheel` (8' drawn, no 16'); 8% unpitched (coloured noise, tract-filtered noise, clicks, silence). Phrases have vibrato (4–8 Hz, up to 120 cents), scoops, falls, drift, legato glides, staccato and rests, tuned off the equal-tempered grid. Every note is rendered at a fixed pitch and read at a varying rate, so its f0 is exact; the renders' f0 were checked against YIN on steady notes (`scripts/labels.js`: median offset under 2.4 cents for every source). Degraded at random: EQ, high- and low-pass, rooms (synthetic impulse responses, `reverb-freeverb`, `reverb-dattorro`, `reverb-fdn`), noise at 0–20 dB SNR (`synth-noise` white, pink, brown, blue), hum, MP3 at 32–128 kbps (`encode-mp3`), level −45 to −1 dBFS, 3% clipped; at 16, 22.05, 44.1 and 48 kHz. 1.92 million frames, one every 25 ms, stored as dB spectra at half precision (1.2 GB), rendered in 67 min on 4 workers.
- **Unlabelled singing** for the self-supervised terms: [VocalSet](https://zenodo.org/records/1193957) 1.1 (Wilkins et al., ISMIR 2018, CC BY 4.0), the first file of every singer and technique, 539 files of 18 singers (1.6 h); singers female9 and male11 (60 files) held out for development. Each frame comes as a pair: the recording and a copy through the same degradations.
- **Losses**: cross-entropy to a Gaussian of 25 cents around the exact f0 (CREPE's target) and binary cross-entropy for voicing, on the renders; PESTO's equivariance term on every pair of crops transposed by up to ±5 semitones (the projection φ(p) = Σ αʲ pⱼ, α = 2^(1/36), must scale by α per bin of transposition; Huber on the log ratio in bins); on the recordings, the degraded view's posterior must match the clean view's, transposed alike, where pYIN is sure of the clean recording (voiced probability ≥ 0.5, pitch steady within 50 cents over ±2 frames), plus pYIN's pitch and voicing there as targets at weight 0.5.
- **Run**: AdamW, one-cycle learning rate to 3·10⁻³, 16,000 steps of 1,536 renders and 768 recording pairs, on the M4 Max's GPU (PyTorch MPS, capped at 3 GB, shared with another training job): 55 min, peak process memory 1.9 GB. The checkpoint kept is the best on development data (step 15,000). A first run with 2,410 parameters (16 channels, 3 blocks, 4 maps, no teacher term) was worse on every development measure:

| Development (not benchmark) data | YIN + pYIN HMM | 2,410 parameters: + HMM / alone | 6,066 parameters: + HMM / alone |
|---|---|---|---|
| 120 synthetic clips of other seeds, degraded: raw pitch | 0.820 | 0.928 / 0.939 | **0.933 / 0.945** |
| same: voicing recall / false alarm | 0.894 / 0.120 | 0.954 / 0.093 | 0.960 / 0.097 |
| 2 held-out singers (60 files), degraded, against pYIN on the clean recording: raw pitch | 0.834 | 0.921 / 0.933 | **0.929 / 0.939** |
| same: voicing recall / false alarm | 0.865 / 0.119 | 0.940 / 0.139 | 0.945 / 0.140 |

κ was chosen there, on the smaller network: 0.5, 1 and 2 gave raw pitch 0.887, 0.928 and 0.934 with voicing false alarm 0.081, 0.093 and 0.099 on the synthetic clips, and 0.864, 0.921 and 0.929 with false alarm 0.115, 0.139 and 0.162 on the singers; 1 is the default.

## Benchmark

Protocol fixed before the first model was trained (`scripts/bench-prep.py`, `bench.js`, `bench-ref.py`, `bench-score.py`); weights, κ and the checkpoint were chosen on development data only (synthetic clips of other seeds, held-out VocalSet singers). Measurement only for the sets' licences:

- **Vocadito** (Bittner et al. 2021, CC BY 4.0): all 40 sung clips, 817 s, 44.1 kHz.
- **MDB-stem-synth** (Salamon et al., ISMIR 2017, CC BY-NC 4.0): every 5th of the 230 resynthesized stems (46: voices, strings, winds, guitars, bass), 30 s from 1 s before the first voiced frame, 44.1 kHz.
- **MIR-1K** (Hsu & Jang 2010, research use): every 4th of the 1,000 clips (250), the voice channel, 16 kHz.
- **Conditions**: clean; pink noise at 20, 10 and 0 dB SNR over the whole clip; reverb: MIT IR Survey h052, a gym weight room, T30 1.0 s, direct-to-reverberant ratio +0.7 dB (Traer & McDermott 2016), a measured room unlike the synthetic ones trained on. Pink noise is among the training degradations, so the noise rows favour this model over the untrained methods; the room is out of its training distribution.
- **Methods**: pYIN (`@audio/pitch-pyin` `track`, defaults: YIN candidates, 50–2000 Hz); this package as pYIN's stage 1 (the same HMM); librosa 1.0 `pyin` (50–2000 Hz, frame 2048 at 44.1 kHz); PESTO 2.0.1 `mir-1k_g7` (LGPL-3.0, measurement only, 10 ms, run on MPS in 10 s chunks, which matches its CPU output within 0.01 cent); this network alone (`pitch()`).
- **Metrics**: `mir_eval.melody.evaluate` 0.8.2 per clip, mean over clips: raw pitch and raw chroma accuracy (within 50 cents; chroma folds octaves), voicing recall and false alarm, overall accuracy (Salamon et al., IEEE Signal Processing Magazine 2014). A method with voicing reports f0 0 where unvoiced, so its raw pitch counts only frames it calls voiced. ¹PESTO decides no voicing and the network alone reports a pitch on every frame: their raw pitch counts every frame, an upper bound for the others' columns, not the same measure.

Raw pitch / raw chroma accuracy (bold: best of the three pYIN-HMM methods):

| Set | Condition | pYIN | this + pYIN HMM | librosa.pyin | PESTO¹ | this, network alone¹ |
|---|---|---|---|---|---|---|
| Vocadito | clean | **0.991 / 0.994** | 0.988 / 0.989 | 0.982 / 0.984 | 0.961 / 0.966 | 0.988 / 0.991 |
| Vocadito | pink 20 dB | 0.987 / 0.990 | **0.988 / 0.988** | 0.982 / 0.983 | 0.960 / 0.966 | 0.988 / 0.991 |
| Vocadito | pink 10 dB | 0.956 / 0.957 | **0.978 / 0.979** | 0.964 / 0.965 | 0.952 / 0.960 | 0.981 / 0.986 |
| Vocadito | pink 0 dB | 0.499 / 0.500 | **0.918 / 0.921** | 0.692 / 0.692 | 0.906 / 0.924 | 0.934 / 0.955 |
| Vocadito | reverb | 0.846 / 0.850 | **0.875 / 0.879** | 0.835 / 0.838 | 0.739 / 0.774 | 0.877 / 0.887 |
| MDB-stem-synth | clean | **0.932 / 0.941** | 0.931 / 0.939 | 0.916 / 0.926 | 0.841 / 0.876 | 0.968 / 0.975 |
| MDB-stem-synth | pink 20 dB | 0.917 / 0.926 | **0.920 / 0.931** | 0.906 / 0.915 | 0.835 / 0.873 | 0.957 / 0.968 |
| MDB-stem-synth | pink 10 dB | 0.853 / 0.862 | **0.883 / 0.899** | 0.859 / 0.868 | 0.799 / 0.846 | 0.923 / 0.942 |
| MDB-stem-synth | pink 0 dB | 0.478 / 0.484 | **0.752 / 0.769** | 0.593 / 0.600 | 0.674 / 0.748 | 0.802 / 0.847 |
| MDB-stem-synth | reverb | 0.836 / 0.853 | **0.840 / 0.873** | 0.807 / 0.832 | 0.682 / 0.750 | 0.856 / 0.887 |
| MIR-1K | clean | **0.973 / 0.977** | 0.963 / 0.966 | 0.964 / 0.966 | 0.980 / 0.983 | 0.967 / 0.972 |
| MIR-1K | pink 20 dB | **0.972 / 0.975** | 0.962 / 0.965 | 0.963 / 0.965 | 0.980 / 0.983 | 0.967 / 0.972 |
| MIR-1K | pink 10 dB | 0.938 / 0.941 | **0.956 / 0.958** | 0.944 / 0.946 | 0.976 / 0.980 | 0.961 / 0.966 |
| MIR-1K | pink 0 dB | 0.508 / 0.509 | **0.893 / 0.894** | 0.666 / 0.667 | 0.933 / 0.951 | 0.901 / 0.916 |
| MIR-1K | reverb | 0.829 / 0.835 | **0.860 / 0.862** | 0.825 / 0.828 | 0.815 / 0.837 | 0.859 / 0.865 |

Voicing recall / false alarm / overall accuracy:

| Set | Condition | pYIN | this + pYIN HMM | librosa.pyin | this, network alone |
|---|---|---|---|---|---|
| Vocadito | clean | 0.997 / 0.264 / 0.906 | 0.996 / 0.296 / 0.895 | 0.994 / 0.345 / 0.873 | 0.999 / 0.464 / 0.838 |
| Vocadito | pink 20 dB | 0.995 / 0.163 / 0.937 | 0.996 / 0.226 / 0.918 | 0.994 / 0.246 / 0.906 | 0.998 / 0.354 / 0.875 |
| Vocadito | pink 10 dB | 0.965 / 0.094 / 0.939 | 0.987 / 0.142 / 0.939 | 0.979 / 0.171 / 0.919 | 0.993 / 0.248 / 0.904 |
| Vocadito | pink 0 dB | 0.508 / 0.019 / 0.657 | 0.930 / 0.063 / 0.924 | 0.712 / 0.066 / 0.770 | 0.958 / 0.167 / 0.891 |
| Vocadito | reverb | 0.947 / 0.613 / 0.687 | 0.976 / 0.653 / 0.694 | 0.951 / 0.663 / 0.663 | 0.994 / 0.780 / 0.652 |
| MDB-stem-synth | clean | 0.970 / 0.050 / 0.945 | 0.945 / 0.039 / 0.944 | 0.968 / 0.097 / 0.927 | 0.989 / 0.051 / 0.970 |
| MDB-stem-synth | pink 20 dB | 0.957 / 0.037 / 0.934 | 0.940 / 0.031 / 0.936 | 0.958 / 0.085 / 0.920 | 0.983 / 0.069 / 0.956 |
| MDB-stem-synth | pink 10 dB | 0.897 / 0.035 / 0.887 | 0.909 / 0.018 / 0.911 | 0.916 / 0.068 / 0.889 | 0.959 / 0.073 / 0.926 |
| MDB-stem-synth | pink 0 dB | 0.538 / 0.005 / 0.590 | 0.784 / 0.002 / 0.813 | 0.659 / 0.043 / 0.684 | 0.872 / 0.077 / 0.817 |
| MDB-stem-synth | reverb | 0.936 / 0.563 / 0.751 | 0.939 / 0.344 / 0.814 | 0.936 / 0.619 / 0.714 | 0.979 / 0.422 / 0.804 |
| MIR-1K | clean | 0.985 / 0.256 / 0.898 | 0.982 / 0.182 / 0.915 | 0.993 / 0.353 / 0.862 | 0.991 / 0.433 / 0.840 |
| MIR-1K | pink 20 dB | 0.983 / 0.071 / 0.961 | 0.981 / 0.075 / 0.953 | 0.992 / 0.190 / 0.920 | 0.990 / 0.187 / 0.921 |
| MIR-1K | pink 10 dB | 0.952 / 0.031 / 0.948 | 0.973 / 0.041 / 0.958 | 0.977 / 0.142 / 0.921 | 0.984 / 0.137 / 0.930 |
| MIR-1K | pink 0 dB | 0.520 / 0.005 / 0.648 | 0.908 / 0.006 / 0.923 | 0.700 / 0.066 / 0.744 | 0.938 / 0.124 / 0.884 |
| MIR-1K | reverb | 0.960 / 0.631 / 0.696 | 0.973 / 0.625 / 0.725 | 0.968 / 0.683 / 0.677 | 0.991 / 0.762 / 0.679 |

Notes, Tony's model on each stage 1, Vocadito, F-measure by `mir_eval.transcription` (onset within 50 ms; + pitch within 50 cents; + offset within 20% of the note or 50 ms), mean over 40 clips and both annotators:

| Condition | pYIN: onset / + pitch / + offset | this + pYIN: onset / + pitch / + offset |
|---|---|---|
| clean | 0.825 / 0.774 / **0.648** | **0.828 / 0.776** / 0.608 |
| pink 20 dB | 0.826 / 0.770 / **0.665** | **0.828** / 0.770 / 0.637 |
| pink 10 dB | 0.795 / 0.743 / 0.603 | **0.807 / 0.753 / 0.620** |
| pink 0 dB | 0.526 / 0.485 / 0.304 | **0.758 / 0.700 / 0.530** |
| reverb | 0.628 / 0.574 / **0.271** | **0.671 / 0.620** / 0.261 |

Reading:

- **Where YIN still wins: clean and lightly noisy audio.** pYIN's own raw pitch is 0.3 points higher on clean Vocadito and 1 point higher on clean and 20 dB MIR-1K, and its note offsets are better on clean, 20 dB and reverberant Vocadito (0.648 against 0.608 clean). The network's voicing lasts into note tails: voicing false alarm on clean Vocadito 0.296 against 0.264, probably because its windows, up to 84 ms, reach past a note's end. On clean and 20 dB audio pYIN stays the better default.
- **Where the network wins: heavy noise and rooms.** At 0 dB SNR pYIN's voicing collapses (recall about 0.5 on every set) and the network keeps 0.89–0.92 raw pitch on the voices, 0.75 on the instruments; Vocadito notes go from 0.53 to 0.76 onset F. At 10 dB it gains 1.8–3.0 points of raw pitch; in the measured room 0.4–3.1 points, and on MDB-stem-synth 0.22 less voicing false alarm.
- **Against PESTO.** On Vocadito and MDB-stem-synth the network alone beats PESTO's `mir-1k_g7` in every condition (for example 0.968 against 0.841 raw pitch on clean MDB-stem-synth, 0.877 against 0.739 in the room on Vocadito). On MIR-1K, the data PESTO's weights were trained on, PESTO leads by 1–3 points except in the room.
- **The network alone** has the best raw pitch in most rows but no memory across frames; through pYIN's HMM it loses a little raw pitch (frames called unvoiced) and gains voicing precision and overall accuracy.

## Speed

`node scripts/speed.js`: one thread, CPU seconds per second of audio (process CPU time), a minute of rendered singing, Apple M4 Max shared with other work (load average about 30):

| | features | network | `pitch()` | `track` + `candidates` | `track`, YIN |
|---|---|---|---|---|---|
| 44.1 kHz | 0.042 | 0.199 | 0.229 | 0.320 | 0.038 |
| 16 kHz | 0.046 | 0.290 | 0.339 | 0.297 | 0.015 |

At pYIN's 5.8 ms hop the network runs 172 times a second: 0.33 ms a frame on a performance core, up to about 1.2 ms under load; the spectrum takes 0.17–0.19 ms. So pYIN on this stage 1 costs about 8 times pYIN on YIN and still runs three times faster than real time on one busy core. With 6,066 weights and 0.9 million multiply-adds a frame, a GPU would spend its time on transfers and dispatch, and WebGPU is not available inside an AudioWorklet; there is no GPU path. The stage reads 2048 samples ahead of a frame's time at 44.1 kHz (46 ms, YIN: 35 ms); pYIN's Viterbi then decides frames as their paths meet.

## Verification

- **JavaScript against PyTorch** (`test.js`, `fixtures/parity.json` from `scripts/export.py`): on 16 development frames, the logits agree within 6.1·10⁻⁵ and the voicing within 3.8·10⁻⁶; PyTorch runs the exported half-precision weights in float32, JavaScript sums in another order and runs the Toeplitz layer as an FFT convolution.
- **Features**: a full-scale sine peaks at its bin at −6.02 dB at 16, 22.05, 44.1 and 48 kHz; a tone's input at 16 and at 48 kHz differs by at most 0.5 dB in the bins within 60 dB of its peak.
- **With pYIN**: vibrato followed within a few cents, a played line in noise gives its five notes with onsets within 50 ms, and streaming in odd blocks equals one call, frame for frame.
- **pYIN without it is unchanged**: `track` and `notes` (whole, streamed, and with a custom range and hop) on 7 Vocadito clips and 3 synthetic signals hash to the same SHA-256 before and after the `candidates` option was added.

## Limitations

- **Clean audio**: pYIN on YIN is slightly better (raw pitch 0.991 against 0.988 on Vocadito, 0.973 against 0.963 on MIR-1K) and ends notes more precisely; this stage's voicing runs into note tails (voicing false alarm 0.296 against 0.264 on clean Vocadito). Keep YIN as the default; use this where noise or rooms are expected.
- **Instruments in heavy noise**: 0.75 raw pitch at 0 dB SNR on MDB-stem-synth; synthetic plucked, struck and FM tones are its weakest development classes (0.89–0.93 raw pitch against 0.97 for the voice).
- **Training data**: audiojs synths and 1.6 h of VocalSet; no recorded instruments. On MIR-1K, where PESTO's weights come from, PESTO's network is 1–3 points better.
- **Range**: fundamentals from 30.9 Hz to about 2.6 kHz; inputs up to 7.6 kHz (a 16 kHz file loses nothing, an 8 kHz one the bins above 4 kHz).
- **Voicing weight κ** (option of `candidates`, default 1): 2 raises recall and false alarms, 0.5 lowers both (development data, above).
- **Time resolution**: windows of up to 84 ms blur fast glides and note ends that YIN's 23 ms window keeps.

## Reproduce

Data lands in `~/.cache/audiojs/data/neural-pitch` (`$NP_DATA`), the benchmark in `~/.cache/audiojs/data/neural-pitch-bench` (`$NP_BENCH`); Python needs torch, soundfile, scipy, librosa and mir_eval.

```sh
node scripts/labels.js                                   # the renders' f0 against YIN
node scripts/factory.js synth train 8000 100000          # 1.92M frames, 1.2 GB
node scripts/factory.js synth dev 400 900000
node scripts/factory.js real real vocalset-train.txt 500000     # paths of VocalSet wavs, one a line
node scripts/factory.js real realdev vocalset-dev.txt 600000    # singers female9, male11
python scripts/train.py --out runs/B --real real --realdev realdev --steps 16000 --batch 1536 \
  --C 24 --T 8 --blocks 1,2,4,8 --D1 72 --D2 216 --teacher 0.5
python scripts/export.py runs/B/best.pt                  # weights.js, fixtures/parity.json
node scripts/dev.js 120 1                                # development check ($NP_VOCAL_DEV: held-out list)
python scripts/bench-prep.py                             # benchmark audio and conditions
node scripts/bench.js ours                               # also pyin, ours-raw, pyin-notes, ours-notes
python scripts/bench-ref.py librosa                      # PESTO: from its own environment
python scripts/bench-score.py
node scripts/speed.js
```

The VocalSet subset is the first file, by sorted path, of each singer's technique folder in VocalSet 1.1's zip: 599 files.

## Reference

M. Mauch, S. Dixon, "pYIN: a fundamental frequency estimator using probabilistic threshold distributions," ICASSP 2014. · A. Riou, S. Lattner, G. Hadjeres, G. Peeters, "PESTO: pitch estimation with self-supervised transposition-equivariant objective," ISMIR 2023, [arXiv:2309.02265](https://arxiv.org/abs/2309.02265). · J. W. Kim, J. Salamon, P. Li, J. P. Bello, "CREPE: a convolutional representation for pitch estimation," ICASSP 2018. · C. Schörkhuber, A. Klapuri, N. Holighaus, M. Dörfler, "A Matlab toolbox for efficient perfect reconstruction time-frequency transforms with log-frequency resolution," AES 53rd Conference, 2014. · J. C. Brown, M. S. Puckette, "An efficient algorithm for the calculation of a constant Q transform," JASA 92(5), 1992. · C. Raffel et al., "mir_eval," ISMIR 2014. · J. Salamon, E. Gómez, D. P. W. Ellis, G. Richard, "Melody extraction from polyphonic music signals," IEEE Signal Processing Magazine 31(2), 2014.

**Use when:** the melody of one voice or instrument where YIN struggles: noise, rooms, thin or band-limited recordings; as pYIN's stage 1 (`track`, `notes`), or frame-wise posteriors for your own decoding.<br>
**Not for:** polyphony (one pitch a frame: [`@audio/neural-transcribe`](../neural-transcribe) for chords); a voice inside a mix without separating it first; fundamentals below 30.9 Hz or above about 2.6 kHz (trained on 32 Hz – 2.2 kHz, transposed by up to 5 semitones).

---

Part of the [@audio/neural](https://github.com/audiojs/neural) lane.

MIT © [audiojs](https://github.com/audiojs)
