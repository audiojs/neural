# @audio/neural-restore

> A studio-grade voice from a bad recording: speech or singing, English, French or Russian, recorded in a room, in
> noise, on a phone or through a codec, back as a 48 kHz full-band take. One network, one pass, in the STFT.

**Status: trained; the float16 weights hosted at [audiojs/restore](https://huggingface.co/audiojs/restore).** 300 000 steps on the laptop's Apple GPU, 2026-10-15 to 2026-10-22. The chosen
weights are step 290 000 (`scripts/runs/restore/net-0290000.pt`: best SI-SDR and mel distance on the validation draw;
270k–300k are within each other's noise on every measure). Exported to `~/.cache/audiojs/neural/restore/`:
`restore.onnx` (70 MB, float32; 108–121 dB from PyTorch on five real takes, one of them 16 s in chunks) and
`restore.fp16.onnx` (35 MB, float16 weights computed in float32; 68–74 dB from PyTorch, the same scores to 0.1 dB SI-SDR).
`restore.js` is the pipeline in JavaScript (STFT, 10 s runs crossfaded, the level, the network through
`@audio/neural-runtime`): 89 dB from PyTorch on the fixture with float32 weights, 73 dB with float16 in Node, 70 dB on
WebGPU and wasm in the browser; 10.5 s restored in 0.36 s on the laptop's WebGPU, 1.3 s on wasm (`node test.js`).
`audio`'s `restore(strength)` op and the playground's Process › Repair › Restore run it. The benches against Adobe
follow [`scripts/HANDOUT.md`](scripts/HANDOUT.md) section 4. The plan and prior art are in `audio`'s `.work/restore.md` (PR audiojs/audio#72).

```
npm install @audio/neural-restore onnxruntime-node   # onnxruntime-web in the browser
```

```js
import restore from '@audio/neural-restore'

let y = await restore(samples, { sampleRate: 44100 })   // one channel: restored, at its rate and length
```

The model file is hosted at [audiojs/restore](https://huggingface.co/audiojs/restore) (`MODEL`, at the commit
`REVISION` pins): without `opts.weights` it is fetched once (35.9 MB), refused unless of the SHA-256 `SHA256`, and in
Node kept in the neural cache (`~/.cache/audiojs/neural/restore/`, or `$AUDIO_NEURAL_CACHE`, where `scripts/export.py`
writes it too) and read from there after, offline as well; a browser keeps it in the Cache API. `opts.weights` reads
another copy: a URL, a path in Node, or the bytes of `restore.fp16.onnx`. Options: `{ sampleRate = 48000, weights,
backend, progress, fetch }`; `backend` 'auto' takes WebGPU in a page that has it, else wasm, and Node's runtime in Node.
The code is MIT; the weights are OpenRAIL: trained on Dialogs, whose licence binds models trained on it, and the audio
they make, to its use restrictions ([LICENSE](https://huggingface.co/audiojs/restore/blob/main/LICENSE)).

## Design, in one paragraph

Vocos's ConvNeXt stack over STFT frames (dim 512, 8 blocks) with two self-attention layers for a room's tail, reading
each frame's log magnitude and phase and writing a change of log magnitude (unbounded: a band can be taken down 60 dB
or raised from nothing) and a rotation of the input's phase (the identity at init: what is there keeps its phase
unless the net turns it); 17.9M parameters, the STFT (2048/512 at 48 kHz) outside the graph as
`@audio/neural-separate`'s models have it. Trained as a regressor first (multi-resolution STFT, log-mel, a compressed complex L1 on the model's grid for the
phase, WavLM Base+ features: FINALLY's recipe), then adversarially against HiFi-GAN's multi-period and UnivNet's multi-resolution
discriminators with feature matching. Every loss sees target and estimate through a low-pass at the target's own
measured bandwidth, so 24 kHz corpora train beside full-band ones in one run. The recordings are made on the fly from
clean voices: a room (a measured response or a synthetic one), noise, a cheap microphone, a compressor, clipping,
a band limit, MP3 / AAC / Opus, 8–16-bit words.

## Data (what trains, by licence)

| Set | Hours | Rate | Licence | Role |
|---|---|---|---|---|
| VCTK 0.92, mic1, speakers p232 and p257 held out | 44 | 48 kHz | CC BY 4.0 | English, full band |
| LibriTTS-R train-clean-100 | 54 | 24 kHz | CC BY 4.0 | English, to 12 kHz |
| SIWIS | 10 | 44.1 kHz | CC BY 4.0 | French, one professional voice |
| Dialogs (langswap) | 17.4 | 44.1 kHz | OpenRAIL (commercial use allowed; its use restrictions flow down to derived models) | Russian, three actors, studio; its manifest is separate so a licence-clean variant is one flag away |
| Supertonic 3 (Supertone, OpenRAIL-M weights), ten voices, public-domain text | ru 12, fr 8, en 4 | 44.1 kHz | the synthesised audio is ours | the only full-band Russian beyond Dialogs; DNSMOS OVRL 3.43 / 3.40 / 3.39 (ru / fr / en) against 3.39 for VCTK's clean test set and 3.20 for Dialogs |
| Pocket TTS (Kyutai, CC BY 4.0), eight voices | fr 6, en 3 | 24 kHz | ours | more voices, to 12 kHz |
| VocalSet (the local subset) | 1.8 | 44.1 kHz | CC BY 4.0 | singing, full band |
| MIT IR Survey, the even-numbered rooms | 135 rooms | 32 kHz | unstated: training only, never redistributed | rooms; the odd-numbered ones are `audio`'s bench rooms and stay out |
| MUSAN noise | 6 | 16 kHz | CC BY 4.0 | noise |

Validation pairs come from VoiceBank's clean test set (p232, p257): the same speakers `audio`'s benches score, never
trained on. A measured room or noise recorded at 16 or 32 kHz is continued to 24 kHz with synthetic content of the
same decay or colour at the level it has under its edge (`degrade.extend_rir`, `extend_noise`), so a measured room
never low-passes the voice by its own rate.

Texts read by the synthetic voices: Project Gutenberg (Austen, Dickens, Melville, Doyle; Hugo, Flaubert, Dumas,
Proust; four Russian titles), Dialogs' own transcripts, and Tatoeba (CC BY 2.0 FR) once its export lands.

## Training on the laptop, measured

Apple M4 Max, 36 GB, PyTorch 2.14 on MPS, batch 8 × 4 s, the full model, WavLM loss on:

| Phase | Steps a second | Memory |
|---|---|---|
| regression (to step 20 000) | 2.2 | 1 GB |
| adversarial, discriminators on one random 0.5 s window of each item | 0.68 | 6 GB |

The discriminators and WavLM are the cost, not the generator: dim 384 and dim 512 cost the same step, so the laptop
trains the full model directly (the handout's stage A is for a CUDA box). The discriminators judge a window rather
than the whole 4 s segment: HiFi-GAN and Vocos train on under a second, and the whole segment took 6.6 s a step and
28 GB. 300 000 steps is about five days of the laptop; the run resumes from its last checkpoint after any interruption.

Found on MPS and worked around in `dsp.py`, each verified against the CPU's gradients (relative error 1e-4 to 1e-3
after; before, 44 % and non-finite at random):
- `F.pad(mode='reflect')` has a wrong backward: `pad_reflect` does the same padding by slices and flips, and every
  STFT (the net's, the losses', the discriminators') is centered with it;
- complex `.abs()` has no gradient where a band-limited target's spectrum is exactly zero: `mag` is
  `sqrt(re² + im² + ε)`;
- `torch.istft` is written out (frames by irfft, windowed, overlap-added by `F.fold`), within 1e-7 of torch's;
- a batch must not be copied with `non_blocking=True` (the loader's buffer is freed under the pending copy), and the
  discriminators' windows are taken by slices, not `gather`.

## Measured

24 validation pairs: VoiceBank's test speakers through the chain (rooms, noise, microphones, codecs, band limits),
each output at the reference's level, scored through the pair's band limit. The baselines are `audio`'s own ops on the
same inputs (`scripts/baselines.mjs`); `scripts/compare.py` makes the table.

| System | LSD ↓ | SI-SDR ↑ | PESQ ↑ | DNSMOS SIG | BAK | OVRL | band, Hz |
|---|---|---|---|---|---|---|---|
| input | 3.30 | −5.2 | 1.77 | 2.50 | 2.41 | 1.97 | 16 947 |
| `deepfilter()` | 2.85 | −2.1 | 2.13 | 2.93 | 3.75 | 2.59 | 17 104 |
| `deepfilter(0)` | 2.71 | −2.1 | 2.07 | 2.91 | 3.90 | 2.63 | 16 366 |
| `auto('speech')` | 2.91 | −4.0 | 1.88 | 2.91 | 3.29 | 2.45 | 16 265 |
| restore, step 50k as trained (phase predicted from scratch) | 1.48 | −5.5 | 1.93 | 2.91 | 3.90 | 2.62 | 24 000 |
| restore, step 50k with the input's phase kept (the residual head, before any further step) | 1.75 | −2.9 | 2.02 | 3.07 | 3.89 | 2.77 | 23 092 |
| restore, step 150k | 0.80 | +5.6 | 2.26 | 3.12 | 3.91 | 2.81 | 24 000 |
| restore, step 200k | 0.85 | +5.8 | 2.36 | 3.16 | 3.92 | 2.85 | 24 000 |
| restore, step 290k (shipped weights) | 0.79 | +6.0 | 2.40 | 3.21 | 3.91 | 2.89 | 24 000 |
| reference | 0 | — | 4.64 | 3.29 | 3.89 | 2.95 | 24 000 |

At a sixth of the run the restorer is ahead of DeepFilterNet3 on DNSMOS and on bandwidth (the band above 17 kHz comes
back; the LSD halves) and within 0.05 PESQ and 0.8 dB SI-SDR of it. The first 50k steps predicted phase from scratch
and never moved SI-SDR off the input's; keeping the input's phase and learning a rotation (the head's bias at the
identity, grafted onto the checkpoint) gave 2.6 dB at once and is what trains from step 50k on. SI-SDR against a dry
reference is a hard measure for this chain (rooms and codec lags put the input itself at −5 dB); the listen decides.

## Run

```
bash scripts/run.sh venv      # the trainer's venv and the TTS venv (uv, Python 3.12); scripts/test.py must print ok
bash scripts/run.sh fetch     # the corpora, one at a time, unpacked and the archives deleted as they land
bash scripts/run.sh texts     # public-domain text → utterance lists
bash scripts/run.sh tts 8     # synthetic voices in the background (Supertonic ru fr en, Pocket fr en), 8 h each
bash scripts/run.sh scan      # manifests, each file's bandwidth
bash scripts/run.sh listen    # 20 pairs in DATA/listen: the half hour that decides more than any hyperparameter
bash scripts/run.sh b         # the run (resumes); bash scripts/run.sh watch folds in corpora as they land
bash scripts/run.sh export RUN
```

Progress: `runs/restore/log.jsonl` (a line per 50 steps, `val` lines every 2000 with LSD, SI-SDR and log-mel distance
on the VoiceBank pairs) and `runs/restore/samples/` (`<step>-<k>-{in,out,ref}.wav`).

| File | What |
|---|---|
| `scripts/dsp.py` | the STFT (2048/512), its padding and inverse, levels, the band limit every loss passes through, mel, resampling, the bandwidth a file carries |
| `scripts/degrade.py` | the degradation chain; T60 and the band continuation of measured rooms and noises |
| `scripts/data.py` | manifests, the room bank, the noise bank read on demand, the pair sampler |
| `scripts/model.py` | the generator and the discriminators |
| `scripts/loss.py` | multi-resolution STFT, log-mel, WavLM features, LS-GAN, feature matching |
| `scripts/train.py` | the run |
| `scripts/tts.py` | texts → utterances → synthetic voices (Supertonic 3, Pocket TTS) |
| `scripts/export.py`, `infer.py`, `score.py`, `test.py` | ONNX export (`[1, 2, 1025, T]` re/im, as neural-separate), chunked inference, scores, the smoke test |
| `scripts/run.sh`, `watch.sh` | the whole run as commands; the watcher |
