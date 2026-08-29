# @audio/neural

> The opt-in ML lane: capture (dry/wet → formula) + pretrained weights.

| Package | What | Status |
|---|---|---|
| `@audio/neural-capture` | staged identifier: dry/wet pair → cheapest audiojs formula (IR → Hammerstein → W–H → W–H+tail → TCN) + null report; probe signal + IR-onset alignment | rungs 1–4 shipped, tested vs sox + real NAM amp |
| `@audio/neural-amp` | NAM `.nam` WaveNet playback, dependency-free | shipped, real-capture verified |
| `@audio/neural-synth` | sound matching: configure any knob-synth to a target sound (CMA-ES + mel-spectral loss) | shipped, patch recovery verified |
| `@audio/neural-runtime` | one inference adapter: ONNX Runtime (onnxruntime-node / onnxruntime-web wasm + webgpu), worklet-ready, cached model fetch (`~/.cache/audiojs/neural`, `$AUDIO_NEURAL_CACHE`) | shipped, 11/11 tests |
| `@audio/neural-asr` | Whisper speech-to-text (transformers.js / ONNX Runtime), segment + word timestamps, `cues` bridge to `@audio/subtitle` | shipped, real-inference verified (whisper-tiny, MIT weights) |
| `@audio/neural-align` | forced alignment: pure-JS CTC trellis (torchaudio `forced_align` semantics) + wav2vec2-base-960h adapter (Apache-2.0) → word timestamps, enhanced LRC | shipped, core tested without a model, adapter verified live |
| `@audio/neural-separate` | stems: Open-Unmix-class spectrogram masking + multichannel Wiener EM (norbert port), Demucs-class waveform path, through `neural-runtime` | pipeline shipped & tested (oracle masks + real ONNX); weights not bundled — `scripts/export-openunmix.py` provided, unrun |
| `@audio/neural-diarize` | who spoke when: `@audio/vad` regions → WavLM speaker embeddings (MIT) → agglomerative clustering → speaker segments, VTT `<v>` voice tags | shipped, clustering tested without a model, adapter verified live |
| `@audio/neural-tts` | text to speech: SpeechT5 (MIT) via transformers.js, sentence chunking, any output rate | shipped, real-synthesis verified |
| `@audio/neural-denoise` | RNNoise / DeepFilterNet class | planned |

See `research.md` for theory (Boyd–Chua feasibility boundary, device-class ladder) and `todo.md` for the plan.

**Policy** (keeps the classical stance honest): classical tools never require this lane; weights are hosted separately and licensed-audited before any promise (many audio models are research-only — the freemium "premium ML weights" conflict in the site todo resolves here); deterministic pipelines stay classical. MIR's deferred ML tier (genre/mood/tags/separate) lands here when it lands.
