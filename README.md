# @audio/neural

> The opt-in ML lane. All planned — gated on policy, not code.

| Package | What |
|---|---|
| `@audio/neural-runtime` | one inference adapter (ONNX Runtime Web / tflite), worklet-ready |
| `@audio/neural-denoise` | RNNoise / DeepFilterNet class |
| `@audio/neural-amp` | NAM-class amp captures |
| `@audio/neural-separate` | stems (Demucs / Open-Unmix class) |

**Policy** (keeps the classical stance honest): classical tools never require this lane; weights are hosted separately and licensed-audited before any promise (many audio models are research-only — the freemium "premium ML weights" conflict in the site todo resolves here); deterministic pipelines stay classical. MIR's deferred ML tier (genre/mood/tags/separate) lands here when it lands.
