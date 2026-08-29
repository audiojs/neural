# Todo — neural lane

## M1 — Measurement foundations
- [x] Probe atom: click + ESS sweep + 4-level noise + amplitude-ramp triangle, deterministic, with segment layout — `neural-capture/probe` (consider migrating to `@audio/measure`)
- [x] Dry/wet pair alignment: GCC-PHAT lag detection + trim built into `capture()` (plain cross-correlation smears on non-white probes), lag reported as `result.latency`
- [x] Null-test verb: `esr` + `nullDepth` shipped in `neural-capture` (consider migrating to `@audio/quality`)

## M2 — Capture: LTI rung
- [x] Deconvolution: Wiener seed + exact CG least squares (unbiased on truncated records) — `neural-capture` rung 1; ESS variant already in `@audio/measure-ir`
- [x] Zero-latency playback: `@audio/reverb-convolution` (uniform-partitioned FFT) — reused, nothing to build
- [x] Validate: multi-tap −148 dB, 3000-tap reverb tail −118 dB (synthetic); sox black boxes: EQ −57, lowpass −88, reverb −49 (`neural-capture/validate.js`)

## M3 — Capture: nonlinear rungs
- [x] Static NL curve fit (Hammerstein: Chebyshev curve + IR, alternating LS) — tanh→FIR nulls −70 dB
- [x] Differentiable W–H cascade — biquads→Chebyshev curve→biquads; curve solved by exact LS each iteration (linear in coeffs — avoids the classic W–H local minimum), Adam on filter params with analytic adjoint gradients (finite-difference-tested); emits `@audio/biquad` cascade stages; skipped when identified memory depth exceeds biquad reach
- [x] `.nam` WaveNet playback (`neural-amp`): weight order pinned by hand-computed micro-model + exact-count parse; real 5150 capture validated (idle −73 dB steady, ~1.4× realtime plain JS)
- [ ] Causal TCN capture rung (train in JS, fallback when formulas fail) — 5150 validation shows the need: formula rungs top out at −5 dB on boosted high-gain
- [x] Long-tail factorization: `wh-tail` rung — short-memory wh alternated with exact LS tail solve; biquad→tanh→FIR device −54 dB where hammerstein stalls at −13
- [x] Staged identifier: auto-escalation + per-rung attempts report — `neural-capture` (rungs 1–2; W–H/TCN slot behind same interface)
- [ ] Validate: amp, compressor, tape captures at ESR ≤ 1% — sox proxies pass (overdrive −39 dB = 0.013% ESR via `wh-tail`; compander −23 dB, adaptive-class ceiling); real NAM 5150 shows the boosted-high-gain limit (formula rungs −5 dB → TCN rung needed); real hardware pairs still open

## M4 — Runtime
- [ ] `neural-runtime`: WASM (RTNeural or own kernels) in AudioWorklet; benchmark NAM standard/feather profiles against 128-sample quantum
- [ ] Grey-box chain player (plain audiojs pipeline — trivially real-time)
- [ ] Browser `.nam` player demo

## M5 — Pretrained: denoise / dereverb
- [ ] Weights license audit (RNNoise, DeepFilterNet variants, Demucs) before any promise
- [ ] `neural-denoise` worklet: RNNoise baseline → DeepFilterNet ceiling
- [ ] Dereverb: pick/audit model (DeepFilterNet3 handles some; else dedicated late-reverb suppressor)
- [ ] Differentiate vs sapphi-red/web-noise-suppressor (quality + audiojs integration)

## M6 — Synth matching (automatic synthesizer programming — for mel)
- [x] Synth interface: `match(render, target, {bounds, budget})` — param vector → offline render fn, any knob-synth qualifies (`neural-synth`)
- [x] Perceptual loss atom: multi-scale spectral convergence + log-mel band distance (the mel term is what makes pitch/cutoff optimizable)
- [x] Derivative-free optimizer: full CMA-ES (Hansen), deterministic, scattered restarts + polish stage; recovers subtractive patch to 220/1200.5/3.0 and FM patch to 330/2.00/1.50
- [ ] Neural inverse model per synth: sample random patches → render → train audio→params predictor; plugs into `match` via `seeds`
- [ ] Differentiable path: DDSP-style synths get direct gradient descent
- [ ] Wire into mel: target vector sound → configured synth patch

## M7 — Ship / moat
- [ ] Browser capture UX: record pair → staged identify → null report
- [ ] Capture → jz compile → VST/AU (the "pay to compile out" boundary)
- [ ] Grey-box chain JSON spec published; `.nam` interop documented
- [ ] Defer: time-varying capture (chorus/phaser conditioning) until core rungs null well

## Backlog

* https://sessionloops.com/pitchnet
