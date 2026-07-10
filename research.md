# Research — neural lane

## Vision

One capture pipeline: feed it a dry/wet pair (same audio without and with a device — plugin, amp, pedal, tape machine, speech enhancer, feedback suppressor), get back a **zero-latency formula** that imitates the device's transfer behavior, with memory windows up to reverb scale (3–4 s). The device is a black box; the output is ideally not a black box — the cheapest structure that nulls against the wet signal.

Two product families fall out of this:

1. **Capture** — train per device from a dry/wet pair (`neural-amp` today; generalizes to `neural-capture`).
2. **Pretrained** — ship audited weights for tasks with no closed-form (`neural-denoise`, `neural-separate`).

## Theory — when is it possible

Boyd & Chua 1985 ("fading memory" theorem): any operator that is **causal**, **time-invariant**, and has **fading memory** (distant past matters less and less) can be approximated arbitrarily well by a finite Volterra series — equivalently by a finite causal network of delays + static nonlinearities. Those three conditions are the entire feasibility boundary:

- **Causal + TI + fading memory** → learnable from dry/wet pairs. Covers EQs, cabs, reverbs, amps, compressors, tape, saturation. Fading ≠ short: a 4 s reverb tail decays, so it qualifies.
- **Lookahead** (limiters) → wet depends on future dry; matchable only at the device's own latency. "Zero *added* latency" is the honest ceiling.
- **Time-varying** (chorus/phaser/wow) → breaks TI; the LFO phase during recording is arbitrary. Still identifiable, but the modulator must become explicit state: extract the modulation trajectory, fit a TI model conditioned on it (Wright & Välimäki, JAES 2021).
- **Stochastic** (analog noise, random LFOs, spring chaos) → matchable in distribution only; null tests never fully null.
- **Adaptive/inferential** (speech enhancement, defeedback, denoise) → still deterministic functions of input history, so approximable in principle — but the underlying computation is *inference*, not a circuit. No compact formula exists; the clone is a DNN roughly the size of the one inside the device. Real complexity barrier, not a tooling gap.
- **Chaotic feedback circuits** (fuzz at self-oscillation) → sensitive dependence; sample-exact match impossible, perceptual match still works.

Coverage caveat: the system is learned only on the signal manifold excited during capture. Excitation design (sweeps + noise at multiple drive levels + program material) matters more than architecture — NAM's standardized ~3 min training file is exactly this.

## Device-class ladder — cheapest formula per class

| Class | Examples | Method | Formula |
|---|---|---|---|
| LTI | EQ, cab, reverb, room | deconvolution `H = W(ω)/D(ω)` (regularized) or ESS (Farina 2000) | impulse response — exact, solved; 3–4 s tail is just a long FIR; zero latency via partitioned convolution, direct-form first partition |
| Static NL | waveshaper, clipper | fit scalar curve | `y = f(x)` — polynomial/spline/tanh |
| NL + short memory | amps, comps, tape, transformers | grey-box Wiener–Hammerstein (differentiable biquads + waveshapers, Kuznetsov/Parker/Esqueda DAFx-20) → causal TCN (NAM) → LSTM (GuitarML) | 2–3 W–H stages capture a shocking fraction of real gear in dozens of coefficients; TCN is the fallback |
| NL + long memory | saturated echo, plate/spring, amp+room | factor: NL short-memory block → linear long tail; or per-harmonic IRs from synchronized swept sine (Novak et al. 2010 — the "dynamic convolution" behind Acustica Nebula / Sintefex) | hybrid cascade — matches how physics composes |
| Time-varying | chorus, phaser, flanger | extract LFO trajectory, condition TI model on phase | per-family structure, no universal recipe |
| Adaptive | denoise, defeedback, de-ess by inference | pretrained DNN lane, not capture | no compact formula |

## The staged identifier (product concept)

Try the cheapest rung; escalate only when the null test fails:

1. Deconvolve → LTI? residual under threshold → done (formula = IR)
2. Static NL + IR (Hammerstein) → done?
3. Differentiable W–H cascade — biquads + waveshapers fit by gradient descent — **the zero-latency formula for most real devices**
4. Causal TCN (NAM-style), receptive field sized to the memory depth measured in step 1's residual
5. Long tails: split stage-3/4 output into a linear convolution tail

Success metric per rung: **null depth**, `wet − model(dry)` in dB (report alongside ESR). LTI deconvolution nulls very deep (−60 dB and beyond); good NAM captures reach ESR ~1% (≈ −20 dB residual) and are widely judged indistinguishable in program material. Stage 4 can often be distilled back down once true memory depth is known (prune, or symbolic-regress the TCN's behavior).

**The audiojs angle** — this is the moat: stage 3's output is *literally an audiojs chain* — biquads from `@audio/filter`, curves from `@audio/saturate`, gains, delays. The capture result is an inspectable, dependency-free, zero-latency pipeline of existing atoms, not an opaque weights blob. And jz can compile that chain to VST/AU — capture in the browser, ship as a plugin. No competitor's capture output is readable or compileable; NAM/ToneX all emit weights.

## Architectures survey

- **WaveNet-class TCN** (dilated causal convs) — NAM's core (Atkinson, NeuralAmpModelerCore, MIT). Sample-causal → zero latency by construction; receptive field tens of ms; the current amp-capture quality bar. C++ core; compiles to WASM (community web players exist).
- **LSTM/GRU** — GuitarML Proteus, AIDA-X (both on Wright's Automated-GuitarAmpModelling lineage); smaller/cheaper than TCN, slightly lower ceiling; runs via RTNeural.
- **micro-TCN** — Steinmetz & Reiss 2021: dynamics processors (compressors) with long-ish memory at tiny compute.
- **Differentiable IIR / grey-box W–H** — Kuznetsov, Parker, Esqueda DAFx-20; also state-trajectory networks (Parker et al. DAFx-19). The compact-formula lane.
- **DDSP** (Engel et al., ICLR 2020) — differentiable oscillators/filters/reverb; relevant for neural-synth ("learn any sound into a synth") more than device capture.
- **State-space models** (S4-class linear recurrence + NL) — formula-like by construction (`x[n+1] = Ax[n] + Bu[n]`); promising, less proven for audio effects than TCN/RNN.
- **Volterra via synchronized swept sine** (Novak et al. 2010) — measurement-based, no training loop; diagonal kernels only, but instant.

## Landscape

- **NAM** (sdatkinson/neural-amp-modeler + NeuralAmpModelerCore) — open standard de facto; thousands of free captures (ToneHunt); Python training, C++ playback.
- **GuitarML** (Proteus, SmartAmp) — LSTM lane, open.
- **AIDA-X** (MOD Audio) — RTNeural playback, open, embedded-focused.
- **RTNeural** (Chowdhury) — the real-time inference C++ lib both above use; the natural WASM target for `neural-runtime` alongside ONNX Runtime Web.
- **Commercial hardware analogues** — Kemper Profiler, IK ToneX, Neural DSP Quad Cortex: proof of market and of capture-quality expectations; all closed, all weights-blob output.
- **Denoise/separate pretrained** — RNNoise, DeepFilterNet (denoise ceiling), Demucs / Open-Unmix (stems); sapphi-red/web-noise-suppressor already ships free ML AudioWorklets (threat noted in site research).
- **Nobody** ships browser-side capture, readable capture output, or capture→plugin compilation. Open niche.

## Needs / open questions

- **Training loop location**: offline Python (NAM-compatible, fastest to ship) vs own CLI (Node+WebGPU?) vs in-browser (WebGPU training is feasible for W–H-sized models — stage 3 has dozens of parameters, not millions). Staged: interop with NAM `.nam` files first, own trainer for the grey-box rung.
- **Runtime budget**: AudioWorklet, 128-sample quantum. W–H cascade is trivially real-time (it's biquads). NAM "standard" WaveNet is roughly one desktop CPU core native — needs WASM SIMD measurement; "feather/lite" profiles likely fine. This is the `neural-runtime` sizing question.
- **Probe/excitation atom**: standardized capture signal (sweeps + multi-level noise + program material) + level calibration; belongs near `@audio/measure`. NAM's input file is prior art.
- **Null-test tooling**: ESR + null-depth report as a first-class `@audio/measure` verb — it's also the marketing artifact ("here's the null").
- **Format**: adopt `.nam` (ecosystem of thousands of captures for free) + own JSON for grey-box chains (which is just an audiojs pipeline spec).
- **Weights licensing** (umbrella policy): many research models are non-commercial (Demucs weights, some DeepFilterNet variants) — audit before any promise; capture-trained weights are the user's own, no such problem.
- **Time-varying capture**: worth a rung? (chorus/phaser identification is published but fiddly) — defer until amp/EQ/comp rungs null well.
- **Defeedback relation**: classical `@audio/defeedback` stays classical (deterministic, zero-latency by design); a *captured* clone of a commercial suppressor is possible in principle but is the adaptive class — low priority, big model.
- **Monetization tie-in** (site research): capture→jz→VST is exactly the "tax at ecosystem boundary" pro tier — free browser capture + null report, pay to compile out.

## Refs

Boyd & Chua 1985 (fading memory, IEEE TCAS) · Farina 2000 (ESS, AES) · Novak et al. 2010 (synchronized swept sine, IEEE TIM) · Damskägg et al. 2019 (ICASSP, tube amp DL) · Wright et al. 2019 (DAFx, real-time RNN amps) · Wright & Välimäki 2021 (JAES, phaser/flanger) · Steinmetz & Reiss 2021 (micro-TCN compressors) · Kuznetsov, Parker, Esqueda 2020 (DAFx, differentiable IIR) · Parker et al. 2019 (DAFx, state-trajectory networks) · Engel et al. 2020 (DDSP, ICLR) · [neural-amp-modeler](https://github.com/sdatkinson/neural-amp-modeler) + [NeuralAmpModelerCore](https://github.com/sdatkinson/NeuralAmpModelerCore) · [RTNeural](https://github.com/jatinchowdhury18/RTNeural) · [Automated-GuitarAmpModelling](https://github.com/Alec-Wright/Automated-GuitarAmpModelling) · [AIDA-X](https://github.com/AidaDSP/AIDA-X) · [GuitarML](https://github.com/GuitarML) · [DeepFilterNet](https://github.com/Rikorose/DeepFilterNet) · [RNNoise](https://github.com/xiph/rnnoise) · [Demucs](https://github.com/facebookresearch/demucs) · [ToneHunt](https://tonehunt.org) (capture ecosystem)
