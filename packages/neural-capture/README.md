# @audio/neural-capture

> Device capture: dry/wet pair in, cheapest audiojs formula out, with a null-depth report.

Feed it the same audio without and with a device (plugin, amp, pedal, tape machine); it identifies the device by climbing a ladder of model classes, cheapest first, escalating only when the null test (`wet − model(dry)`) fails:

| Rung | Device class | Formula |
|---|---|---|
| `lti` | EQ, cab, reverb, room | impulse response |
| `hammerstein` | waveshaper, clipper + tone stack | static curve → IR |
| `wh` | drive + tone shaping (amp-class, short memory) | biquads → static curve → biquads |
| `wh-tail` | NL + long tail (drive into cab/room) | wh → linear tail IR |
| *planned* | anything with fading memory (high-gain amps) | causal TCN (NAM-class) |

The result is an inspectable chain of shipped atoms, not a weights blob: biquad cascades for [`@audio/biquad`](https://github.com/audiojs/filter), a Chebyshev curve for [`@audio/saturate-waveshaper`](https://github.com/audiojs/saturate), an IR for [`@audio/reverb-convolution`](https://github.com/audiojs/reverb), in the `@audio/chain` recipe vocabulary (`{ atom, name, params, why }`, JSON-safe).

```js
import capture, { render, nullDepth } from '@audio/neural-capture'
import probe from '@audio/neural-capture/probe'

let r = capture(dry, wet)
r.rung        // 'wh' – cheapest class that nulled
r.nullDb      // -59.0 – null depth in dB (10·log10 ESR)
r.latency     // 19 – device lag in samples, trimmed from the pair (IR-onset detection)
r.ir          // impulse response (lti/hammerstein rungs)
r.curve       // { type: 'chebyshev', coeffs, scale } from the hammerstein rung up
r.pre, r.post // wh rung: biquad cascades ({ b0, b1, b2, a1, a2 } sections)
r.stages      // chain recipe: [{ atom: '@audio/biquad', name: 'cascade', params, why }, …]
r.attempts    // [{ rung, nullDb, esr }, …] – every rung tried (or skipped, with reason)

render(r, dry)                    // → predicted wet (verify the null yourself)
capture(dry, wet, { maxRung: 'lti', targetDb: -40, reg: 1e-6 })

// standardized excitation: click + ESS sweep + 4-level noise + amplitude ramp, ~16 s
let { signal, layout } = probe({ fs: 48000 })   // deterministic: regenerate anytime
```

| Option | Default | |
|---|---|---|
| `targetDb` | `-60` | stop escalating once the null reaches this depth |
| `maxRung` | `'wh-tail'` | cap the ladder |
| `align` | `true` | lag detection (Wiener-IR onset, excitation-whitened) + trim; the lag lands in `result.latency` |
| `irLength` | auto | fixed IR length; auto = envelope-vs-noise-floor detection + extend-while-it-pays |
| `reg` | `1e-8` | Tikhonov regularization vs input energy; raise for noisy captures |
| `order` | `11` | Chebyshev order of the static curve |
| `whSections` | `2` | biquad sections per wh filter (pre and post each) |
| `whIters` | `1500` | wh Adam iteration cap (`whLr` 0.1, cosine-decayed) |
| `whMaxMemory` | `2048` | skip wh when the identified IR's 99%-energy depth exceeds this |
| `whSkipLinearDb` | `-40` | skip wh when a cheaper rung already nulls this deep (device essentially linear) |

**Identification.** The linear stage is exact least squares, not spectral division: a Wiener deconvolution seeds memory-depth detection, then conjugate gradient with FFT Toeplitz matvecs solves the FIR (unbiased on truncated records, machine-deep on clean pairs). The Hammerstein rung alternates that IR solve with a linear-in-coefficients Chebyshev curve fit. The wh rung re-solves the curve by exact LS every iteration (given the filters the model is linear in curve coefficients, which sidesteps the classic W–H local minimum where the curve absorbs everything), while Adam moves only the filter parameters using analytic adjoint gradients, poles kept stable by construction (radius/angle parameterization). Gradients are finite-difference-tested in the suite.

**Measured.** Synthetic devices (`test.js`): multi-tap system −148 dB, 3000-tap decaying reverb tail −118 dB, tanh→FIR cascade −70 dB, biquad→tanh→biquad −59 dB, biquad→tanh→long-FIR −54 dB (`wh-tail`, where hammerstein stalls at −13). Third-party black boxes (`validate.js`, sox at 22.05 kHz): EQ −57, lowpass −88, reverb −49 (all rung 1), overdrive −39 (`wh-tail`; ESR 0.013%, well under the ~1% NAM-quality bar), compander −23 (adaptive class: shallow, as theory says it must be). Real boosted 5150 (via `@audio/neural-amp` playback): formula rungs top out near −5 dB, the honest boundary where the planned TCN rung takes over.

**Semantics**: the curve is fitted on `|x| ≤ scale` (polynomial extrapolation beyond); `render` is identification-domain, no oversampling, matching how the formula was fitted and how the null was measured. Excitation coverage is on you: the device is learned only on the signal manifold the dry take excites; the `probe` signal (or full-level, full-band material) beats a quiet noodle.

**Use when:** capturing a device you can feed a dry signal through; the output is a readable, dependency-free, zero-latency pipeline.<br>
**Not for:** time-varying devices (chorus/phaser: LFO phase breaks time invariance), lookahead limiters (matchable only at the device's own latency), adaptive processors (denoise/suppressors/companders: inference, not circuit; see `@audio/neural-denoise`).

---

Part of the [@audio/neural](https://github.com/audiojs/neural) lane.

MIT © [audiojs](https://github.com/audiojs)
