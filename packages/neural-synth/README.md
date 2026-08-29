# @audio/neural-synth

> Sound matching: configure any knob-synth to reproduce a target sound.

The class of algorithm is automatic synthesizer programming (SpiegeLib / Sound2Synth lineage): the synth is a black box (knob vector in, audio out), the objective is a multi-scale spectral distance, and the search is CMA-ES: no gradients required, so any synth qualifies, including plugins and hardware renders.

```js
import match, { specLoss, cmaes } from '@audio/neural-synth'

let r = match(params => mySynth.render(params), targetAudio, {
  bounds: [[80, 800], [300, 6000], [1, 8]],   // one [min, max] per knob
  budget: 2400,                                // render calls you're willing to pay
  loss: (a, b) => specLoss(a, b, { fs: 22050 }),
})
r.params   // knob settings that reproduce the target
r.loss     // residual spectral distance (0 = exact)
```

| Option | Default | |
|---|---|---|
| `bounds` | required | `[min, max]` per knob, real units |
| `budget` | `2000` | total render evaluations |
| `restarts` | `3` | independent CMA-ES starts, scattered over the knob box |
| `seeds` | none | warm starts (e.g. from a learned inverse model); searched with tight sigma |
| `loss` | `specLoss` | any `(target, candidate) → number` |

**Search**: knobs are normalized to the unit box (quadratic penalty outside); restarts share 70% of the budget from scattered points, the winner gets a 30% polish run at wider sigma; that hop matters because the spectral loss keeps shallow local basins (cutoff/drive trade-offs, harmonic aliases). `cmaes` is the full Hansen formulation (weighted recombination, rank-1 + rank-μ covariance, CSA step-size), deterministic under `seed`.

**Loss**: per-frame linear spectral convergence (fine detail) + log-mel band distance on the averaged spectrum per FFT scale. The mel term is what makes pitch and cutoff optimizable at all; raw bin-wise losses are combs with basins too narrow to find.

**Measured** (`test.js`): subtractive patch (pitch/cutoff/drive) recovered to 220.0 Hz / 1200.5 Hz / 3.0 from a 2400-render budget; FM patch (carrier/ratio/index) to 330.0 Hz / 2.00 / 1.50 from 3000.

**Use when:** replicating a sound with a synth you can render offline: patch matching, preset mining, vector-sound tooling.<br>
**Not for:** streaming/real-time adaptation, or synths whose render is not deterministic in their params (per-note random mod needs averaging first).

---

Part of the [@audio/neural](https://github.com/audiojs/neural) lane.

MIT © [audiojs](https://github.com/audiojs)
