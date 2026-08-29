// Ladder vs a real amp: capture the 5150 NAM model (a WaveNet trained on real
// hardware) as a black box with @audio/neural-capture. Dev script:
//   node validate.js        (needs fixtures/5150.nam — see fixtures/README.md)
// Honest expectation: a high-gain amp is NL + memory beyond rungs 1–3's formula
// classes; shallow nulls here are the case for the TCN rung, not a failure.
//
// Result (2026-08-14, Helga B 5150 BlockLetter Boosted, probe at 0.5 peak):
//   lti −3.6 dB · hammerstein −4.8 dB · wh −5.1 dB · wh-tail −4.6 dB
// Boosted high-gain is the hardest class (deep clipping, fading-memory state);
// the formula rungs top out near −5 dB — the TCN rung is what closes this gap.
import { readFileSync } from 'node:fs'
import nam from './nam.js'
import capture from '../neural-capture/capture.js'
import probe from '../neural-capture/probe.js'

const amp = nam(readFileSync(new URL('./fixtures/5150.nam', import.meta.url), 'utf8'))
const fs = 48000
const { signal: dry } = probe({ fs, sweepDuration: 1.5, noiseDuration: 0.75, rampDuration: 1.5 })
for (let i = 0; i < dry.length; i++) dry[i] *= 0.5

console.time('nam render')
const wet = amp(Float32Array.from(dry))
console.timeEnd('nam render')

console.time('capture')
const r = capture(dry, wet)
console.timeEnd('capture')
console.log('rung', r.rung, 'null', r.nullDb.toFixed(1), 'dB', 'latency', r.latency)
for (const a of r.attempts)
  console.log(' ', a.rung.padEnd(12), a.skipped ? 'skip: ' + a.skipped : a.nullDb.toFixed(1) + ' dB')
