// Real-pair validation: probe signal through sox (third-party DSP black boxes),
// capture the pair, report rung + null depth per device. Dev script, needs sox:
//   node validate.js
// Honest expectations: EQ/reverb are LTI (deep null); overdrive is NL (rung 2–3);
// compand is adaptive (attack/release state) — outside rungs 1–3, shallow null.
//
// Results (2026-08-14, sox 14.4.2, fs 22050):
//   device     rung         null dB  attempts
//   eq         lti          -57.0    lti:-57.0 hammerstein:-57.0 wh/wh-tail:skip(linear)
//   lowpass    lti          -88.2    lti:-88.2
//   overdrive  wh-tail      -39.0    lti:-11.7 hammerstein:-30.9 wh:-17.1 wh-tail:-39.0
//   reverb     lti          -49.3    lti:-49.3 hammerstein:-49.3 wh/wh-tail:skip(linear)
//   compand    hammerstein  -22.7    lti:-20.3 hammerstein:-22.7 wh:-22.7 wh-tail:-22.7

import { execFileSync } from 'node:child_process'
import { writeFileSync, readFileSync, mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import capture from './capture.js'
import probe from './probe.js'

const fs = 22050
const { signal: dry } = probe({ fs, sweepDuration: 2, noiseDuration: 1, rampDuration: 2 })
// headroom: boosting devices (+6 dB EQ, reverb sum) must not clip in sox
for (let i = 0; i < dry.length; i++) dry[i] *= 0.4

const DEVICES = [
  ['eq', ['equalizer', '800', '1q', '+6', 'equalizer', '3000', '2q', '-8', 'bass', '+4']],
  ['lowpass', ['lowpass', '2500']],
  ['overdrive', ['overdrive', '15', '30']],
  ['reverb', ['reverb', '60', '50', '100']],
  ['compand', ['compand', '0.02,0.1', '-40,-30,-20,-15', '0']],
]

const dir = mkdtempSync(join(tmpdir(), 'ncap-'))
const dryRaw = join(dir, 'dry.raw')
writeFileSync(dryRaw, Buffer.from(dry.buffer, dry.byteOffset, dry.byteLength))
const soxFmt = ['-t', 'f32', '-r', String(fs), '-c', '1']

console.log('device'.padEnd(10), 'rung'.padEnd(12), 'null dB'.padEnd(8), 'latency', ' attempts')
for (const [name, effect] of DEVICES) {
  const wetRaw = join(dir, name + '.raw')
  execFileSync('sox', [...soxFmt, dryRaw, ...soxFmt, wetRaw, ...effect])
  const buf = readFileSync(wetRaw)
  const wet = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength >> 2)
  const t0 = Date.now()
  const r = capture(dry, wet)
  console.log(
    name.padEnd(10),
    r.rung.padEnd(12),
    r.nullDb.toFixed(1).padEnd(8),
    String(r.latency).padEnd(7),
    r.attempts.map(a => a.skipped ? `${a.rung}:skip` : `${a.rung}:${a.nullDb.toFixed(1)}`).join(' '),
    ((Date.now() - t0) / 1000).toFixed(1) + 's'
  )
}
rmSync(dir, { recursive: true, force: true })
