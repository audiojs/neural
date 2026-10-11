#!/usr/bin/env node
// Freeze the benchmark before any training result: pin the files that define it (the voice split,
// the render and augmentation recipe, the frontend) by SHA-256 in <data>/bench.json. scripts/bench.js
// refuses to report against a changed pin. Refuses to overwrite an existing manifest; a pin moves
// only with --repin FILE --reason TEXT, recorded in the manifest's `repins`.
//
//   node scripts/freeze.js [--data dir]
//   node scripts/freeze.js --repin features.js --reason '...'
import { readFileSync, writeFileSync, existsSync } from 'node:fs'
import { createHash } from 'node:crypto'
import path from 'node:path'
import os from 'node:os'
import { fileURLToPath } from 'node:url'
import { CONDITIONS } from './augment.js'
import { REF_NOTES, REF_VELOCITIES, REF_HOLD } from './recipe.js'

const arg = k => { let i = process.argv.indexOf(k); return i > 0 ? process.argv[i + 1] : undefined }
const data = arg('--data') ?? path.join(os.homedir(), '.cache/audiojs/data/dx7')
const here = path.dirname(fileURLToPath(import.meta.url))
const sha = f => createHash('sha256').update(readFileSync(f)).digest('hex')
export const PINS = { 'voices.json': path.join(data, 'voices.json'), 'patches.u8': path.join(data, 'patches.u8'),
	'scripts/recipe.js': path.join(here, 'recipe.js'), 'scripts/augment.js': path.join(here, 'augment.js'), 'features.js': path.join(here, '../features.js') }
export const pins = () => Object.fromEntries(Object.entries(PINS).map(([k, f]) => [k, sha(f)]))

if (process.argv[1] === fileURLToPath(import.meta.url) && arg('--repin')) {
	let out = path.join(data, 'bench.json'), bench = JSON.parse(readFileSync(out, 'utf8')), file = arg('--repin'), reason = arg('--reason')
	if (!PINS[file] || !reason) throw new Error('--repin one of ' + Object.keys(PINS) + ' with --reason')
	let now = sha(PINS[file])
	;(bench.repins ??= []).push({ date: new Date().toISOString(), file, from: bench.pins[file], to: now, reason })
	bench.pins[file] = now
	writeFileSync(out, JSON.stringify(bench, null, 1))
	console.log('repinned', file)
} else if (process.argv[1] === fileURLToPath(import.meta.url)) {
	let out = path.join(data, 'bench.json')
	if (existsSync(out)) throw new Error(`${out} exists: the benchmark is frozen`)
	let voices = JSON.parse(readFileSync(PINS['voices.json'], 'utf8')).voices, held = voices.filter(v => v.split === 'heldout' && !v.silent)
	writeFileSync(out, JSON.stringify({
		version: 1, frozen: new Date().toISOString(),
		what: 'Held-out DX7 cartridges: every audible held-out voice is one catalog entry (references: C2-C6 at velocities 64 and 112, held 1 s) and one query note at a note and velocity no reference uses, under each condition. Near copies: held-out voices within 3 parameters of a training voice, reported apart.',
		voices: held.length, novel: held.filter(v => v.near >= 4).length, nearCopies: held.filter(v => v.near <= 3).length,
		references: { notes: REF_NOTES, velocities: REF_VELOCITIES, hold: REF_HOLD }, conditions: Object.keys(CONDITIONS),
		metrics: ['top-1 and top-5 against the held-out catalog', 'top-1 and top-5 against the catalog of every voice', 'rebuild: specLoss of the retrieved preset before and after seeded match, against blind match at the same budget'],
		pins: pins(),
	}, null, 1))
	console.log('froze', out)
}
