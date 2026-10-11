// What gets rendered: every row of a feature set as { voice, kind, note, velocity, hold, pre, cond,
// seed }, drawn from seeds derived from the voice id, so a set is the same wherever it is built.
//
//   view   training view: any note 33-99, velocity 20-127, held or staccato, a training chain
//   ref    catalog reference: C2, C3, C4, C5, C6 (MIDI 36-84) at velocities 64 and 112, held 1 s, clean
//   query  benchmark query: a note and velocity no reference uses (MIDI 38-94, velocity 30-127),
//          held or staccato, under one condition of augment.js CONDITIONS
import { createHash } from 'node:crypto'
import dx7 from '@audio/synth-dx7'
import { rng, chain, CONDITIONS, FS } from './augment.js'

export const REF_NOTES = [36, 48, 60, 72, 84], REF_VELOCITIES = [64, 112], REF_HOLD = 1
export const LENGTH = 1.6 // seconds rendered after the pre-roll: the 1.52 s a view spans, plus codec delay
export const seedOf = (...parts) => parseInt(createHash('sha1').update(parts.join(':')).digest('hex').slice(0, 8), 16)

const hold = r => r() < 0.35 ? r.uniform(0.06, 0.4) : r.uniform(0.4, 2.5)
const queryHold = r => r() < 0.5 ? r.uniform(0.08, 0.3) : r.uniform(0.6, 2)

// another voice of the same split under a mix: pool is a list of voice rows, self excluded
const other = (pool, self) => r => {
	let v = self
	while (v === self) v = r.pick(pool)
	return { voice: v, note: r.int(40, 84), velocity: r.int(60, 120), hold: r.uniform(0.3, 1.5), offset: r.uniform(-0.5, 0.5) }
}

export function views(voice, id, count) {
	return Array.from({ length: count }, (_, i) => {
		let seed = seedOf(id, 'view', i), r = rng(seed)
		return { voice, kind: 'view', note: r.int(33, 99), velocity: r.int(20, 127), hold: hold(r), pre: r.uniform(0, 0.03), cond: 'chain', seed }
	})
}

export const refs = voice => REF_NOTES.flatMap(note => REF_VELOCITIES.map(velocity => ({ voice, kind: 'ref', note, velocity, hold: REF_HOLD, pre: 0, cond: 'clean', seed: 0 })))

// One query draw per voice (note, velocity, hold, pre-roll), repeated under each condition so the
// conditions compare on the same notes.
export function queries(voice, id, conds) {
	let r = rng(seedOf(id, 'query')), note, velocity
	do note = r.int(38, 94); while (REF_NOTES.includes(note))
	do velocity = r.int(30, 127); while (REF_VELOCITIES.includes(velocity))
	let base = { voice, kind: 'query', note, velocity, hold: queryHold(r), pre: r.uniform(0, 0.05) }
	return conds.map(cond => ({ ...base, cond, seed: seedOf(id, 'query', cond) }))
}

// The augmentation a row carries, from its seed (a view's note was drawn from the same seed, so
// its chain takes a derived one).
export function specOf(row, pool) {
	let r = rng(row.kind === 'view' ? seedOf(row.seed, 'chain') : row.seed), o = other(pool, row.voice)
	if (row.cond === 'chain' && row.kind === 'view') return chain(r, o)
	return CONDITIONS[row.cond](r, o)
}

// The dry note: pre-roll of silence, then `LENGTH` seconds with the key up after `hold`.
export function note(patch, { note, velocity, hold, pre = 0 }) {
	let freq = 440 * 2 ** ((note - 69) / 12), held = Math.min(hold, LENGTH)
	let y = dx7(freq, { patch, velocity: velocity / 127, duration: held, release: LENGTH - held, fs: FS })
	let x = new Float32Array(Math.round(pre * FS) + y.length)
	x.set(y, x.length - y.length)
	return x
}

// the other voice's note under a mix, shifted by its offset against the target's onset
export function mixNote(patch, m, pre) {
	let y = note(patch, { ...m, pre: 0 }), shift = Math.round((pre + m.offset) * FS)
	let x = new Float32Array(Math.round(pre * FS) + Math.round(LENGTH * FS))
	for (let i = 0; i < x.length; i++) { let j = i - shift; if (j >= 0 && j < y.length) x[i] = y[j] }
	return x
}
