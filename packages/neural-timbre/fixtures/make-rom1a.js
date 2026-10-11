#!/usr/bin/env node
// fixtures/rom1a.json and rom1a.f16: the 32 voices of Yamaha's ROM1A played at a note and velocity
// no catalog reference uses (F4, velocity 90, held 0.8 s, 0.8 s release), rendered by
// @audio/synth-dx7 and prepared by features.js: float16, band-major, one voice after another. The
// test ranks them against the bundled dx7-factory catalog with no synth or cartridge at hand, and
// checks them against a live render when @audio/synth-dx7 resolves.
//
//   node fixtures/make-rom1a.js      needs @audio/synth-dx7 and $TIMBRE_DATA/factory/rom1a.syx
//                                    (default ~/.cache/audiojs/data/dx7; README, Training)
import { readFileSync, writeFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import os from 'node:os'
import path from 'node:path'
import dx7, { parse } from '@audio/synth-dx7'
import { features, featuresSha256, FEATURES } from '../timbre.js'
import { soundKey } from '../scripts/voices.js'

export const NOTE = { midi: 65, velocity: 90, duration: 0.8, release: 0.8 }

// one voice's note, as the fixture holds it
export const render = patch => dx7(440 * 2 ** ((NOTE.midi - 69) / 12), { patch, velocity: NOTE.velocity / 127, duration: NOTE.duration, release: NOTE.release })

if (process.argv[1] === fileURLToPath(import.meta.url)) {
	let data = process.env.TIMBRE_DATA ?? path.join(os.homedir(), '.cache/audiojs/data/dx7')
	let voices = parse(readFileSync(path.join(data, 'factory/rom1a.syx'))), size = FEATURES.bands * FEATURES.frames
	let f16 = new Float16Array(voices.length * size)
	voices.forEach((v, i) => f16.set(features(render(v.patch)), i * size))
	writeFileSync(new URL('./rom1a.f16', import.meta.url), new Uint8Array(f16.buffer))
	writeFileSync(new URL('./rom1a.json', import.meta.url), JSON.stringify({
		cartridge: 'ROM1A', note: NOTE, featuresSha256: await featuresSha256(),
		voices: voices.map((v, i) => ({ index: i + 1, name: v.name, id: soundKey(v.patch) })),
	}, null, '\t') + '\n')
	console.log(`${voices.length} voices, ${f16.byteLength} bytes`)
}
