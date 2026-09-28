// Label check: steady notes (no vibrato, scoop, drift or glide), clean, per source; each labelled
// frame against @audio/pitch-pyin's single-frame estimate over the 2048 samples whose YIN window is
// centred on it. YIN's own octave errors show up as the tail; the median is the labels' offset.
// node scripts/labels.js
import pyin from '@audio/pitch-pyin'
import { clip, SOURCE_NAMES } from './data.js'

console.log('source frames within-10c within-50c median-c p90-c')
for (let src of SOURCE_NAMES) {
	let d = []
	for (let s = 0; s < 6; s++) {
		let c = await clip(1000 + s, { source: src, clean: true, steady: true, fs: 44100 })
		for (let j = 0; j < c.frames.length; j++) {
			let a = c.frames[j] - 512
			if (!(c.f0[j] > 0) || !c.weight[j] || a < 0 || a + 2048 > c.x.length) continue
			let e = pyin(c.x.subarray(a, a + 2048), { fs: c.fs, minFreq: 30, maxFreq: 3000 })
			if (e) d.push(Math.abs(1200 * Math.log2(e.freq / c.f0[j])))
		}
	}
	d.sort((a, b) => a - b)
	let q = p => d[Math.floor(p * (d.length - 1))], share = t => (d.filter(v => v < t).length / d.length).toFixed(3)
	console.log(src, d.length, share(10), share(50), q(0.5).toFixed(2), q(0.9).toFixed(1))
}
