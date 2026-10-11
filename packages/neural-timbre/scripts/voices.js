#!/usr/bin/env node
// Every DX7 voice in a folder of sysex files, deduplicated by the sound it makes and split by
// cartridge, for training and benchmarking the timbre encoder.
//
//   node scripts/voices.js [dir]          dir: default ~/.cache/audiojs/data/dx7
//
// Reads every *.syx under dir (sources.json names the subfolders and their terms), writes
//   dir/voices.json   one row per distinct sound: name, cartridges it appears in, split, algorithm
//   dir/patches.u8    155 bytes a row, the VCED voice as first seen
//
// Sound key: the 145 voice parameters (names excluded) with what the MSFA engine ignores zeroed
// (synth-dx7 README, "Not modeled"): amplitude modulation (AMS, AMD), oscillator key sync, the LFO
// when its pitch depth or sensitivity is 0, pitch-EG rates when its four levels are equal. Then
// sounds whose raw engine output is identical at four notes and velocities merge too (a silent
// modulator's envelope, say), and sounds that stay below -78 dBFS there are marked silent.
//
// Split by cartridge, never by voice: a cartridge's role comes from a hash of its file, and a
// sound in any training cartridge is a training sound, in any development cartridge a
// development one, so held-out sounds appear only in held-out cartridges. (Grouping cartridges
// that share a sound chains 3,931 of 4,115 into one group through compilations, so the hash is
// per cartridge.) Yamaha's factory and VRC cartridges always train: they are the catalog a user
// queries, and the held-out read must come from cartridges the encoder never saw. Near-copies
// are flagged: `near` is the fewest parameters in which a sound differs from any training sound.
import { readFileSync, writeFileSync, readdirSync, statSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { createHash } from 'node:crypto'
import path from 'node:path'
import os from 'node:os'
import { parse, render } from '@audio/synth-dx7'

const dir = process.argv[2] ?? path.join(os.homedir(), '.cache/audiojs/data/dx7')
const ROLES = [['train', 75], ['dev', 5], ['heldout', 20]] // percent of cartridge groups
const ALWAYS_TRAIN = ['factory/', 'vrc/']

// Every DX7 voice message in a file: VMEM banks (format 9) and VCED voices (format 0) one by one,
// so a DX7II file's performance and supplement messages don't hide its voices. The byte count is
// not trusted: a tenth of circulating banks write 10 00 (hexadecimal 0x1000) where the manual's
// 7-bit count is 20 00; their 4096 data bytes are intact.
function voicesOf(bytes) {
	if (bytes[0] !== 0xf0) try { return parse(bytes) } catch { return [] }
	let out = []
	for (let i = 0; i + 6 < bytes.length; i++) {
		if (bytes[i] !== 0xf0 || bytes[i + 1] !== 0x43) continue
		let format = bytes[i + 3], size = format === 9 ? 4096 : format === 0 ? 155 : 0
		if (!size || i + 6 + size > bytes.length) continue
		try { out.push(...parse(bytes.subarray(i + 6, i + 6 + size))) } catch {}
		i += 6 + size - 1
	}
	return out
}

export const soundKey = patch => createHash('sha1').update(canonical(patch)).digest('hex').slice(0, 16)

function canonical(patch) {
	let p = Uint8Array.from(patch.subarray(0, 145))
	for (let op = 0; op < 6; op++) p[op * 21 + 14] = 0 // AMS
	p[140] = 0 // AMD
	p[136] = 0 // oscillator key sync
	if (p[139] === 0 || p[143] === 0) p.fill(0, 137, 144) // LFO inaudible: speed, delay, PMD, AMD, sync, wave, PMS
	if (p[130] === p[131] && p[131] === p[132] && p[132] === p[133]) p.fill(0, 126, 130) // flat pitch EG: rates unused
	return p
}

function* files(root) {
	for (let name of readdirSync(root).sort()) {
		let f = path.join(root, name)
		if (statSync(f).isDirectory()) yield* files(f)
		else if (/\.syx$/i.test(name)) yield f
	}
}

const hash32 = s => parseInt(createHash('sha1').update(s).digest('hex').slice(0, 8), 16)

// Raw int32 output at four keys and velocities, 0.3 s held and 0.2 s released: equal hashes mean
// equal samples. Peak in int16 steps (2^13 of MSFA's 2^28 full scale is 2^-15 of it, one LSB).
const PROBES = [[48, 40], [60, 100], [72, 127], [84, 70]]
function fingerprint(patch) {
	let h = createHash('sha1'), peak = 0
	for (let [note, velocity] of PROBES) {
		let y = render(patch, { note: note + patch[144] - 24, velocity, duration: 0.3, release: 0.2 })
		h.update(new Uint8Array(y.buffer, y.byteOffset, y.byteLength))
		for (let v of y) peak = Math.max(peak, Math.abs(v))
	}
	return { hash: h.digest('hex').slice(0, 16), peak: peak / 2 ** 13 }
}

// fewest differing parameters (sound keys' canonical bytes) from each of `rows` to any of `pool`
function nearest(rows, pool) {
	return rows.map(a => {
		let best = 145
		for (let b of pool) {
			let d = 0
			for (let i = 0; i < 145 && d < best; i++) if (a[i] !== b[i]) d++
			if (d < best) best = d
		}
		return best
	})
}

function main() {
	let sources = JSON.parse(readFileSync(path.join(dir, 'sources.json'), 'utf8')).sources
	let carts = [], byContent = new Map(), unreadable = []
	for (let src of sources) for (let f of files(path.join(dir, src.dir))) {
		let rel = path.relative(dir, f), voices = voicesOf(readFileSync(f))
		if (!voices.length) { unreadable.push(rel); continue }
		let content = createHash('sha1').update(Buffer.concat(voices.map(v => v.patch))).digest('hex')
		let c = byContent.get(content)
		if (c) { c.copies.push(rel); continue }
		c = { file: rel, source: src.id, copies: [], voices }
		byContent.set(content, c); carts.push(c)
	}

	// distinct sounds, and the cartridges holding each
	let sounds = new Map()
	carts.forEach((c, ci) => c.voices.forEach((v, vi) => {
		let key = soundKey(v.patch), s = sounds.get(key)
		if (!s) sounds.set(key, s = { key, name: v.name, patch: v.patch, at: [] })
		if (!s.at.some(a => a[0] === ci)) s.at.push([ci, vi])
	}))

	// identical renders are one sound
	let byRender = new Map(), silent = new Set()
	for (let [key, s] of sounds) {
		let { hash, peak } = fingerprint(s.patch)
		if (peak < 4) silent.add(key)
		let first = byRender.get(hash)
		if (!first) { byRender.set(hash, s); continue }
		for (let a of s.at) if (!first.at.some(b => b[0] === a[0])) first.at.push(a)
		sounds.delete(key)
	}

	let role = carts.map(c => {
		if (ALWAYS_TRAIN.some(p => c.file.startsWith(p))) return 'train'
		let h = hash32(c.file) % 100
		for (let acc = 0, k = 0; k < ROLES.length; k++) if (h < (acc += ROLES[k][1])) return ROLES[k][0]
	})

	let rows = [], bytes = []
	for (let s of sounds.values()) {
		let roles = new Set(s.at.map(([ci]) => role[ci]))
		let split = roles.has('train') ? 'train' : roles.has('dev') ? 'dev' : 'heldout'
		let at = s.at.find(([cj]) => role[cj] === split), [ci, vi] = at, c = carts[ci], v = c.voices[vi]
		rows.push({ id: s.key, name: v.name, split, algorithm: v.patch[134] + 1, source: c.source, cartridge: c.file, index: vi,
			also: s.at.filter(a => a !== at).map(([cj, vj]) => `${carts[cj].file}#${vj}`) })
		bytes.push(v.patch)
	}
	let train = rows.map((r, i) => r.split === 'train' ? canonical(bytes[i]) : null).filter(Boolean)
	let other = rows.map((r, i) => [r, i]).filter(([r]) => r.split !== 'train')
	nearest(other.map(([, i]) => canonical(bytes[i])), train).forEach((d, k) => { other[k][0].near = d })
	rows.forEach(r => { if (silent.has(r.id)) r.silent = true })
	writeFileSync(path.join(dir, 'patches.u8'), Buffer.concat(bytes))
	let count = r => rows.filter(x => x.split === r).length
	let summary = {
		files: carts.length + carts.reduce((n, c) => n + c.copies.length, 0) + unreadable.length, unreadable: unreadable.length,
		cartridges: carts.length, voices: carts.reduce((n, c) => n + c.voices.length, 0), sounds: rows.length,
		heldoutCartridges: role.filter(r => r === 'heldout').length,
		train: count('train'), dev: count('dev'), heldout: count('heldout'), silent: silent.size,
		heldoutNearCopies: rows.filter(r => r.split === 'heldout' && r.near <= 3).length,
	}
	writeFileSync(path.join(dir, 'voices.json'), JSON.stringify({ version: 1, created: new Date().toISOString().slice(0, 10), summary, voices: rows }))
	console.log(JSON.stringify(summary))
}

// importable for soundKey: the key a catalog entry's `id` is, to find the voice in your own sysex
if (process.argv[1] === fileURLToPath(import.meta.url)) main()
