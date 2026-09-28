// WAV (RIFF) reader for the scripts: PCM 8/16/24/32-bit and IEEE float 32/64, any channel count.
// readWav(file, { channel }) → { data: Float32Array (channel, or the mean of all), fs, channels }

import { readFileSync } from 'node:fs'

export function readWav(file, { channel } = {}) {
	let b = readFileSync(file), v = new DataView(b.buffer, b.byteOffset, b.byteLength)
	if (b.toString('ascii', 0, 4) !== 'RIFF' || b.toString('ascii', 8, 12) !== 'WAVE') throw new Error(`${file}: not a WAV file`)
	let fmt, dataAt, dataLen
	for (let o = 12; o + 8 <= b.length;) {
		let id = b.toString('ascii', o, o + 4), len = v.getUint32(o + 4, true)
		// WAVE_FORMAT_EXTENSIBLE (0xFFFE) carries the real format tag at the start of its sub-format GUID
		if (id === 'fmt ') fmt = { tag: v.getUint16(o + 8, true), ch: v.getUint16(o + 10, true), fs: v.getUint32(o + 12, true), bits: v.getUint16(o + 22, true) }
		if (id === 'fmt ' && fmt.tag === 0xfffe) fmt.tag = v.getUint16(o + 32, true)
		if (id === 'data') { dataAt = o + 8; dataLen = Math.min(len, b.length - o - 8) }
		o += 8 + len + (len & 1)
	}
	if (!fmt || dataAt == null) throw new Error(`${file}: no fmt or data chunk`)
	let { ch, fs, bits, tag } = fmt
	let bytes = bits >> 3, n = Math.floor(dataLen / (bytes * ch)), data = new Float32Array(n)
	let read = tag === 3 ? (bits === 64 ? o => v.getFloat64(o, true) : o => v.getFloat32(o, true))
		: bits === 8 ? o => (v.getUint8(o) - 128) / 128
		: bits === 16 ? o => v.getInt16(o, true) / 32768
		: bits === 24 ? o => ((v.getUint8(o) | v.getUint8(o + 1) << 8 | v.getInt8(o + 2) << 16)) / 8388608
		: o => v.getInt32(o, true) / 2147483648
	for (let i = 0; i < n; i++) {
		let o = dataAt + i * bytes * ch
		if (channel != null) data[i] = read(o + channel * bytes)
		else { let s = 0; for (let c = 0; c < ch; c++) s += read(o + c * bytes); data[i] = s / ch }
	}
	return { data, fs, channels: ch }
}
