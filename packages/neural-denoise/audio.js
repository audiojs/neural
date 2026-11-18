// atom manifest: RNNoise per @audio/compile CONTRACT, streaming at any sample rate.
//
// RNNoise (rnnoise.js, bit-exact to upstream's C) takes 10 ms frames, 480 samples at 48 kHz, and returns each
// 960 samples late. Blocks of any size collect into frames; the output queue starts with FRAME − 1 = 479
// samples of silence, the least that never runs dry for any block size (the worklet's 448 serves 128-sample
// quanta only), so the delay is constant: 960 + 479 = 1439 samples at 48 kHz, 30 ms. Other rates go to
// 48 kHz and back through a streaming form of @audio/resample-sinc's Lanczos kernel, the resampler denoise()
// uses offline; its lookahead adds to the delay (1354 samples at 44.1 kHz, 30.7 ms). The output is
// denoise()'s: at 48 kHz sample for sample; at other rates, with resample-sinc 1.3.0, except within 50 ms of
// the end, where the offline resampler sees the end and a stream sees the host's trailing silence.
//
// limit (dB): the most RNNoise may take off. The input is mixed back in at a = 10^(−limit/20) at every value (share()),
// so noise drops by at most `limit`: at 0, a = 1 and the output is the input, the latency late (at 48 kHz its samples
// themselves; at other rates as the two converters carry it to 48 kHz and back); at the top, 100, a = 10^−5, RNNoise's
// output with the input 100 dB under it mixed in; denoise() and the worklet take Infinity, a = 0, RNNoise's output
// alone (README, The limit's scale). The default is denoise()'s, 16 dB (rnnoise.js LIMIT): unlimited, RNNoise's current model gates
// clean speech and removes the voice on some noisy files; 16 is the most before the voice itself suffers (README,
// Accuracy). The share falls as the limit rises, over the whole range, so a host's ramp or automation curve from one
// value to another passes only through shares between theirs. limit declares no `smoothing`: a host hands a jump as it
// is, and the share glides (slide(), below): from the share in force at the call that brings the value, in a straight
// line over 20 ms to the new one, carried sample by sample across calls, the same samples however the host cuts them,
// where a host's ramp would hand a value a call and so shift the output with its cuts. Each input sample takes
// the share in force as it arrives: the mix is made at 48 kHz, frame by frame, of the input and of RNNoise's output to
// it, so a change is heard the latency later, on the input it came with. At 48 kHz each output sample is the two
// renders the values either side of a jump give held, mixed by the line, up to float32's rounding (test.js bounds it,
// with music 'enhance' and in the guard's ramp); at other rates the converter back spreads the line's corners over its
// taps, which can carry a sample a little past both (test.js: the peak at most 0.01 dB over the louder).
// An instance sounds at the limit it is built with from its first sample; held, the share is the one number it was,
// and the output as before the glide.
//
// music: 'pass' (default) has music pass through untouched, as denoise() does (guard.js): each 48 kHz input frame
// goes to the guard too, and the output frame two before it, the one RNNoise gives back then, takes the guard's
// decision so far; the gain between the denoised and the dry signal ramps over 200 ms, at 48 kHz, through the same
// resampler back, and mixes at the host's rate with the input delayed by the latency, so what passes is the input
// itself. Nothing is added to the delay: a stream decides on what has arrived, music is denoised for its first second
// or so, and each switch comes about a second late (README, Music). 'enhance' denoises everything, as before 0.4.
//
// tail: what rings on after the input stops is RNNoise's DC-blocking biquad (rnnoise.js run(), upstream's
// rnn_biquad, a = [−1.99599, 0.996]: poles at r·e^(±iθ), r = √0.996, θ for 18.65 Hz at 48 kHz). With no input its
// output is r^n·(P cos nθ + Q sin nθ), P and Q set by its state; after a DC step or a held tone of any frequency,
// stopping at any phase, the envelope √(P² + Q²) is at most 1.4183 of the input's peak (a 16.3 Hz tone; DC 1.2952),
// 2.0058 of a tone's RMS. RNNoise passes that ring ungated where it takes it for voice or for silence (a quiet DC
// step, a low tone) and less where it gates it (its gains are at most 1, its pitch filter keeps each band's energy),
// so the denoised part, (1 − a) of the output with a the limit's share, is 60 dB under the input's level
// ln(1000·2.0058·(1 − a)) / −ln r samples at 48 kHz on: 79.0 ms at the top, 100, 77.3 ms at the default 16, and none
// at 0, where no part of the output is RNNoise's (1 − a = 0). In seconds it
// is the same at every host rate: the biquad runs at 48 kHz, and the rest rings for less, the 20 ms window's
// overlap-add 7 ms past an edge, the resamplers' kernels on the edge of the dry share (which stops with the input)
// T₁ + T₂ frames, 32 at 8 to 44.1 kHz and 64 at 96, under the contract's 128; the guard's gain only scales signals
// that stop.
//
// Channels are denoised independently, each of ctx.maxChannels with its state made in the processor (RNNoise's 70 KB,
// the guard's 0.3 MB, queues sized by ctx.maxBlockSize). A call longer than ctx.maxBlockSize or wider than
// ctx.maxChannels, which the contract forbids, throws a RangeError naming what it carried and the bound (two compares
// a call) rather than overrun the queues. process allocates nothing (CONTRACT.md): what changes lives in typed arrays
// and the queues are rings of a fixed size. A float passed to or returned from a call V8 does not inline is a new
// object; the one function of the package called per sample, blend() in blends(), is one V8 inlines. So is each float
// in a function V8 has not optimized: the processor's own closure holds none (slide() takes the limit's), and the first
// processor made has V8 optimize the code before any stream (warm(), below), all but what V8 still compiles in a
// process's first stream, which allocates there (README, Speed and memory).

import { phases } from '@audio/resample-sinc'
import { model, create, FRAME, LIMIT, weights } from './rnnoise.js'
import { guardNet, online, ramps, blends } from './guard.js'

const RATE = 48000, DELAY = 2 * FRAME, PRIME = FRAME - 1, HALF = 16
const net = model(await weights()), guard = await guardNet()

const pow2 = n => 2 ** Math.ceil(Math.log2(n))

/** Lanczos taps each side from rate `from` to `to`, as resample-sinc: 16, widened by the ratio on downsampling. */
function taps(from, to) {
	let rate = from / to, scale = rate > 1 ? 1 / rate : 1
	return Math.ceil(HALF / scale)
}

/** Delay at host rate sr, in samples. Output i of the converter back to sr reads 48 kHz samples up to
 *  floor(i·48000/sr) + T₂; each of those waits for the end of its frame, at most 1439 samples on; that 48 kHz
 *  sample reads host input up to floor(·sr/48000) + T₁. So output i needs input up to i + D, no further. */
const latency = sr => sr === RATE ? DELAY + PRIME : Math.floor((taps(RATE, sr) + DELAY + PRIME) * sr / RATE) + taps(sr, RATE)

// The biquad's decay per sample and the envelope of its slowest free response against an input's RMS (the tail
// paragraph above; 1.4183 of a tone's peak, at 16.3 Hz, is the largest over 0.05 to 200 Hz in steps of 0.01 Hz, from
// the biquad's state for each tone, every stop phase at once: README, RNNoise; it falls off as 1/f above and to DC's
// 1.2952 below)
const DECAY = -Math.log(Math.sqrt(Math.fround(0.996))), ENVELOPE = 1.4183 * Math.SQRT2

/** Seconds after the input stops until the output stays 60 dB under its level, at `limit` dB */
const tail = limit => Math.max(0, Math.log(1e3 * ENVELOPE * (1 - share(limit))) / DECAY / RATE)

/** The share of the input a limit of v dB mixes back, 10^(−|v|/20) at every value: 1 at 0, 10^−5 at 100, 0 at Infinity.
 *  No arm, and the sign put on the divisor, so that no path and no kind of number (a negated whole number) is first
 *  taken at a value a stream brings late */
const share = v => 10 ** (Math.abs(v) / -20)
/** A setting's limit: a host's Float32Array of one, a number, or none (the default) */
const limitOf = params => { let l = params?.limit; return l?.[0] ?? l ?? LIMIT }

/** A FIFO in a ring of `size` samples (a power of two), primed with `len` zeros: at = [index of the first, how many]. */
const queue = (size, len) => ({ buf: new Float32Array(size), mask: size - 1, at: Int32Array.of(0, len) })

/** put(q, x, n): x[0…n) onto the end of q */
function put(q, x, n) {
	let { buf, mask, at } = q, w = at[0] + at[1]
	for (let i = 0; i < n; i++) buf[(w + i) & mask] = x[i]
	at[1] += n
}

/** pull(q, y, n): the first n of q into y[0…n), zeros past what it holds */
function pull(q, y, n) {
	let { buf, mask, at } = q, r = at[0], m = Math.min(n, at[1])
	for (let i = 0; i < m; i++) y[i] = buf[(r + i) & mask]
	y.fill(0, m, n)
	at[0] = (r + m) & mask; at[1] -= m
}

/**
 * @audio/resample-sinc's resample() (Lanczos, a = 16) as a stream, in its arithmetic, so its samples: (x, n, q) takes
 * x[0…n) and puts into the queue q every output whose taps have all arrived. Output i sits at input position i·from/to;
 * at integer rates it reads resample()'s phase table, at others it weighs by angle addition as resample() does there;
 * taps before the start are left out and the weights renormalized, as resample() does at its edges. Made `tagged`, it
 * takes a tag per input sample too, (x, n, q, t, tq), and puts each output's base sample's tag into tq, a queue of
 * q's size whose places follow q's (the limit's share: an output sample takes the share of the input it stands on).
 */
function resampler(from, to, tagged) {
	let rate = from / to, scale = rate > 1 ? 1 / rate : 1, T = Math.ceil(HALF / scale), K = HALF / (Math.PI * Math.PI)
	// input history: an output reads 2T samples, the newest of them just arrived
	let mask = (1 << Math.ceil(Math.log2(2 * T + 1))) - 1, ring = new Float32Array(mask + 1), tags = tagged ? new Float64Array(mask + 1) : null
	let tab = phases(from, to)
	if (tab) {
		// resample()'s position, base and phase p, kept relative to the ring, so each count stays a small integer however
		// long the stream: the ring index of the next sample, the samples in past base, base's ring index, base itself up
		// to T (below T the input's start cuts taps), p
		let { L, M, W, J0, J1 } = tab, S = 2 * T, st = new Int32Array(5)
		return (x, n, q, t, tq) => {
			let head = st[0], ahead = st[1], b = st[2], bs = st[3], p = st[4]
			let { buf, mask: qm, at } = q, w = at[0] + at[1], m = 0
			for (let s = 0; s < n; s++) {
				ring[head] = x[s]; if (tags) tags[head] = t[s]
				head = (head + 1) & mask; ahead++
				while (ahead > T) {
					let lo = b + 1 - T, o = p * S, sum = 0, wt = 0
					for (let j = Math.max(T - 1 - bs, J0[p]), e = J1[p]; j < e; j++) { let k = W[o + j]; sum += ring[(lo + j) & mask] * k; wt += k }
					if (tags) tq.buf[(w + m) & qm] = tags[b]
					buf[(w + m++) & qm] = wt !== 0 ? sum / wt : 0
					if ((p += M) >= L) { let c = Math.floor(p / L); b = (b + c) & mask; ahead -= c; bs = Math.min(T, bs + c); p -= c * L }
				}
			}
			st[0] = head; st[1] = ahead; st[2] = b; st[3] = bs; st[4] = p
			at[1] += m
		}
	}
	let A = new Float64Array(2 * T), B = new Float64Array(2 * T), C = new Float64Array(2 * T), D = new Float64Array(2 * T)
	for (let j = 0, t = 1 - T; t <= T; t++, j++) {
		let a = Math.PI * scale * t
		A[j] = Math.sin(a); B[j] = Math.cos(a); C[j] = Math.sin(a / HALF); D[j] = Math.cos(a / HALF)
	}
	// samples in and outputs out, absolute as resample()'s position needs them: floats in a typed array
	let st = new Float64Array(2)
	return (x, n, q, tg, tq) => {
		let N = st[0], i = st[1], { buf, mask: qm, at } = q, wr = at[0] + at[1], m = 0
		for (let s = 0; s < n; s++) {
			if (tags) tags[N & mask] = tg[s]
			ring[N++ & mask] = x[s]
			for (;;) {
				let pos = i * rate, base = Math.floor(pos)
				if (base + T >= N) break
				let frac = pos - base
				let f = Math.PI * scale * frac, sf = Math.sin(f), cf = Math.cos(f), sg = Math.sin(f / HALF), cg = Math.cos(f / HALF)
				let sum = 0, w = 0
				for (let j = 0, t = 1 - T; t <= T; t++, j++) {
					if (base + t < 0) continue
					let y = (t - frac) * scale
					if (Math.abs(y) >= HALF) continue
					// sinc(y)·sinc(y/HALF) written out: so rare a call is never inlined, and its float argument and result
					// would be objects
					let k, u = y / HALF
					if (Math.abs(y) < 1e-3) k = (y === 0 ? 1 : Math.sin(Math.PI * y) / (Math.PI * y)) * (u === 0 ? 1 : Math.sin(Math.PI * u) / (Math.PI * u))
					else k = (A[j] * cf - B[j] * sf) * (C[j] * cg - D[j] * sg) * K / (y * y)
					sum += ring[(base + t) & mask] * k; w += k
				}
				if (tags) tq.buf[(wr + m) & qm] = tags[base & mask]
				buf[(wr + m++) & qm] = w !== 0 ? sum / w : 0
				i++
			}
		}
		st[0] = N; st[1] = i; at[1] += m
	}
}

/** At 48 kHz the converters copy: the input into the 48 kHz queue with its shares (tq's places follow q's), a frame
 *  into a queue. They take the resamplers' calls, so run() and done() take one path at every rate: a branch on the rate
 *  had no type feedback in an instance at another rate than the first's, and Node 20 deoptimized done() there on every
 *  frame of a 44.1 kHz stream after a 48 kHz one, for its whole 30 s */
function direct(x, n, q, t, tq) {
	let { buf, mask, at } = tq, w = at[0] + at[1]
	for (let i = 0; i < n; i++) buf[(w + i) & mask] = t[i]
	put(q, x, n)
}
function copy(z, n, q) { put(q, z, n) }

/**
 * One channel's state at host rate sr, in blocks of at most `max` samples; run() drives it. The queues of the output
 * and of the guard's gain at sr start with the latency's D samples of silence and hold at most D + max + T₁ + T₂ +
 * 3·480·sr/48000: after N samples in the resamplers have given at most N + 1, and after running dry they lag by at
 * most T₁ + T₂ + 1440·sr/48000 + 2 (T₁, T₂ the taps each side in and back). The dry input's holds D + a block.
 */
function channel(sr, max) {
	let D = latency(sr), same = sr === RATE, ti = same ? 0 : taps(sr, RATE), to = same ? 0 : taps(RATE, sr)
	let size = pow2(max + D + ti + to + Math.ceil(3 * FRAME * sr / RATE) + 8), x48 = queue(pow2(Math.ceil(max * RATE / sr) + 4), 0)
	return {
		st: create(net), frame: new Float32Array(FRAME), den: new Float32Array(FRAME), z: new Float32Array(FRAME), gz: new Float32Array(FRAME),
		// this frame's input and the two before it, and the limit's share each sample came with: RNNoise's output for
		// frame f is frame f − 2's time
		dry: new Float32Array(3 * FRAME), shr: new Float64Array(3 * FRAME),
		// samples in this frame so far, its slot in dry (its index modulo 3), frames done up to 2
		at: new Int32Array(3),
		// the guard; its gain's ramp starts at 1, nothing decided yet (ramp())
		decide: online(guard), phi: Float64Array.of(1),
		// the 48 kHz input and its shares, place for place (one `at`)
		x48, s48: { buf: new Float64Array(x48.buf.length), mask: x48.mask, at: x48.at },
		into: same ? direct : resampler(sr, RATE, true), back: same ? copy : resampler(RATE, sr), gback: same ? copy : resampler(RATE, sr),
		q: queue(size, D), gq: queue(size, D), dq: queue(pow2(D + max), D),
		g: new Float32Array(max), d: new Float32Array(max),
	}
}

/** run(ch, x, y, shares, pass): host samples x in, the denoised ones latency(sr) later out into y; shares[i] the
 *  input's share mixed back for x[i] (share(), slide()); pass: music passes */
function run(ch, x, y, shares, pass) {
	let n = x.length, m = y.length
	ch.into(x, n, ch.x48, shares, ch.s48)
	frames(ch, pass)
	put(ch.dq, x, n)
	pull(ch.q, y, m); pull(ch.gq, ch.g, m); pull(ch.dq, ch.d, m)
	blends(ch.g, y, ch.d, m)
}

/** The 48 kHz samples waiting in ch.x48 into frames, each one done as it fills */
function frames(ch, pass) {
	let { x48: { buf, mask, at: qa }, s48: { buf: sb }, frame, dry, shr, at } = ch, r = qa[0], len = qa[1]
	for (let k = 0; k < len; k++) {
		let v = buf[(r + k) & mask], fill = at[0], o = at[1] * FRAME + fill
		frame[fill] = v * 32768; dry[o] = v; shr[o] = sb[(r + k) & mask]
		if ((at[0] = fill + 1) === FRAME) done(ch, pass)
	}
	qa[0] = (r + len) & mask; qa[1] = 0
}

/** A full frame through RNNoise and the guard; from the third on, the output for frame f − 2 queued as denoise()'s
 *  offline path makes it: 960 samples on, the input mixed back at the share each of its samples came with, the guard's
 *  gain beside it */
function done(ch, pass) {
	let { at, dry, shr, den, z, gz } = ch, s = at[1]
	ch.st.run(ch.frame, den)
	let on = ch.decide(dry, s * FRAME, (s + 1) * FRAME)
	if (at[2] === 2) {
		let o = ((s + 1) % 3) * FRAME
		for (let k = 0; k < FRAME; k++) { let a = shr[o + k]; z[k] = (den[k] / 32768) * (1 - a) + dry[o + k] * a }
		ramps(ch.phi, !(on && pass), gz, 0, FRAME)
		ch.back(z, FRAME, ch.q); ch.gback(gz, FRAME, ch.gq)
	} else at[2]++
	at[1] = s === 2 ? 0 : s + 1; at[0] = 0
}

// The limit's glide. A host hands a jump of limit at a call's first frame, the call split there, and its calls are of
// any length (contract 1.0: Parameters, Time; What a host owes, 3): a glide that took a short call before a change for
// a host's ramp would sound otherwise as the calls were cut. So the share moves by the samples alone: a call whose value
// differs from the line's end starts a line from the share in force, the last sample's, reaching the new one GLIDE
// seconds on (a whole number of samples, fixed for the instance), the call's first sample the line's first step; a
// value that comes mid-line starts a line of its own from the share then in force; each sample's share is the line's
// own arithmetic at its place, so a jump gives the same samples whatever calls carry it and those before it. From the
// line's last sample on the share is the value itself, and a value held is the one number it was. A host's curve, a
// value at most every 32 frames (Time), is chased by lines from where the share is to each value. An instance starts on
// the limit it is built with (ctx.params), so it sounds at it from its first sample, and no path is an instance's first
// call alone. 20 ms: the line's two corners, where its slope starts and stops, spread over every frequency, the louder
// the shorter the line, and are heard where RNNoise's output and the input part (a held chord in 'enhance'); the longer
// the line, the further a host's curve of values lags behind, by about the line's length.
// s: the share in force, the line's start, its end, its samples so far, its samples in all, made with the instance.
// limit: the param's array; the shares of the call's n samples into row. Only arrays and whole numbers cross the call:
// a float in a function V8 has not optimized is a new object, so the floats stay here, where the loop over the call's
// samples has V8 optimize it within a few calls
const GLIDE = 0.02
function slide(s, limit, row, n) {
	let to = share(limit[0]), N = s[4]
	if (to !== s[2]) { s[1] = s[0]; s[2] = to; s[3] = 0 }
	let from = s[1], k = s[3], u = s[0]
	for (let i = 0; i < n; i++) {
		if (k < N) { k++; u = k < N ? from + (to - from) * (k / N) : to }
		row[i] = u
	}
	s[0] = u; s[3] = k
}

/** The error for a call the contract forbids: more frames than ctx.maxBlockSize, more channels than ctx.maxChannels */
const beyond = (n, c, max, chans) => new RangeError(n > max
	? `rnnoise: a call of ${n} frames; ctx.maxBlockSize is ${max}, the most a call may carry (each channel's queues are sized by it)`
	: `rnnoise: a call of ${c} channels; ctx.maxChannels is ${chans}, the most a call may carry (each channel's state is made by it)`)

// V8 compiles a function on the kinds of value it has seen there, and deoptimizes it, with a burst of allocation, the
// first time a stream brings another: a fraction where it saw only whole numbers (digital silence; a loud input's
// energies, whole in float32 from 2^24 on), a path not taken yet (a silent frame, the guard's first patch after
// silence, a limit's first move, music's first switch, an instance's first frames), another function at a call
// (another rate's converters, another instance's closures). And until V8 has optimized a function, each float in it is
// a new object: one called once a call is optimized after some thousands of calls, seconds of a stream in blocks of 128,
// and the guard's network, called once a patch, 10 times a second a channel, after some dozens. So the first processor
// made shows the code each kind and each path, and calls it as often, before any stream: processors of two channels
// made for it, in calls of 8 frames of quiet noise, loud noise, a voiced tone, digital silence and noise again, the
// limit moving to a fraction, to 0 and back, music switched to 'enhance' and back. One at 48 kHz and one at 44.1 kHz,
// 0.5 s, call by call, so that a call of a converter meets both kinds before V8 compiles it for one (else, on Node 20
// and 22, done() compiled for the 48 kHz copy deoptimizes at a resampler, on every frame of a stream at another rate);
// then one at 48 kHz, 1 s, whose first frames find the code compiled; then the guard's network 100 times on a patch of
// noise. Every later instance shares the code they compiled. On Node 25, V8 still compiles the guard's decision, which a
// stream makes once a patch, within a process's first stream's first second (README, Speed and memory).
let warm = () => {
	warm = () => {}
	let F = 8, s = 1, rnd = () => (s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 4294967296 - .5
	// a processor of two channels at rate sr for `sec` seconds: each call of the function returned makes its next call,
	// and returns false once it has made them all
	let play = (sr, sec) => {
		let limit = Float32Array.of(LIMIT), params = { limit, music: 'pass' }, x = new Float32Array(F), i = 0, b = 0
		let p = rnnoise({ sampleRate: sr, maxBlockSize: 128, maxChannels: 2, params }), ins = [[x, x]], outs = [[new Float32Array(F), new Float32Array(F)]]
		return () => {
			if (b >= sec * sr / F) return false
			for (let k = 0; k < F; k++, i++) {
				let t = i / sr
				x[k] = t < .15 ? 2e-3 * rnd() : t < .3 ? .6 * rnd() : t < .7 ? .1 * (Math.sin(2 * Math.PI * 180 * t) + .5 * Math.sin(2 * Math.PI * 360 * t)) : t < .9 ? 0 : .1 * rnd()
			}
			let c = b % 600
			limit[0] = c >= 100 && c < 110 ? LIMIT + 6.5 : c >= 300 && c < 310 ? 0 : LIMIT
			params.music = b >= 3000 && b < 3600 ? 'enhance' : 'pass'
			p(ins, outs, params)
			b++
			return true
		}
	}
	for (let round of [() => [play(RATE, .5), play(44100, .5)], () => [play(RATE, 1)]]) {
		let calls = round()
		for (let more = true; more;) { more = false; for (let next of calls) more = next() || more }
	}
	let p1 = new Float32Array(16 * 9 * 64)
	for (let r = 0; r < 100; r++) { for (let i = 0; i < p1.length; i++) p1[i] = 4 * rnd(); guard.rest(p1) }
}

export const rnnoise = (ctx) => {
	let sr = ctx.sampleRate, max = ctx.maxBlockSize, chans = []
	for (let k of ['maxBlockSize', 'maxChannels']) if (!(ctx[k] >= 1)) throw new TypeError(`rnnoise: ctx.${k} is ${ctx[k]}; a host gives it (contract 1.0, Context), and the processor's state is made by it`)
	warm()
	for (let c = 0; c < ctx.maxChannels; c++) chans.push(channel(sr, max))
	// the limit: its glide's state (slide()), at the share of the limit the instance is built with, and the share of the
	// input mixed back for each sample of a call
	let a = share(limitOf(ctx.params)), N = Math.max(1, Math.round(GLIDE * sr)), lim = Float64Array.of(a, a, a, N, N), shares = new Float64Array(max)
	return (inputs, outputs, params) => {
		let inp = inputs[0], out = outputs[0]
		if (!inp || !inp.length) return
		let n = inp[0].length
		if (n > max || inp.length > chans.length) throw beyond(n, inp.length, max, chans.length)
		// the share glided, not the dB value: the output is affine in the share, so along a line in it the output is the
		// renders the two values give held, mixed by the line
		slide(lim, params.limit, shares, n)
		let pass = params.music !== 'enhance'
		for (let c = 0; c < inp.length; c++) run(chans[c], inp[c], out[c], shares, pass)
	}
}
rnnoise.channels = 'any'
rnnoise.latency = ctx => latency(ctx.sampleRate)
// the same at every host rate (the tail paragraph above); the limit's share, which stops with the input, shortens it
rnnoise.tail = ({ params } = {}) => tail(limitOf(params))
rnnoise.params = {
	limit: { type: 'number', min: 0, max: 100, default: LIMIT, unit: 'dB' },
	music: { type: 'enum', values: ['pass', 'enhance'], default: 'pass' },
}
