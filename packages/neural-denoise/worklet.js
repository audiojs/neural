// AudioWorkletProcessor 'neural-denoise': RNNoise on each channel of a 48 kHz AudioContext.
//
//   await ctx.audioWorklet.addModule(new URL('@audio/neural-denoise/worklet', import.meta.url))
//   let node = new AudioWorkletNode(ctx, 'neural-denoise', { processorOptions: { weights: await weights() } })
//
// Render quanta (128 samples) are collected into 10 ms frames (480); the output queue starts with
// 480 − gcd(quantum, 480) samples of silence, the least that never runs dry, so the node's latency is
// constant: that plus RNNoise's 960 samples (1408 samples, 29.3 ms, at the 128-sample quantum).
// processorOptions.limit (dB) caps the attenuation by mixing the input back in: 20 unless given, 0 for
// none (upstream's output).

import { model, create, FRAME, LIMIT } from './rnnoise.js'

const gcd = (a, b) => b ? gcd(b, a % b) : a

class NeuralDenoise extends AudioWorkletProcessor {
	constructor({ processorOptions: { weights, limit } = {} } = {}) {
		super()
		if (sampleRate !== 48000) throw new Error(`neural-denoise: RNNoise runs at 48 kHz, this context runs at ${sampleRate} Hz; create it with { sampleRate: 48000 }`)
		if (!weights) throw new Error('neural-denoise: pass processorOptions.weights (await weights())')
		this.net = model(weights)
		limit ??= LIMIT
		this.lim = limit ? 10 ** (-Math.abs(limit) / 20) : 0
		this.chans = []
	}

	channel(c, quantum) {
		let prime = FRAME - gcd(quantum, FRAME), size = FRAME + prime + quantum
		return this.chans[c] ??= {
			st: create(this.net), frame: new Float32Array(FRAME), out: new Float32Array(FRAME), fill: 0,
			// output ring: queued samples between `read` and `read + queued`
			ring: new Float32Array(size), read: 0, queued: prime,
			// dry ring for the attenuation limit, delayed by the node's latency
			dry: new Float32Array(2 * FRAME + prime + quantum), dw: 2 * FRAME + prime, dr: 0,
		}
	}

	process(inputs, outputs) {
		let input = inputs[0], output = outputs[0]
		for (let c = 0; c < output.length; c++) {
			let x = input[c] ?? input[0], y = output[c], n = y.length, s = this.channel(c, n)
			for (let i = 0; i < n; i++) {
				let v = x ? x[i] : 0
				s.frame[s.fill++] = v * 32768
				if (this.lim) { s.dry[s.dw] = v; s.dw = (s.dw + 1) % s.dry.length }
				if (s.fill === FRAME) {
					s.st.process(s.frame, s.out)
					for (let k = 0, w = (s.read + s.queued) % s.ring.length; k < FRAME; k++, w = (w + 1) % s.ring.length) s.ring[w] = s.out[k] / 32768
					s.queued += FRAME; s.fill = 0
				}
			}
			for (let i = 0; i < n; i++) {
				let v = s.queued > 0 ? s.ring[s.read] : 0
				if (s.queued > 0) { s.read = (s.read + 1) % s.ring.length; s.queued-- }
				if (this.lim) { v = v * (1 - this.lim) + s.dry[s.dr] * this.lim; s.dr = (s.dr + 1) % s.dry.length }
				y[i] = v
			}
		}
		return true
	}
}

registerProcessor('neural-denoise', NeuralDenoise)
