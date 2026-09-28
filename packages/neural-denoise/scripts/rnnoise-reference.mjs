// Upstream RNNoise as the reference: builds xiph/rnnoise at 70f1d25 with its default model, writes the
// weight blobs with upstream's own dump_weights_blob, checks rnnoise.bin against them, runs the portable C
// path (vec.h, -ffp-contract=off) on the test inputs and writes their output hashes to fixtures/rnnoise.json.
// The little model's blob goes to $AUDIO_NEURAL_CACHE/rnnoise/ for the optional test.
//
//   node scripts/rnnoise-reference.mjs [workdir]       (needs git, curl, tar and a C compiler: cc)
//
// inputs() is also what test.js feeds the port.

import { execFileSync } from 'node:child_process'
import { createHash } from 'node:crypto'
import { existsSync, mkdirSync, readFileSync, writeFileSync, copyFileSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import lena from 'audio-lena/raw'

export const COMMIT = '70f1d256acd4b34a572f999a05c87bf00b67730d'
export const MODEL_SHA = '0a8755f8e2d834eff6a54714ecc7d75f9932e845df35f8b59bc52a7cfe6e8b37' // model_version at COMMIT

// Test inputs at int16 scale, whole 480-sample frames: lena's samples read as 48 kHz; the same with
// uniform noise (24-bit LCG) at ±1500; digital silence, a near-silent second (the E < 0.04 path), speech.
export function inputs() {
	let s = 1, rnd = () => (s = (s * 1664525 + 1013904223) % 4294967296, s / 4294967296 - .5)
	let L = new Float32Array(lena), n = Math.floor(L.length / 480) * 480
	let clean = Float32Array.from(L.subarray(0, n), v => v * 32768)
	let noisy = Float32Array.from(clean, v => v + 3000 * rnd())
	let quiet = new Float32Array(144000)
	for (let i = 48000; i < 96000; i++) quiet[i] = 0.05 * rnd()
	quiet.set(noisy.subarray(0, 48000), 96000)
	return { lena: clean, 'lena+noise': noisy, quiet }
}

export const sha = b => createHash('sha256').update(b instanceof Uint8Array ? b : new Uint8Array(b.buffer, b.byteOffset, b.byteLength)).digest('hex')

// Reads float32 frames at int16 scale, writes the denoised frames and one VAD probability per frame.
const HARNESS = `#include <stdio.h>
#include "rnnoise.h"
int main(int argc, char **argv) {
  FILE *fi = fopen(argv[1], "rb"), *fo = fopen(argv[2], "wb"), *fv = fopen(argv[3], "wb");
  RNNModel *m = rnnoise_model_from_filename(argv[4]);
  DenoiseState *st = rnnoise_create(m);
  float x[480];
  if (!st) return 1;
  while (fread(x, sizeof(float), 480, fi) == 480) {
    float v = rnnoise_process_frame(st, x, x);
    fwrite(x, sizeof(float), 480, fo); fwrite(&v, sizeof(float), 1, fv);
  }
  rnnoise_destroy(st); rnnoise_model_free(m); fclose(fi); fclose(fo); fclose(fv);
  return 0;
}
`

function main(work) {
	let run = (cmd, args, cwd = work) => execFileSync(cmd, args, { cwd, stdio: ['ignore', 'pipe', 'inherit'] })
	mkdirSync(work, { recursive: true })
	let src = path.join(work, 'rnnoise')
	if (!existsSync(src)) run('git', ['clone', '-q', 'https://github.com/xiph/rnnoise.git', src])
	run('git', ['checkout', '-q', COMMIT], src)
	let tgz = path.join(src, `rnnoise_data-${MODEL_SHA}.tar.gz`)
	if (!existsSync(tgz)) run('curl', ['-sfL', '-o', tgz, `https://media.xiph.org/rnnoise/models/rnnoise_data-${MODEL_SHA}.tar.gz`])
	if (sha(readFileSync(tgz)) !== MODEL_SHA) throw new Error('model tarball checksum mismatch')
	run('tar', ['xzmf', tgz], src)

	// dump_weights_blob, as upstream builds it (DUMP_BINARY_WEIGHTS, default DISABLE_DEBUG_FLOAT), per model
	let blobs = {}
	for (let v of ['', '_little']) {
		let dir = path.join(work, 'blob' + v)
		mkdirSync(dir, { recursive: true })
		copyFileSync(path.join(src, `src/rnnoise_data${v}.c`), path.join(dir, 'rnnoise_data.c'))
		copyFileSync(path.join(src, `src/rnnoise_data${v}.h`), path.join(dir, 'rnnoise_data.h'))
		copyFileSync(path.join(src, 'src/write_weights.c'), path.join(dir, 'write_weights.c'))
		run('cc', ['-O1', '-DDUMP_BINARY_WEIGHTS', '-DDISABLE_DEBUG_FLOAT', '-I', dir, '-I', path.join(src, 'include'), '-I', path.join(src, 'src'), 'write_weights.c', '-o', 'dump_weights_blob'], dir)
		run('./dump_weights_blob', [], dir)
		blobs[v || '_regular'] = path.join(dir, 'weights_blob.bin')
	}
	let bundled = sha(readFileSync(new URL('../rnnoise.bin', import.meta.url)))
	if (sha(readFileSync(blobs._regular)) !== bundled) throw new Error('rnnoise.bin differs from dump_weights_blob output')
	let cache = path.join(process.env.AUDIO_NEURAL_CACHE || path.join(os.homedir(), '.cache', 'audiojs', 'neural'), 'rnnoise')
	mkdirSync(cache, { recursive: true })
	copyFileSync(blobs._little, path.join(cache, 'rnnoise-little.bin'))

	// the portable reference: vec.h's C loops, no NEON/AVX, no FMA contraction
	writeFileSync(path.join(work, 'ref.c'), HARNESS)
	let files = ['denoise', 'rnn', 'pitch', 'kiss_fft', 'celt_lpc', 'nnet', 'nnet_default', 'parse_lpcnet_weights', 'rnnoise_data', 'rnnoise_tables'].map(f => path.join(src, 'src', f + '.c'))
	run('cc', ['-O2', '-ffp-contract=off', '-DDISABLE_NEON', '-DUSE_WEIGHTS_FILE', '-DDISABLE_DEBUG_FLOAT', '-I', path.join(src, 'include'), '-I', path.join(src, 'src'), ...files, 'ref.c', '-o', 'ref', '-lm'])

	let out = { commit: COMMIT, model: MODEL_SHA, blob: bundled, little: sha(readFileSync(blobs._little)), build: 'cc -O2 -ffp-contract=off -DDISABLE_NEON', cases: {} }
	for (let [name, x] of Object.entries(inputs())) {
		let i = path.join(work, 'in.f32'), o = path.join(work, 'out.f32'), v = path.join(work, 'vad.f32')
		writeFileSync(i, new Uint8Array(x.buffer))
		out.cases[name] = { frames: x.length / 480 }
		for (let [key, blob] of [['regular', blobs._regular], ['little', blobs._little]]) {
			run(path.join(work, 'ref'), [i, o, v, blob])
			out.cases[name][key] = { out: sha(readFileSync(o)), vad: sha(readFileSync(v)) }
		}
	}
	writeFileSync(new URL('../fixtures/rnnoise.json', import.meta.url), JSON.stringify(out, null, '\t') + '\n')
	console.log(JSON.stringify(out, null, 2))
}

if (process.argv[1] === fileURLToPath(import.meta.url)) main(path.resolve(process.argv[2] || path.join(os.tmpdir(), 'rnnoise-reference')))
