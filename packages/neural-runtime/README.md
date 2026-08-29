# @audio/neural-runtime

> One inference adapter: load an ONNX model, run it in Node or the browser, worklet-ready.

Every neural atom needs the same three things — load bytes, run tensors through a model, free the session — on top of two different runtimes (`onnxruntime-node` on the server, `onnxruntime-web` in the browser/worker/worklet). This package is that one adapter: `asr`, `align`, `separate`, `diarize`, `tts`, `denoise` and anything else in the lane build on it instead of each wiring ONNX Runtime themselves.

```
npm install @audio/neural-runtime
```

```js
import { load, tensor } from '@audio/neural-runtime'

let session = await load('https://example.com/model.onnx') // fetched, cached, progress-able
let out = await session.run({ input: tensor(new Float32Array(...), [1, 40, 1]) })
out.output.data   // Float32Array | Int32Array | BigInt64Array | Uint8Array | string[]
session.free()
```

`load` accepts a URL string, `Uint8Array`, or `ArrayBuffer`. `tensor(data, dims, type?)` builds a plain `{ data, dims, type }` — no ORT import at call time; the adapter converts it to `ort.Tensor` lazily, inside `run()`, once a backend is resolved.

## Backends

| `opts.backend` | Runtime | Environment |
|---|---|---|
| `'auto'` (default) | picks `'node'` in Node, `'wasm'` in a browser/worker/worklet | either |
| `'node'` | `onnxruntime-node` | Node only |
| `'wasm'` | `onnxruntime-web` | browser/worker/worklet only |
| `'webgpu'` | `onnxruntime-web/webgpu` | browser/worker/worklet only |
| `'webgl'` | `onnxruntime-web` | browser/worker/worklet only |

Neither `onnxruntime-node` nor `onnxruntime-web` is a hard dependency — both are optional peers (`peerDependenciesMeta.optional`), resolved by dynamic `import()` the first time a backend is used and memoized after. Asking for a backend your environment can't run (`'webgpu'` in Node, `'node'` in a browser) throws immediately, naming what's missing and where it does work; a resolvable but uninstalled package throws naming the install command. `backends()` reports what's actually usable here: `['node']` in Node, `['wasm']` or `['wasm', 'webgpu']` (when `navigator.gpu` exists) in a browser.

`run()` queues concurrent calls onto one sequential chain per session — safe regardless of whether the underlying ORT build tolerates overlapping `run()` calls. `free()` maps to `session.release()`, is idempotent, and `run()` after `free()` throws.

`inputs`/`outputs` on the session report `{ name, dims, type }` when the installed ORT version exposes `inputMetadata`/`outputMetadata` (present since onnxruntime-common added shape/type reporting), else `{ name }` only — never guessed.

## Caching

`load(url, opts)` and `fetchModel(url, opts)` (the fetch primitive, also used by `fetchJson` for tokenizer/config files hosted next to a model) cache by default (`opts.cache = false` to disable):

- **Node**: `~/.cache/audiojs/neural/<sha256(url)>`, override the base directory with `$AUDIO_NEURAL_CACHE`. A sidecar `.size` file records the expected byte count; a mismatch (truncated or otherwise corrupted entry) is treated as a cache miss and re-fetched.
- **Browser**: the [Cache API](https://developer.mozilla.org/en-US/docs/Web/API/Cache), store `'audio-neural'`, keyed by URL.
- `file://` URLs read straight from disk (Node only), bypassing the cache — there's nothing to cache against.
- `opts.progress?.({ loaded, total })` fires as bytes arrive; `total` is `null` when the server doesn't send `Content-Length`.
- `opts.fetch` overrides the ambient `fetch` (a custom client, auth headers, a mock in tests).

## Worklets

`onnxruntime-web` loads its `.wasm` binary from a URL it resolves relative to its own module by default — a resolution an `AudioWorkletProcessor` can't do (no relative module location, no `document`). Pass `opts.wasmPaths` (a URL prefix string, or a `{ 'ort-wasm.wasm': url, ... }` map) to point it at wherever you're serving the onnxruntime-web assets; it becomes `ort.env.wasm.wasmPaths` before the session is created. `opts.threads` sets `ort.env.wasm.numThreads` (SharedArrayBuffer + cross-origin isolation required for >1).

## API

| Export | |
|---|---|
| `load(model, opts?)` | → `Promise<Session>` — `model`: URL string \| `Uint8Array` \| `ArrayBuffer` |
| `tensor(data, dims, type?)` | → plain `Tensor`; `type` inferred from `data`'s constructor when omitted |
| `backends()` | → `Promise<string[]>` — backends usable in this environment |
| `fetchModel(url, opts?)` | → `Promise<Uint8Array>` — cached fetch with progress |
| `fetchJson(url, opts?)` | → `Promise<unknown>` — same cache/progress path, `JSON.parse`d |

`Session`: `{ run(feeds, outputNames?) → Promise<Record<string, Tensor>>, inputs, outputs, backend, free() }`.

`load(model, opts)` — `opts`: `{ backend = 'auto', threads?, cache = true, progress?, fetch?, executionProviders?, graphOptimizationLevel?: 'all' | 'basic' | 'disabled', logLevel?, wasmPaths? }`.

**Not for**: choosing or shipping a model — this package has no model zoo, no bundled weights, and no opinion on architecture. `neural-amp` and the rest of the lane bring the model; this loads and runs it.

---

Part of the [@audio/neural](https://github.com/audiojs/neural) lane. Weights are out of scope here — see the umbrella README's weights-licensing policy for anything that ships pretrained parameters.

MIT © [audiojs](https://github.com/audiojs)
