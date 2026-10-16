// The ONNX Runtime of a browser, worker or worklet: onnxruntime-web, one build per backend, each a literal import that
// a bundler or a CDN (esm.sh) resolves, each loaded only when its backend runs. '#ort' wherever "node" is not a
// condition (package.json "imports").
export const isNode = false
export const runtimes = {
  wasm: () => import('onnxruntime-web/wasm'),
  webgpu: () => import('onnxruntime-web/webgpu'),
  webgl: () => import('onnxruntime-web/webgl'),
}
