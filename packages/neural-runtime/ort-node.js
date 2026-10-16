// The ONNX Runtime of Node (and of Bun and Deno): onnxruntime-node, on the CPU and on the GPU through its own providers
// (WebGPU on Dawn, CoreML). runtime.js imports it as '#ort', which package.json "imports" maps here by Node's "node"
// condition: no browser build resolves onnxruntime-node, and no process polyfill decides which runtime loads.
export const isNode = true
export const runtimes = { node: () => import('onnxruntime-node') }
