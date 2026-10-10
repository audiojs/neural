// Mel-Band RoFormer on WebGPU, kernels of its own: the GPU form scripts/export-roformer.py writes (--engine), one 8 s
// segment a run, what its ONNX graph computes (47.8 dB from onnxruntime's CPU on the float32 graph: the float16 weights'
// rounding, as the float16 ONNX file's). Matrix products and attention on the GPU's matrix units through
// chromium_experimental_subgroup_matrix (Apple's simdgroup matrices: Chromium on Metal), float32 throughout: under the same
// load, 1.6 s a segment where onnxruntime-web's WebGPU takes 2.6 to 3.3 (an M4 Max). Attention in two passes over the
// keys (the rows' maxima, then the output in registers), never past 128 MiB a dispatch needs no binding beyond it.
//   let rf = await engine(meta, bytes)                  // null where the GPU lacks subgroup matrices
//   let { stems_spec } = await rf.run({ mix_spec })     // as an onnxruntime session: [1, 4, F, T] in and out
const TILE = 32

/** A device with what the kernels need, or null */
export async function gpu() {
  let adapter = await globalThis.navigator?.gpu?.requestAdapter({ powerPreference: 'high-performance' }).catch(() => null)
  if (!adapter || adapter.info?.isFallbackAdapter || !['subgroups', 'chromium-experimental-subgroup-matrix'].every(f => adapter.features.has(f))) return null
  let cfg = [...(adapter.info.subgroupMatrixConfigs ?? [])]
  if (!cfg.some(c => c.componentType === 'f32' && c.M === 8 && c.N === 8 && c.K === 8)) return null
  return adapter.requestDevice({
    requiredFeatures: ['subgroups', 'chromium-experimental-subgroup-matrix'],
    requiredLimits: { maxComputeWorkgroupStorageSize: 32768, maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize, maxBufferSize: adapter.limits.maxBufferSize },
  }).catch(() => null)
}

/** The engine as an onnxruntime-shaped session ({ inputs, outputs, run(feeds), free() }), or null without such a GPU */
export async function engine(meta, bytes, device) {
  device ??= await gpu()
  if (!device) return null
  let rf = await roformer(device, meta, bytes), { freqs: F, frames: T } = meta
  return {
    inputs: [{ name: 'mix_spec', dims: [1, 4, F, T], type: 'float32' }], outputs: [{ name: 'stems_spec', dims: [1, 4, F, T], type: 'float32' }],
    backend: 'roformer',
    async run(feeds) { return { stems_spec: { data: await rf.run(feeds.mix_spec.data), dims: [1, 4, F, T], type: 'float32' } } },
    free() { device.destroy() },
  }
}

export default async function roformer(device, meta, bytes) {
  const { frames: T, freqs: F, channels: S, dim: D, heads: H, depth, bands: B, groups } = meta
  const DH = 64, FF = 1536, MH = 1536, Tp = Math.ceil(T / TILE) * TILE, QKV = 3 * H * DH, QKVG = Math.ceil((QKV + H) / TILE) * TILE
  const COLS = F * S * 2, LT = Math.ceil(T / 16) * 16, LF = Math.ceil(B / 16) * 16
  const pad16 = w => Math.ceil(w / 16) * 16
  const G = groups.map(([b0, b1, W]) => ({ b0, b1, n: b1 - b0, W, Wp: pad16(W) }))

  // ---------------------------------------------------------------- tensors
  const view = n => {
    const t = meta.tensors[n], len = t.shape.reduce((a, b) => a * b, 1)
    return t.type === 'float16' ? new Float16Array(bytes.buffer, bytes.byteOffset + t.offset, len)
      : t.type === 'int32' ? new Int32Array(bytes.buffer, bytes.byteOffset + t.offset, len)
      : new Float32Array(bytes.buffer, bytes.byteOffset + t.offset, len)
  }
  const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
  const buf = (n, type = 'f32') => device.createBuffer({ size: Math.max(16, n * 4), usage })
  const upload = data => { const b = device.createBuffer({ size: Math.max(16, data.byteLength), usage, mappedAtCreation: true }); new data.constructor(b.getMappedRange()).set(data); b.unmap(); return b }
  const f32 = a => a instanceof Float32Array ? a : Float32Array.from(a)
  // a weight [batch, K, N] to [batch, Kp, Np], zeros past (rows, cols)
  const padded = (a, batch, K, N, Kp, Np, place = (k, n) => k * Np + n) => {
    const o = new Float32Array(batch * Kp * Np)
    for (let b = 0; b < batch; b++) for (let k = 0; k < K; k++) for (let n = 0; n < N; n++) o[b * Kp * Np + place(k, n)] = a[(b * K + k) * N + n]
    return o
  }

  const W = { split: [], splitB: [], splitIdx: [], m2: [], m2B: [] }
  for (const [g, gr] of G.entries()) {
    W.split.push(upload(padded(view(`split_w.${g}`), gr.n, gr.W, D, gr.Wp, D)))
    W.splitB.push(upload(f32(view(`split_b.${g}`))))
    const idx = view(`si${g}`), pi = new Int32Array(gr.n * gr.Wp).fill(COLS)
    for (let j = 0; j < gr.n; j++) for (let e = 0; e < gr.W; e++) pi[j * gr.Wp + e] = idx[j * gr.W + e]
    W.splitIdx.push(upload(pi))
    // the last mask layer: its value half to [0, W), its gate half to [Wp, Wp + W)
    const w = view(`m2_w.${g}`), b = view(`m2_b.${g}`), Wp = gr.Wp, wo = new Float32Array(gr.n * MH * 2 * Wp), bo = new Float32Array(gr.n * 2 * Wp)
    for (let j = 0; j < gr.n; j++) {
      for (let k = 0; k < MH; k++) for (let e = 0; e < gr.W; e++) {
        wo[(j * MH + k) * 2 * Wp + e] = w[(j * MH + k) * 2 * gr.W + e]
        wo[(j * MH + k) * 2 * Wp + Wp + e] = w[(j * MH + k) * 2 * gr.W + gr.W + e]
      }
      for (let e = 0; e < gr.W; e++) { bo[j * 2 * Wp + e] = b[j * 2 * gr.W + e]; bo[j * 2 * Wp + Wp + e] = b[j * 2 * gr.W + gr.W + e] }
    }
    W.m2.push(upload(wo)); W.m2B.push(upload(bo))
  }
  W.mw = [0, 1].map(i => upload(f32(view(`mw.${i}`)))); W.mb = [0, 1].map(i => upload(f32(view(`mb.${i}`))))
  W.blocks = []
  for (let d = 0; d < depth; d++) for (let k = 0; k < 2; k++) {
    const p = `blocks.${d}.${k}.`, a = p + 'layers.0.0.', f = p + 'layers.0.1.'
    W.blocks.push({
      out: upload(f32(view(p + 'out'))),
      qkvg: upload(padded(view(a + 'qkvg'), 1, D, QKV + H, D, QKVG)), gb: upload(f32(view(a + 'gb'))), wo: upload(f32(view(a + 'wo'))),
      w1: upload(f32(view(f + 'w1'))), b1: upload(f32(view(f + 'b1'))), w2: upload(f32(view(f + 'w2'))), b2: upload(f32(view(f + 'b2'))),
    })
  }
  const rope = k => { const c = view(`blocks.0.${k}.layers.0.0.cos`), s = view(`blocks.0.${k}.layers.0.0.sin`); return [upload(f32(c)), upload(f32(s))] }
  const [cosT, sinT] = rope(0), [cosF, sinF] = rope(1)
  // the masks' sources: per output column, two (base, stride: the group's Wp) into the GLU outputs, and its weight
  const i1 = view('i1'), i2 = view('i2'), wt = view('wt'), gofs = [], cols = []
  let mOff = 0, cOff = 0
  for (const gr of G) { gofs.push(mOff); cols.push(cOff); mOff += gr.n * Tp * gr.Wp; cOff += gr.n * gr.W }
  const src = c => {
    if (c >= cOff) return [0, 0, 0]
    let g = cols.findLastIndex(o => o <= c), gr = G[g], r = c - cols[g], j = Math.floor(r / gr.W), e = r % gr.W
    return [gofs[g] + j * Tp * gr.Wp + e, gr.Wp, 1]
  }
  const map = new Uint32Array(COLS * 8)                   // per column: base1, stride1, on1, base2, stride2, on2, -, -
  for (let o = 0; o < COLS; o++) map.set([...src(i1[o]), ...src(i2[o])], o * 8)
  const maskMap = upload(map)
  const maskWt = upload(Float32Array.from({ length: COLS }, (_, o) => wt[o]))

  // ---------------------------------------------------------------- activations
  const E = buf(Tp * (COLS + 1)), X = buf(B * Tp * D), XN = buf(B * Tp * D), Y = buf(B * Tp * QKVG)
  const att = Math.max(B * H * LT, T * H * LF) * DH
  const Qb = buf(att), Kb = buf(att), Vb = buf(att), Ob = buf(att), Gt = buf(B * Tp * H), Z = buf(B * Tp * H * DH)
  const FFH = buf(B * Tp * FF), H2 = buf(B * Tp * MH), Mg = buf(mOff), OUT = buf(4 * F * T)
  const splitA = G.map(gr => buf(gr.n * Tp * gr.Wp)), glu = G.map(gr => buf(gr.n * Tp * 2 * gr.Wp)), scratch = buf(4)

  // ---------------------------------------------------------------- kernels
  const pipes = new Map()
  const pipe = (key, code) => pipes.get(key) ?? pipes.set(key, device.createComputePipeline({ layout: 'auto', compute: { module: device.createShaderModule({ code }), entryPoint: 'main' } })).get(key)
  const uni = (...v) => { const b = device.createBuffer({ size: Math.ceil(v.length / 4) * 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }); device.queue.writeBuffer(b, 0, new Uint32Array(v.map(x => typeof x === 'number' && !Number.isInteger(x) ? new Uint32Array(new Float32Array([x]).buffer)[0] : x))); return b }
  const steps = []                                          // [pipeline, bindings, [x, y, z]]
  const step = (p, bindings, grid) => steps.push([p, device.createBindGroup({ layout: p.getBindGroupLayout(0), entries: bindings.map((b, i) => ({ binding: i, resource: { buffer: b } })) }), grid])

  const HEAD = `enable subgroups;\nenable chromium_experimental_subgroup_matrix;\n`
  // C[b] (M×N, row stride ldc, at c0 + b·sc) = A[b] (M×K, at a0 + b·sa) · Bm[b] (K×N, at b0 + b·sb), then the epilogue:
  // ep: '' as is; 'b' + bias (at bias0 + b·sbias); then 'g' gelu or 't' tanh; 'r' C += it (a residual)
  const MM = ep => pipe('mm' + ep, HEAD + `
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> Bm: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
struct P { M: u32, N: u32, K: u32, a0: u32, sa: u32, b0: u32, sb: u32, c0: u32, sc: u32, ldc: u32, bias0: u32, sbias: u32 }
@group(0) @binding(4) var<uniform> p: P;
var<workgroup> tile: array<f32, 1024>;
@compute @workgroup_size(32)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(subgroup_invocation_id) lane: u32) {
  _ = bias[0];   // bound whether read or not: the layout is the module's own
  let r0 = wg.y * 32u; let n0 = wg.x * 32u; let b = wg.z;
  let ab = p.a0 + b * p.sa + r0 * p.K; let bb = p.b0 + b * p.sb + n0;
  ${[0, 1, 2, 3].map(i => [0, 1, 2, 3].map(j => `var c${i}${j}: subgroup_matrix_result<f32, 8, 8>;`).join(' ')).join('\n  ')}
  for (var k = 0u; k < p.K; k += 8u) {
    ${[0, 1, 2, 3].map(i => `let a${i} = subgroupMatrixLoad<subgroup_matrix_left<f32, 8, 8>>(&A, ab + ${i * 8}u * p.K + k, false, p.K);`).join('\n    ')}
    ${[0, 1, 2, 3].map(j => `let b${j} = subgroupMatrixLoad<subgroup_matrix_right<f32, 8, 8>>(&Bm, bb + k * p.N + ${j * 8}u, false, p.N);`).join('\n    ')}
    ${[0, 1, 2, 3].map(i => [0, 1, 2, 3].map(j => `c${i}${j} = subgroupMatrixMultiplyAccumulate(a${i}, b${j}, c${i}${j});`).join(' ')).join('\n    ')}
  }
  ${[0, 1, 2, 3].map(i => [0, 1, 2, 3].map(j => `subgroupMatrixStore(&tile, ${i * 8 * 32 + j * 8}u, c${i}${j}, false, 32u);`).join(' ')).join('\n  ')}
  workgroupBarrier();
  for (var e = lane; e < 1024u; e += 32u) {
    let r = r0 + e / 32u; let n = n0 + e % 32u;
    if (r < p.M && n < p.N) {
      var v = tile[e];
      ${ep.includes('b') ? 'v += bias[p.bias0 + b * p.sbias + n];' : ''}
      ${ep.includes('g') ? 'v = 0.5 * v * (1.0 + erf_(v * 0.7071067811865476));' : ep.includes('t') ? 'v = tanh(v);' : ''}
      let o = p.c0 + b * p.sc + r * p.ldc + n;
      ${ep.includes('r') ? 'C[o] += v;' : 'C[o] = v;'}
    }
  }
}
fn erf_(x: f32) -> f32 {   // Abramowitz & Stegun 7.1.26 is 1.5e-7; GELU's own float32 erf is as close
  let t = 1.0 / (1.0 + 0.3275911 * abs(x));
  let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-x * x);
  return select(-y, y, x >= 0.0);
}`)
  const mm = (ep, A, Bm, C, bias, { M, N, K, batch = 1, a0 = 0, sa = 0, b0 = 0, sb = 0, c0 = 0, sc = 0, ldc = N, bias0 = 0, sbias = 0 }) =>
    step(MM(ep), [A, Bm, C, bias ?? maskWt, uni(M, N, K, a0, sa, b0, sb, c0, sc, ldc, bias0, sbias)], [N / 32, M / 32, batch])

  // rows of n: y = x / max(|x|, 1e-12), or in place, scaled by g (a vector): x = x / max(|x|, 1e-12) · g
  const NORM = scaled => pipe('norm' + scaled, HEAD + `
@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(2) var<storage, read> g: array<f32>;
struct P { rows: u32, n: u32 }
@group(0) @binding(3) var<uniform> p: P;
@compute @workgroup_size(32)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(subgroup_invocation_id) lane: u32) {
  _ = g[0]; _ = y[0];
  let r = wg.x + wg.y * 65535u;
  if (r >= p.rows) { return; }
  var s = 0.0;
  for (var i = lane; i < p.n; i += 32u) { let v = x[r * p.n + i]; s += v * v; }
  let k = 1.0 / max(sqrt(subgroupAdd(s)), 1e-12);
  for (var i = lane; i < p.n; i += 32u) { ${scaled ? 'x[r * p.n + i] = x[r * p.n + i] * k * g[i];' : 'y[r * p.n + i] = x[r * p.n + i] * k;'} }
}`)
  const grid1 = n => [Math.min(n, 65535), Math.ceil(n / 65535), 1]
  const norm = (x, y, rows, n, g) => step(NORM(!!g), [x, g ? scratch : y, g ?? maskWt, uni(rows, n)], grid1(rows))

  // a group's band inputs: band j's columns of E at frame t (none past T), normalized: A[j, t, e]
  const SPLIT = pipe('split', HEAD + `
@group(0) @binding(0) var<storage, read> E: array<f32>;
@group(0) @binding(1) var<storage, read> idx: array<i32>;
@group(0) @binding(2) var<storage, read_write> A: array<f32>;
struct P { Wp: u32, Tp: u32, T: u32, cols: u32 }
@group(0) @binding(3) var<uniform> p: P;
@compute @workgroup_size(32)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(subgroup_invocation_id) lane: u32) {
  let t = wg.x; let j = wg.y;
  var s = 0.0;
  for (var e = lane; e < p.Wp; e += 32u) { let v = select(0.0, E[t * (p.cols + 1u) + u32(idx[j * p.Wp + e])], t < p.T); s += v * v; }
  let k = 1.0 / max(sqrt(subgroupAdd(s)), 1e-12);
  for (var e = lane; e < p.Wp; e += 32u) {
    let v = select(0.0, E[t * (p.cols + 1u) + u32(idx[j * p.Wp + e])], t < p.T);
    A[(j * p.Tp + t) * p.Wp + e] = v * k;
  }
}`)

  // q, k, v of each head to sequences, the rotation on q and k, the gates: token r = band·Tp + t; over time a sequence is
  // a band, its positions frames (n T, padded to L); over frequency a frame, its positions bands (n B, padded to L)
  const ROPE = pipe('rope', HEAD + `
@group(0) @binding(0) var<storage, read> Yq: array<f32>;
@group(0) @binding(1) var<storage, read_write> Q: array<f32>;
@group(0) @binding(2) var<storage, read_write> K: array<f32>;
@group(0) @binding(3) var<storage, read_write> V: array<f32>;
@group(0) @binding(4) var<storage, read_write> Gt: array<f32>;
@group(0) @binding(5) var<storage, read> cs: array<f32>;
@group(0) @binding(6) var<storage, read> sn: array<f32>;
@group(0) @binding(7) var<storage, read> gb: array<f32>;
struct P { time: u32, Tp: u32, T: u32, L: u32, ld: u32 }
@group(0) @binding(8) var<uniform> p: P;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) li: vec3u) {
  let t = wg.x; let band = wg.y; let h = wg.z; let d = li.x;
  if (t >= p.T) { return; }
  let r = band * p.Tp + t;
  var seq = band; var pos = t;
  if (p.time == 0u) { seq = t; pos = band; }
  let o = ((seq * ${H}u + h) * p.L + pos) * 64u + d;
  let y = r * p.ld + h * 64u;
  let rot = select(Yq[y + d - 32u], -Yq[y + d + 32u], d < 32u);
  let rk = select(Yq[y + ${H * DH}u + d - 32u], -Yq[y + ${H * DH}u + d + 32u], d < 32u);
  Q[o] = Yq[y + d] * cs[pos * 64u + d] + rot * sn[pos * 64u + d];
  K[o] = Yq[y + ${H * DH}u + d] * cs[pos * 64u + d] + rk * sn[pos * 64u + d];
  V[o] = Yq[y + ${2 * H * DH}u + d];
  if (d == 0u) { Gt[r * ${H}u + h] = 1.0 / (1.0 + exp(-(Yq[r * p.ld + ${QKV}u + h] + gb[h]))); }
}`)
  // and back: Z[r, h·64 + d] = O · gate
  const MERGE = pipe('merge', HEAD + `
@group(0) @binding(0) var<storage, read> O: array<f32>;
@group(0) @binding(1) var<storage, read> Gt: array<f32>;
@group(0) @binding(2) var<storage, read_write> Zb: array<f32>;
struct P { time: u32, Tp: u32, T: u32, L: u32 }
@group(0) @binding(3) var<uniform> p: P;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) li: vec3u) {
  let t = wg.x; let band = wg.y; let h = wg.z; let d = li.x;
  let r = band * p.Tp + t;
  if (t >= p.T) { Zb[r * ${H * DH}u + h * 64u + d] = 0.0; return; }
  var seq = band; var pos = t;
  if (p.time == 0u) { seq = t; pos = band; }
  Zb[r * ${H * DH}u + h * 64u + d] = O[((seq * ${H}u + h) * p.L + pos) * 64u + d] * Gt[r * ${H}u + h];
}`)
  // attention in two passes over the keys, 32 a block, 16 query rows a subgroup: the rows' maxima first (Q Kᵀ alone),
  // then exp(S − max) and the output accumulated in registers, no rescaling; keys past n masked
  const BR = 16, BC = 32, SG = 4, I2 = [0, 1], J4 = [0, 1, 2, 3], D8 = [...Array(8).keys()]
  const scores = `${I2.map(i => J4.map(j => `var s${i}_${j}: subgroup_matrix_result<f32, 8, 8>;`).join(' ')).join('\n    ')}
    for (var k = 0u; k < 8u; k++) {
      ${I2.map(i => `let q${i} = subgroupMatrixLoad<subgroup_matrix_left<f32, 8, 8>>(&Q, base + (r0 + ${i * 8}u) * 64u + k * 8u, false, 64u);`).join('\n      ')}
      ${J4.map(j => `let k${j} = subgroupMatrixLoad<subgroup_matrix_right<f32, 8, 8>>(&K, base + (c0 + ${j * 8}u) * 64u + k * 8u, true, 64u);`).join('\n      ')}
      ${I2.map(i => J4.map(j => `s${i}_${j} = subgroupMatrixMultiplyAccumulate(q${i}, k${j}, s${i}_${j});`).join(' ')).join('\n      ')}
    }
    ${I2.map(i => J4.map(j => `subgroupMatrixStore(&S[sg], ${i * 8 * BC + j * 8}u, s${i}_${j}, false, ${BC}u);`).join(' ')).join('\n    ')}
    workgroupBarrier();`
  const ATT = pipe('att', HEAD + `
@group(0) @binding(0) var<storage, read> Q: array<f32>;
@group(0) @binding(1) var<storage, read> K: array<f32>;
@group(0) @binding(2) var<storage, read> V: array<f32>;
@group(0) @binding(3) var<storage, read_write> O: array<f32>;
struct P { n: u32, np: u32 }
@group(0) @binding(4) var<uniform> p: P;
var<workgroup> S: array<array<f32, ${BR * BC}>, ${SG}>;
var<workgroup> Ow: array<array<f32, ${BR * 64}>, ${SG}>;
@compute @workgroup_size(${32 * SG})
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(subgroup_id) sg: u32, @builtin(subgroup_invocation_id) lane: u32) {
  let b = wg.y;
  let r0 = (wg.x * ${SG}u + sg) * ${BR}u;
  let base = b * p.np * 64u;
  let row = lane / 2u; let half = (lane % 2u) * ${BC / 2}u;    // two lanes a row, 16 columns each
  var mx = -1e30;
  for (var c0 = 0u; c0 < p.n; c0 += ${BC}u) {
    ${scores}
    for (var j = 0u; j < ${BC / 2}u; j++) { if (c0 + half + j < p.n) { mx = max(mx, S[sg][row * ${BC}u + half + j]); } }
    workgroupBarrier();
  }
  mx = max(mx, subgroupShuffleXor(mx, 1u));
  var l = 0.0;
  ${I2.map(i => D8.map(t => `var o${i}_${t}: subgroup_matrix_result<f32, 8, 8>;`).join(' ')).join('\n  ')}
  for (var c0 = 0u; c0 < p.n; c0 += ${BC}u) {
    ${scores}
    for (var j = 0u; j < ${BC / 2}u; j++) {
      let at = row * ${BC}u + half + j;
      let e = select(0.0, exp(S[sg][at] - mx), c0 + half + j < p.n);
      S[sg][at] = e; l += e;
    }
    workgroupBarrier();
    ${I2.map(i => J4.map(j => `let p${i}_${j} = subgroupMatrixLoad<subgroup_matrix_left<f32, 8, 8>>(&S[sg], ${i * 8 * BC + j * 8}u, false, ${BC}u);`).join(' ')).join('\n    ')}
    ${D8.map(t => J4.map(j => `let v${j}_${t} = subgroupMatrixLoad<subgroup_matrix_right<f32, 8, 8>>(&V, base + (c0 + ${j * 8}u) * 64u + ${t * 8}u, false, 64u);`).join(' ') + '\n    ' + I2.map(i => J4.map(j => `o${i}_${t} = subgroupMatrixMultiplyAccumulate(p${i}_${j}, v${j}_${t}, o${i}_${t});`).join(' ')).join(' ')).join('\n    ')}
    workgroupBarrier();
  }
  l += subgroupShuffleXor(l, 1u);
  ${I2.map(i => D8.map(t => `subgroupMatrixStore(&Ow[sg], ${i * 8 * 64 + t * 8}u, o${i}_${t}, false, 64u);`).join(' ')).join('\n  ')}
  workgroupBarrier();
  // lane pairs hold their row's sum; each lane writes half its row
  for (var d = 0u; d < 32u; d++) {
    let r = r0 + row;
    if (r < p.n) { O[base + r * 64u + half * 2u + d] = Ow[sg][row * 64u + half * 2u + d] / l; }
  }
}`)
  // GLU: value half by the sigmoid of the gate half
  const GLU = pipe('glu', HEAD + `
@group(0) @binding(0) var<storage, read> Yg: array<f32>;
@group(0) @binding(1) var<storage, read_write> Mo: array<f32>;
struct P { rows: u32, Wp: u32, at: u32 }
@group(0) @binding(2) var<uniform> p: P;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gi: vec3u) {
  let i = gi.x + gi.y * 65535u * 64u;
  if (i >= p.rows * p.Wp) { return; }
  let r = i / p.Wp; let e = i % p.Wp;
  Mo[p.at + i] = Yg[r * 2u * p.Wp + e] / (1.0 + exp(-Yg[r * 2u * p.Wp + p.Wp + e]));
}`)
  // the masks averaged over the bands that cover a column, times the spectrum: out planes [s·2 + c][f][t]
  const MASK = pipe('mask', HEAD + `
@group(0) @binding(0) var<storage, read> Mo: array<f32>;
@group(0) @binding(1) var<storage, read> map: array<u32>;
@group(0) @binding(2) var<storage, read> wt: array<f32>;
@group(0) @binding(3) var<storage, read> E: array<f32>;
@group(0) @binding(4) var<storage, read_write> Out: array<f32>;
struct P { T: u32, F: u32, cols: u32 }
@group(0) @binding(5) var<uniform> p: P;
fn mask(t: u32, o: u32) -> f32 {
  let a = map[o * 8u]; let sa = map[o * 8u + 1u]; let on1 = map[o * 8u + 2u];
  let b = map[o * 8u + 3u]; let sb = map[o * 8u + 4u]; let on2 = map[o * 8u + 5u];
  var v = 0.0;
  if (on1 == 1u) { v += Mo[a + t * sa]; }
  if (on2 == 1u) { v += Mo[b + t * sb]; }
  return v * wt[o];
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gi: vec3u) {
  let t = gi.x; let fs = gi.y;          // fs = f·S + s
  if (t >= p.T) { return; }
  let f = fs / ${S}u; let s = fs % ${S}u;
  let o = fs * 2u;
  let er = E[t * (p.cols + 1u) + o]; let ei = E[t * (p.cols + 1u) + o + 1u];
  let mr = mask(t, o); let mi = mask(t, o + 1u);
  Out[((s * 2u) * p.F + f) * p.T + t] = er * mr - ei * mi;
  Out[((s * 2u + 1u) * p.F + f) * p.T + t] = er * mi + ei * mr;
}`)

  // ---------------------------------------------------------------- the run, recorded once
  for (const [g, gr] of G.entries()) {
    step(SPLIT, [E, W.splitIdx[g], splitA[g], uni(gr.Wp, Tp, T, COLS)], [Tp, gr.n, 1])
    mm('b', splitA[g], W.split[g], X, W.splitB[g], { M: Tp, N: D, K: gr.Wp, batch: gr.n, sa: Tp * gr.Wp, sb: gr.Wp * D, c0: gr.b0 * Tp * D, sc: Tp * D, sbias: D })
  }
  for (const [i, w] of W.blocks.entries()) {
    const time = i % 2 === 0, L = time ? LT : LF, n = time ? T : B, seqs = time ? B : T
    norm(X, XN, B * Tp, D)
    mm('', XN, w.qkvg, Y, null, { M: B * Tp, N: QKVG, K: D })
    step(ROPE, [Y, Qb, Kb, Vb, Gt, time ? cosT : cosF, time ? sinT : sinF, w.gb, uni(time ? 1 : 0, Tp, T, L, QKVG)], [Tp, B, H])
    step(ATT, [Qb, Kb, Vb, Ob, uni(n, L)], [Math.ceil(L / (BR * SG)), seqs * H, 1])
    step(MERGE, [Ob, Gt, Z, uni(time ? 1 : 0, Tp, T, L)], [Tp, B, H])
    mm('r', Z, w.wo, X, null, { M: B * Tp, N: D, K: H * DH })
    norm(X, XN, B * Tp, D)
    mm('bg', XN, w.w1, FFH, w.b1, { M: B * Tp, N: FF, K: D })
    mm('br', FFH, w.w2, X, w.b2, { M: B * Tp, N: D, K: FF })
    norm(X, X, B * Tp, D, w.out)
  }
  mm('bt', X, W.mw[0], FFH, W.mb[0], { M: Tp, N: MH, K: D, batch: B, sa: Tp * D, sb: D * MH, sc: Tp * MH, sbias: MH })
  mm('bt', FFH, W.mw[1], H2, W.mb[1], { M: Tp, N: MH, K: MH, batch: B, sa: Tp * MH, sb: MH * MH, sc: Tp * MH, sbias: MH })
  for (const [g, gr] of G.entries()) {
    mm('b', H2, W.m2[g], glu[g], W.m2B[g], { M: Tp, N: 2 * gr.Wp, K: MH, batch: gr.n, a0: gr.b0 * Tp * MH, sa: Tp * MH, sb: MH * 2 * gr.Wp, sc: Tp * 2 * gr.Wp, sbias: 2 * gr.Wp })
    const n = gr.n * Tp * gr.Wp
    step(GLU, [glu[g], Mg, uni(gr.n * Tp, gr.Wp, gofs[g])], [Math.min(Math.ceil(n / 64), 65535), Math.ceil(n / 64 / 65535), 1])
  }
  step(MASK, [Mg, maskMap, maskWt, E, OUT, uni(T, F, COLS)], [Math.ceil(T / 64), F * S, 1])
  const zero = device.createCommandEncoder()
  for (const b of [Qb, Kb, Vb]) zero.clearBuffer(b)
  device.queue.submit([zero.finish()])

  const stage = new Float32Array(Tp * (COLS + 1)), read = device.createBuffer({ size: 4 * F * T * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ })
  return {
    steps: steps.length,
    async run(spec) {
      // planes [s·2 + c][f][t] → E[t][(f·S + s)·2 + c], a zero column past them, rows past T zero
      stage.fill(0)
      for (let s = 0; s < S; s++) for (let c = 0; c < 2; c++) for (let f = 0; f < F; f++) {
        const plane = ((s * 2 + c) * F + f) * T, col = (f * S + s) * 2 + c
        for (let t = 0; t < T; t++) stage[t * (COLS + 1) + col] = spec[plane + t]
      }
      device.queue.writeBuffer(E, 0, stage)
      const e = device.createCommandEncoder(), pass = e.beginComputePass()
      for (const [p, bg, grid] of steps) { pass.setPipeline(p); pass.setBindGroup(0, bg); pass.dispatchWorkgroups(...grid) }
      pass.end()
      e.copyBufferToBuffer(OUT, 0, read, 0, 4 * F * T * 4)
      device.queue.submit([e.finish()])
      await read.mapAsync(GPUMapMode.READ)
      const out = new Float32Array(read.getMappedRange().slice(0))
      read.unmap()
      return out
    },
  }
}
