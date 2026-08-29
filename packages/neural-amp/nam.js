// NAM (.nam) playback — the classic standard WaveNet (Atkinson,
// neural-amp-modeler, MIT). Offline full-buffer inference in plain JS, no
// dependencies. Covers the architecture behind the ToneHunt ecosystem's
// standard/lite/feather captures: stacked layer arrays of dilated 1-D convs
// with condition mixin, tanh (or ReLU) activation, 1×1 residual, per-layer
// head accumulation, head rechanneling, head_scale output.
//
// Weight order (verified against nam/models/wavenet export_weights and the
// exact float count of the shipped example models): per layer array —
// rechannel (1×1, no bias); per layer [dilated conv w+b, input-mixin 1×1 (no
// bias), layer1x1 w+b]; head-rechannel (bias per head_bias) — then head_scale
// as the final float. Unsupported variants (gated, FiLM, head1x1, custom
// heads, LSTM) throw rather than guess.

const ACT = {
  Tanh: Math.tanh,
  ReLU: x => (x > 0 ? x : 0),
  Hardtanh: x => (x < -1 ? -1 : x > 1 ? 1 : x),
  LeakyReLU: x => (x > 0 ? x : 0.01 * x),
}

export function parse(source) {
  let nam = typeof source === 'string' ? JSON.parse(source) : source
  if (nam.architecture !== 'WaveNet')
    throw new Error(`nam: unsupported architecture ${nam.architecture} (WaveNet only)`)
  let { layers, head, head_scale } = nam.config
  if (head) throw new Error('nam: custom head not supported')
  let weights = nam.weights
  let w = 0
  let take = n => {
    let out = weights.slice(w, w + n)
    if (out.length < n) throw new Error('nam: weights exhausted')
    w += n
    return out
  }

  let arrays = layers.map(cfg => {
    if (cfg.gated) throw new Error('nam: gated layers not supported')
    if (cfg.condition_size !== 1) throw new Error('nam: condition_size must be 1')
    let act = ACT[cfg.activation]
    if (!act) throw new Error(`nam: unsupported activation ${cfg.activation}`)
    let C = cfg.channels, K = cfg.kernel_size
    let rechannel = { w: take(C * cfg.input_size), out: C, in: cfg.input_size }
    let layerList = cfg.dilations.map(dilation => ({
      dilation,
      conv: { w: take(C * C * K), b: take(C) },
      mixin: { w: take(C * cfg.condition_size) },
      layer1x1: { w: take(C * C), b: take(C) },
    }))
    let headRechannel = {
      w: take(cfg.head_size * C),
      b: cfg.head_bias ? take(cfg.head_size) : null,
      out: cfg.head_size,
      in: C,
    }
    return { ...cfg, act, rechannel, layers: layerList, headRechannel }
  })
  let headScale = weights[w]
  if (w + 1 !== weights.length)
    throw new Error(`nam: weight count mismatch (${w + 1} used, ${weights.length} given)`)

  let receptiveField = 1
  for (let a of arrays) for (let l of a.layers) receptiveField += (a.kernel_size - 1) * l.dilation

  return { arrays, headScale, receptiveField, sampleRate: nam.sample_rate ?? null, version: nam.version }
}

// 1×1 conv: rows[out][i] = (b[out] +) Σ_in w[out*In+in]·x[in][i]
function conv1x1(x, In, Out, w, b, L) {
  let y = Array.from({ length: Out }, (_, o) => {
    let yo = new Float32Array(L)
    if (b) yo.fill(b[o])
    return yo
  })
  for (let o = 0; o < Out; o++) {
    let yo = y[o]
    for (let c = 0; c < In; c++) {
      let wc = w[o * In + c], xc = x[c]
      for (let i = 0; i < L; i++) yo[i] += wc * xc[i]
    }
  }
  return y
}

// Causal dilated conv, kernel taps ordered oldest→newest (torch layout, output
// aligned to the last tap): y[o][i] = b[o] + Σ_c Σ_j w[o][c][j]·x[c][i-(K-1-j)·d]
function dilatedConv(x, C, K, d, w, b, L) {
  let y = Array.from({ length: C }, () => new Float32Array(L))
  for (let o = 0; o < C; o++) {
    let yo = y[o]
    yo.fill(b[o])
    for (let c = 0; c < C; c++) {
      let xc = x[c]
      for (let j = 0; j < K; j++) {
        let wj = w[(o * C + c) * K + j]
        if (wj === 0) continue
        let shift = (K - 1 - j) * d
        for (let i = shift; i < L; i++) yo[i] += wj * xc[i - shift]
      }
    }
  }
  return y
}

export function process(model, data) {
  let L = data.length
  let cond = [Float32Array.from(data)]
  let x = cond
  let head = null
  for (let a of model.arrays) {
    x = conv1x1(x, a.rechannel.in, a.rechannel.out, a.rechannel.w, null, L)
    let C = a.channels
    for (let layer of a.layers) {
      let z = dilatedConv(x, C, a.kernel_size, layer.dilation, layer.conv.w, layer.conv.b, L)
      for (let c = 0; c < C; c++) {
        let zc = z[c], m = layer.mixin.w[c], h0 = cond[0], act = a.act
        for (let i = 0; i < L; i++) zc[i] = act(zc[i] + m * h0[i])
      }
      if (head === null) head = z.map(r => Float32Array.from(r))
      else for (let c = 0; c < C; c++) { let hc = head[c], zc = z[c]; for (let i = 0; i < L; i++) hc[i] += zc[i] }
      let res = conv1x1(z, C, C, layer.layer1x1.w, layer.layer1x1.b, L)
      for (let c = 0; c < C; c++) { let xc = x[c], rc = res[c]; for (let i = 0; i < L; i++) xc[i] += rc[i] }
    }
    head = conv1x1(head, a.headRechannel.in, a.headRechannel.out, a.headRechannel.w, a.headRechannel.b, L)
  }
  let out = head[0], s = model.headScale
  for (let i = 0; i < L; i++) data[i] = s * out[i]
  return data
}

// nam(json) → callable amp: amp(data) processes in place and returns it.
export default function nam(source) {
  let model = parse(source)
  let fn = data => process(model, data)
  fn.model = model
  fn.receptiveField = model.receptiveField
  fn.sampleRate = model.sampleRate
  return fn
}
