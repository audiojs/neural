// CMA-ES minimizer (Hansen's standard formulation: weighted recombination,
// rank-1 + rank-μ covariance update, cumulative step-size adaptation).
// Deterministic: seeded RNG, no Math.random. Dimensions here are synth knob
// counts (n ≤ ~20), so the eigendecomposition is a plain Jacobi sweep.

function mulberry32(seed) {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

function gaussian(rnd) {
  let spare = null
  return () => {
    if (spare !== null) {
      let v = spare
      spare = null
      return v
    }
    let u, v, s
    do {
      u = 2 * rnd() - 1
      v = 2 * rnd() - 1
      s = u * u + v * v
    } while (s >= 1 || s === 0)
    let m = Math.sqrt(-2 * Math.log(s) / s)
    spare = v * m
    return u * m
  }
}

// Jacobi eigendecomposition of symmetric C → { B (columns = eigenvectors), D (sqrt eigenvalues) }
function eigen(C, n) {
  let A = C.map(r => Float64Array.from(r))
  let B = Array.from({ length: n }, (_, i) => Float64Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)))
  for (let sweep = 0; sweep < 60; sweep++) {
    let off = 0
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += A[p][q] * A[p][q]
    if (off < 1e-24) break
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) {
      if (Math.abs(A[p][q]) < 1e-18) continue
      let theta = (A[q][q] - A[p][p]) / (2 * A[p][q])
      let t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1))
      let c = 1 / Math.sqrt(t * t + 1), s = t * c
      for (let k = 0; k < n; k++) {
        let akp = A[k][p], akq = A[k][q]
        A[k][p] = c * akp - s * akq
        A[k][q] = s * akp + c * akq
      }
      for (let k = 0; k < n; k++) {
        let apk = A[p][k], aqk = A[q][k]
        A[p][k] = c * apk - s * aqk
        A[q][k] = s * apk + c * aqk
      }
      for (let k = 0; k < n; k++) {
        let bkp = B[k][p], bkq = B[k][q]
        B[k][p] = c * bkp - s * bkq
        B[k][q] = s * bkp + c * bkq
      }
    }
  }
  let D = Float64Array.from({ length: n }, (_, i) => Math.sqrt(Math.max(A[i][i], 1e-20)))
  return { B, D }
}

// cmaes(f, x0, sigma0, { maxEvals = 100·n², seed = 1, tolFun = 1e-12 }) → { x, fx, evals }
export default function cmaes(f, x0, sigma0, opts = {}) {
  let n = x0.length
  let seed = opts.seed ?? 1
  let maxEvals = opts.maxEvals ?? 100 * n * n
  let rnd = mulberry32(seed)
  let randn = gaussian(rnd)

  let lambda = opts.lambda ?? 4 + Math.floor(3 * Math.log(n))
  let mu = Math.floor(lambda / 2)
  let weights = Float64Array.from({ length: mu }, (_, i) => Math.log(mu + 0.5) - Math.log(i + 1))
  let wsum = weights.reduce((s, w) => s + w, 0)
  for (let i = 0; i < mu; i++) weights[i] /= wsum
  let mueff = 1 / weights.reduce((s, w) => s + w * w, 0)

  let cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
  let cs = (mueff + 2) / (n + mueff + 5)
  let c1 = 2 / ((n + 1.3) ** 2 + mueff)
  let cmu = Math.min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
  let damps = 1 + 2 * Math.max(0, Math.sqrt((mueff - 1) / (n + 1)) - 1) + cs
  let chiN = Math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))

  let m = Float64Array.from(x0)
  let sigma = sigma0
  let C = Array.from({ length: n }, (_, i) => Float64Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)))
  let pc = new Float64Array(n), ps = new Float64Array(n)
  let { B, D } = eigen(C, n)
  let eigenEvals = 0
  let evals = 0
  let bestX = Float64Array.from(x0), bestF = f(x0)
  evals++

  while (evals + lambda <= maxEvals) {
    // sample λ candidates: x = m + σ·B·(D∘z)
    let zs = [], xs = [], fs = []
    for (let k = 0; k < lambda; k++) {
      let z = Float64Array.from({ length: n }, () => randn())
      let x = Float64Array.from(m)
      for (let i = 0; i < n; i++) {
        let bd = 0
        for (let j = 0; j < n; j++) bd += B[i][j] * D[j] * z[j]
        x[i] += sigma * bd
      }
      zs.push(z)
      xs.push(x)
      fs.push(f(x))
      evals++
    }
    let order = fs.map((v, i) => i).sort((a, b) => fs[a] - fs[b])
    if (fs[order[0]] < bestF) {
      bestF = fs[order[0]]
      bestX = Float64Array.from(xs[order[0]])
    }

    // recombination
    let mOld = Float64Array.from(m)
    m.fill(0)
    let zw = new Float64Array(n)
    for (let r = 0; r < mu; r++) {
      let x = xs[order[r]], z = zs[order[r]], w = weights[r]
      for (let i = 0; i < n; i++) {
        m[i] += w * x[i]
        zw[i] += w * z[i]
      }
    }

    // step-size path (uses B·zw — isotropic under C)
    let bz = new Float64Array(n)
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) bz[i] += B[i][j] * zw[j]
    let csf = Math.sqrt(cs * (2 - cs) * mueff)
    let psNorm = 0
    for (let i = 0; i < n; i++) {
      ps[i] = (1 - cs) * ps[i] + csf * bz[i]
      psNorm += ps[i] * ps[i]
    }
    psNorm = Math.sqrt(psNorm)
    let hsig = psNorm / Math.sqrt(1 - (1 - cs) ** (2 * evals / lambda)) / chiN < 1.4 + 2 / (n + 1) ? 1 : 0

    // covariance paths
    let ccf = Math.sqrt(cc * (2 - cc) * mueff)
    for (let i = 0; i < n; i++) pc[i] = (1 - cc) * pc[i] + hsig * ccf * (m[i] - mOld[i]) / sigma
    let c1a = c1 * (1 - (1 - hsig) * cc * (2 - cc))
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      let rankMu = 0
      for (let r = 0; r < mu; r++) {
        let x = xs[order[r]]
        rankMu += weights[r] * (x[i] - mOld[i]) * (x[j] - mOld[j]) / (sigma * sigma)
      }
      C[i][j] = (1 - c1a - cmu) * C[i][j] + c1 * pc[i] * pc[j] + cmu * rankMu
    }

    sigma *= Math.exp((cs / damps) * (psNorm / chiN - 1))

    if (evals - eigenEvals > lambda / (c1 + cmu) / n / 10) {
      eigenEvals = evals
      ;({ B, D } = eigen(C, n))
    }
    if (sigma * Math.max(...D) < (opts.tolX ?? 1e-12)) break
    if (fs[order[0]] <= (opts.tolFun ?? -Infinity)) break
  }
  return { x: bestX, fx: bestF, evals }
}
