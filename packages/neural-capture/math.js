// Shared small math — FFT convolution, Chebyshev basis/eval, dense symmetric solve.

import { fft, ifft } from 'fourier-transform'

export const nextPow2 = n => 1 << Math.ceil(Math.log2(Math.max(2, n)))

// Half-spectrum real FFT (re/im of N/2+1 bins) — fft() returns zero-copy views
// into a per-size cache, copy immediately.
export function fftCopy(signal, N) {
  let f = new Float64Array(N)
  let n = Math.min(signal.length, N)
  for (let i = 0; i < n; i++) f[i] = signal[i]
  let [re, im] = fft(f)
  return [re.slice(), im.slice()]
}

// Linear convolution via FFT, truncated to `outLen` (reverb-convolution semantics).
export function convolve(x, h, outLen = x.length) {
  let N = nextPow2(x.length + h.length - 1)
  let [xr, xi] = fftCopy(x, N)
  let [hr, hi] = fftCopy(h, N)
  for (let k = 0; k < xr.length; k++) {
    let re = xr[k] * hr[k] - xi[k] * hi[k]
    xi[k] = xr[k] * hi[k] + xi[k] * hr[k]
    xr[k] = re
  }
  let y = ifft(xr, xi, new Float64Array(N))
  return y.subarray(0, outLen)
}

// f(x) = Σₖ cₖ·Tₖ(x/scale), k = 1..order (T₀ enters implicitly via even orders).
export function chebyEval(coeffs, scale, x) {
  let t = x / scale
  let tp = 1, tc = t, y = 0
  for (let k = 1; k <= coeffs.length; k++) {
    y += coeffs[k - 1] * tc
    let tn = 2 * t * tc - tp
    tp = tc
    tc = tn
  }
  return y
}

// Basis signals Tₖ(x/scale), k = 1..order.
export function chebyBasis(x, scale, order) {
  let n = x.length
  let basis = []
  let tp = new Float64Array(n).fill(1)
  let tc = new Float64Array(n)
  for (let i = 0; i < n; i++) tc[i] = x[i] / scale
  for (let k = 1; k <= order; k++) {
    basis.push(tc)
    let tn = new Float64Array(n)
    for (let i = 0; i < n; i++) tn[i] = 2 * (x[i] / scale) * tc[i] - tp[i]
    tp = tc
    tc = tn
  }
  return basis
}

// Least squares y ≈ Σ cₖ·zₖ via normal equations.
export function lsFit(z, y) {
  let P = z.length, n = y.length
  let G = z.map(() => new Array(P).fill(0))
  let b = new Array(P).fill(0)
  for (let j = 0; j < P; j++) {
    for (let k = j; k < P; k++) {
      let s = 0
      for (let i = 0; i < n; i++) s += z[j][i] * z[k][i]
      G[j][k] = G[k][j] = s
    }
    let s = 0
    for (let i = 0; i < n; i++) s += z[j][i] * y[i]
    b[j] = s
  }
  return solve(G, b)
}

// Gaussian elimination with partial pivot — tiny symmetric systems (order ≤ ~15).
export function solve(G, b) {
  let n = b.length
  let A = G.map((row, i) => [...row, b[i]])
  for (let c = 0; c < n; c++) {
    let p = c
    for (let r = c + 1; r < n; r++) if (Math.abs(A[r][c]) > Math.abs(A[p][c])) p = r
    ;[A[c], A[p]] = [A[p], A[c]]
    if (Math.abs(A[c][c]) < 1e-30) continue
    for (let r = 0; r < n; r++) {
      if (r === c) continue
      let f = A[r][c] / A[c][c]
      for (let k = c; k <= n; k++) A[r][k] -= f * A[c][k]
    }
  }
  return A.map((row, i) => (Math.abs(row[i]) < 1e-30 ? 0 : row[n] / row[i]))
}
