"""Reference separation for the package tests: a deterministic 9 s stereo mix (two 7.8 s
Demucs segments overlap on it) and its separation by the original Python implementation,
written next to the exported weights:

    <dir>/test.f32             mix, planar float32 (L then R), 44.1 kHz
    <dir>/test.<target>.f32    reference stem, same layout

test.js runs @audio/neural-separate on the same mix and compares, whenever the files exist.
"""
import numpy as np

RATE = 44100


def test_mix(seconds=9.0, rate=RATE):
    t = np.arange(int(seconds * rate)) / rate
    rng = np.random.RandomState(0)
    # voice-like: 220 Hz harmonics, 5 Hz vibrato, 3 Hz syllables
    phase = 2 * np.pi * np.cumsum(220 * (1 + 0.01 * np.sin(2 * np.pi * 5 * t))) / rate
    voice = sum(np.sin(k * phase) / k for k in range(1, 9)) * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t) ** 2) * 0.15
    bass = (np.sin(2 * np.pi * 55 * t) + 0.5 * np.sin(2 * np.pi * 110 * t)) * 0.2
    beat = t % 0.5
    drums = np.sin(2 * np.pi * 60 * beat) * np.exp(-beat * 30) * 0.4 + rng.randn(t.size) * np.exp(-((t + 0.25) % 0.5) * 60) * 0.1
    pad = sum(np.sin(2 * np.pi * f * t) for f in (261.63, 329.63, 392.0)) * 0.05
    return np.stack([voice + 1.1 * bass + drums + 0.7 * pad, voice + 0.9 * bass + drums + 1.3 * pad]).astype(np.float32)


def normalized(mix, run):
    """run() on the mix less its mean over its deviation (of the channels' mean, ddof 1, + 1e-8), its output scaled
    back, the mean added to no source (separate.js's separateComplex)."""
    mono = mix.astype(np.float64).mean(0)
    mean, std = mono.mean(), mono.std(ddof=1) + 1e-8
    return run(((mix - mean) / std).astype(np.float32)) * std


def chunked(mix, run, L, step=0.25):
    """ZFTurbo/Music-Source-Separation-Training's demix() (utils/model_utils.py, generic mode) over run(segment
    (C, L) float32) -> (S, C, L): segments every step·L until one reaches the end, linear fades of L/10 (the first
    segment not in, the last not out), the input reflected L - step out on each side when longer than
    2 (L - step), a segment past the end reflected out when more than half of it is input, else zero-padded;
    float64 sums (separate.js's separateComplex). -> (S, C, N)"""
    C, N = mix.shape
    hop = max(1, int(L * step))
    border = L - hop
    wide = N > 2 * border and border > 0
    x = np.pad(mix, ((0, 0), (border, border)), mode="reflect") if wide else mix
    M, fade = x.shape[1], L // 10
    starts = [k * hop for k in range(max(1, -(-(M - L) // hop) + 1))]  # until one reaches the end
    acc, wsum = None, np.zeros(M)
    for k, s in enumerate(starts):
        part = x[:, s:s + L]
        n = part.shape[1]
        part = np.pad(part, ((0, 0), (0, L - n)), mode="reflect" if n > L // 2 else "constant")
        y = np.asarray(run(part.astype(np.float32)), dtype=np.float64)
        w = np.ones(L)
        if k > 0:
            w[:fade] = np.linspace(0, 1, fade)
        if k < len(starts) - 1:
            w[L - fade:] = np.linspace(1, 0, fade)
        acc = np.zeros((y.shape[0], C, M)) if acc is None else acc
        acc[..., s:s + n] += y[..., :n] * w[:n]
        wsum[s:s + n] += w[:n]
    out = acc / np.where(wsum > 0, wsum, 1)
    return out[..., border:border + N] if wide else out


def write(out_dir, separate, seconds=9.0):
    """separate(mix (2, N) float32) -> { target: (2, N) array }"""
    mix = test_mix(seconds)
    mix.tofile(out_dir / "test.f32")
    for name, x in separate(mix).items():
        np.asarray(x, dtype=np.float32).tofile(out_dir / f"test.{name}.f32")
    print(f"wrote {out_dir}/test.f32 and its reference stems")
