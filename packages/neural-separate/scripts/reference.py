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


def write(out_dir, separate):
    """separate(mix (2, N) float32) -> { target: (2, N) array }"""
    mix = test_mix()
    mix.tofile(out_dir / "test.f32")
    for name, x in separate(mix).items():
        np.asarray(x, dtype=np.float32).tofile(out_dir / f"test.{name}.f32")
    print(f"wrote {out_dir}/test.f32 and its reference stems")
