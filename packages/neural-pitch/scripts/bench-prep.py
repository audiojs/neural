# Benchmark audio: three sets, five conditions, fixed before any tuning. Measurement only.
#
#   vocadito        all 40 clips, 44.1 kHz (Bittner et al. 2021, CC BY 4.0), f0 annotation
#   mdb-stem-synth  every 5th of the 230 stems by sorted name, 30 s from 1 s before the first voiced
#                   annotation (Salamon et al., ISMIR 2017, CC BY-NC 4.0), 44.1 kHz
#   mir-1k          every 4th of the 1000 clips by sorted name, right channel (voice), 16 kHz
#                   (Hsu & Jang 2010), labels in semitones every 20 ms from 20 ms
#
# Conditions: clean; pink noise at 20, 10 and 0 dB SNR (Gaussian, 1/f power spectrum, SNR over the
# whole clip, seeded by clip and condition); reverb: MIT IR Survey h052 (gym weight room, T30 1.0 s,
# direct-to-reverberant ratio +0.7 dB; Traer & McDermott, PNAS 2016), resampled to the clip's rate,
# cut before its direct-path peak, output at the input's RMS.
#
# Writes $NP_BENCH (default ~/.cache/audiojs/data/neural-pitch-bench): <set>/<cond>/<clip>.f32,
# <set>/ref/<clip>.csv (time s, f0 Hz, 0 unvoiced), <set>/meta.json ({clip: fs}).
import os, sys, json, glob, zlib
import numpy as np, soundfile as sf
from scipy.signal import resample_poly, fftconvolve

D = os.path.expanduser('~/.cache/audiojs/data')
OUT = os.environ.get('NP_BENCH', os.path.join(D, 'neural-pitch-bench'))
CONDS = ['clean', 'pink20', 'pink10', 'pink0', 'reverb']
IR = os.path.join(D, 'mit-ir', 'Audio', 'h052_Gym_WeightRoom_3txts.wav')


def pink(n, seed):
    w = np.random.default_rng(seed).standard_normal(n)
    W = np.fft.rfft(w)
    f = np.arange(W.size)
    W[1:] /= np.sqrt(f[1:])
    W[0] = 0
    p = np.fft.irfft(W, n)
    return p / np.sqrt(np.mean(p ** 2))


def ir_at(fs):
    h, sr = sf.read(IR)
    if h.ndim > 1: h = h[:, 0]
    h = h[int(np.argmax(np.abs(h))):]
    g = np.gcd(int(fs), int(sr))
    return resample_poly(h, int(fs) // g, int(sr) // g)


def degrade(x, fs, cond, key):
    if cond == 'clean': return x
    if cond == 'reverb':
        y = fftconvolve(x, ir_at(fs))[:x.size]
        return y * np.sqrt(np.mean(x ** 2) / max(np.mean(y ** 2), 1e-20))
    snr = float(cond[4:])
    n = pink(x.size, zlib.crc32(f'{key} {cond}'.encode()))
    return x + n * np.sqrt(np.mean(x ** 2)) / 10 ** (snr / 20)


def write(name, clips):
    meta = {}
    for key, x, fs, ref in clips:
        meta[key] = fs
        os.makedirs(os.path.join(OUT, name, 'ref'), exist_ok=True)
        np.savetxt(os.path.join(OUT, name, 'ref', key + '.csv'), ref, delimiter=',', fmt='%.6f')
        for c in CONDS:
            os.makedirs(os.path.join(OUT, name, c), exist_ok=True)
            degrade(x.astype(np.float64), fs, c, key).astype(np.float32).tofile(os.path.join(OUT, name, c, key + '.f32'))
    json.dump(meta, open(os.path.join(OUT, name, 'meta.json'), 'w'), indent=1)
    print(name, len(clips), 'clips', sum(x.size / fs for _, x, fs, _ in clips) / 60, 'min')


def vocadito():
    V = os.path.join(D, 'vocadito')
    for i in range(1, 41):
        k = f'vocadito_{i}'
        x, fs = sf.read(os.path.join(V, 'Audio', k + '.wav'))
        if x.ndim > 1: x = x.mean(1)
        yield k, x, fs, np.loadtxt(os.path.join(V, 'Annotations', 'F0', k + '_f0.csv'), delimiter=',')


def mdb():
    M = os.path.join(D, 'mdb-stem-synth', 'MDB-stem-synth')
    names = sorted(os.path.basename(f)[:-len('.RESYN.wav')] for f in glob.glob(os.path.join(M, 'audio_stems', '*.RESYN.wav')))
    assert len(names) == 230, len(names)
    for k in names[::5]:
        x, fs = sf.read(os.path.join(M, 'audio_stems', k + '.RESYN.wav'))
        if x.ndim > 1: x = x.mean(1)
        ref = np.loadtxt(os.path.join(M, 'annotation_stems', k + '.RESYN.csv'), delimiter=',')
        t0 = max(0.0, ref[np.argmax(ref[:, 1] > 0), 0] - 1)
        a, b = int(round(t0 * fs)), int(round((t0 + 30) * fs))
        r = ref[(ref[:, 0] >= t0) & (ref[:, 0] < t0 + 30)].copy()
        r[:, 0] -= t0
        yield k, x[a:b], fs, r


def mir1k():
    R = os.path.join(D, 'mir-1k', 'MIR-1K')
    for f in sorted(glob.glob(os.path.join(R, 'Wavfile', '*.wav'))):
        k = os.path.basename(f)[:-4]
        x, fs = sf.read(f)
        m = np.loadtxt(os.path.join(R, 'PitchLabel', k + '.pv'))
        t = 0.02 + 0.02 * np.arange(m.size)
        yield k, x[:, 1], fs, np.c_[t, np.where(m > 0, 440 * 2 ** ((m - 69) / 12), 0)]


if __name__ == '__main__':
    sets = {'vocadito': vocadito, 'mdb-stem-synth': mdb, 'mir-1k': mir1k}
    for name in (sys.argv[1:] or sets):
        write(name, list(sets[name]()))
