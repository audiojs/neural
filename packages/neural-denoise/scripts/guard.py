# The music guard's network from upstream, and its reference for test.js. Fetches inaSpeechSegmenter's speech/music/noise
# CNN (ina-foss/inaSpeechSegmenter, MIT; release 'models', keras_speech_music_noise_cnn.hdf5, sha256 pinned), folds each
# batch norm into the layer before it and writes guard.bin: per layer the kernel ([out][kh][kw][in] for convolutions,
# [out][in] for dense layers) then the bias, float16, little-endian, 782,403 values. Then runs the original float32
# model, batch norms as keras applies them, on ina's features of lena (test.js's inputs().lena / 32768, read as 48 kHz,
# 48 → 16 kHz by guard.js's 31-tap low-pass) and writes the probabilities of every patch guard.js takes (one in 10) to
# fixtures/guard.json.
#
#   python -m venv g && g/bin/pip install numpy h5py
#   g/bin/python scripts/guard.py
import os, json, hashlib, urllib.request, numpy as np, h5py

URL = 'https://github.com/ina-foss/inaSpeechSegmenter/releases/download/models/keras_speech_music_noise_cnn.hdf5'
SHA = 'f04b5e3c86fa2e81666d106b0867350fe7858c7bd006c2a74c27b5b45507e7d8'
CACHE = os.environ.get('AUDIO_NEURAL_CACHE') or os.path.expanduser('~/.cache/audiojs/neural')
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STEP = 10

def fetch():
    p = os.path.join(CACHE, 'guard', os.path.basename(URL))
    if not os.path.exists(p):
        os.makedirs(os.path.dirname(p), exist_ok=True); urllib.request.urlretrieve(URL, p)
    if hashlib.sha256(open(p, 'rb').read()).hexdigest() != SHA: raise SystemExit(f'{p}: sha256 is not {SHA}')
    return p

def layers(path):
    # [(kind, weights)] in model order: conv/dense (kernel, bias), bn (gamma, beta, mean, variance, epsilon), pool, relu
    f = h5py.File(path); cfg = json.loads(f.attrs['model_config'])['config']['layers']; g = f['model_weights']; out = []
    for l in cfg:
        c, n = l['config'], l['config']['name']; w = lambda k: np.array(g[n][n][k + ':0'], np.float32)
        if l['class_name'] in ('Conv2D', 'Dense'):
            out.append(('conv' if l['class_name'] == 'Conv2D' else 'dense', w('kernel'), w('bias')))
            if c['activation'] != 'linear': out.append((c['activation'],))
        elif l['class_name'] == 'BatchNormalization': out.append(('bn', w('gamma'), w('beta'), w('moving_mean'), w('moving_variance'), np.float32(c['epsilon'])))
        elif l['class_name'] == 'MaxPooling2D': out.append(('pool', tuple(c['pool_size']), c['padding']))
        elif l['class_name'] == 'Activation': out.append((c['activation'],))
    return out

def fold(L):
    # each batch norm into the conv or dense before it: w·k, (b − mean)·k + beta, k = gamma / √(var + ε)
    parts = []
    for i, l in enumerate(L):
        if l[0] not in ('conv', 'dense'): continue
        w, b = l[1].astype(np.float64), l[2].astype(np.float64)
        if i + 1 < len(L) and L[i + 1][0] == 'bn':
            _, g, be, m, v, eps = L[i + 1]; k = g / np.sqrt(v.astype(np.float64) + eps); w = w * k; b = (b - m) * k + be
        parts += [w.transpose(3, 0, 1, 2).ravel() if l[0] == 'conv' else w.T.ravel(), b]
    return np.concatenate(parts).astype('<f2')

def forward(L, x):
    # the keras model in float32: x [68, 21] standardized → [speech, music, noise]
    x = x[:, :, None].astype(np.float32)
    for l in L:
        if l[0] == 'conv':
            k = l[1]; kh, kw = k.shape[:2]; H, W = x.shape[0] - kh + 1, x.shape[1] - kw + 1
            cols = np.stack([x[i:i + H, j:j + W] for i in range(kh) for j in range(kw)], 2)  # H, W, kh·kw, C
            x = np.einsum('hwkc,kco->hwo', cols, k.reshape(kh * kw, k.shape[2], k.shape[3])) + l[2]
        elif l[0] == 'dense': x = x.reshape(-1) @ l[1] + l[2]
        elif l[0] == 'bn': x = (x - l[3]) / np.sqrt(l[4] + l[5]) * l[1] + l[2]
        elif l[0] == 'relu': x = np.maximum(x, 0)
        elif l[0] == 'softmax': e = np.exp(x - x.max()); x = e / e.sum()
        elif l[0] == 'pool':
            (ph, pw), pad = l[1], l[2]; H, W = x.shape[:2]
            oh, ow = (-(-H // ph), -(-W // pw)) if pad == 'same' else (H // ph, W // pw)
            th, tw = max((oh - 1) * ph + ph - H, 0), max((ow - 1) * pw + pw - W, 0)
            x = np.pad(x, ((th // 2, th - th // 2), (tw // 2, tw - tw // 2), (0, 0)), constant_values=-np.inf) if pad == 'same' else x
            x = x[:oh * ph, :ow * pw].reshape(oh, ph, ow, pw, -1).max((1, 3))
    return x

def lowpass():
    t = np.arange(31) - 15; h = np.where(t == 0, 1 / 3, np.sin(np.pi * t / 3) / (np.pi * np.where(t == 0, 1, t))) * np.kaiser(31, 6)
    return h / h.sum()

def mspec(x16):
    # sidekit's mfcc() at ina's settings: framing 400/160, per-frame pre-emphasis, numpy.hanning, |rfft 512|², trfbank, log
    mel = lambda f: 2595 * np.log10(1 + f / 700.); imel = lambda z: 700 * (10 ** (z / 2595.) - 1)
    f = imel(mel(100) + np.arange(26) * (mel(8000) - mel(100)) / 25); h = 2 / (f[2:] - f[:-2]); fb = np.zeros((24, 257)); hz = np.arange(512) / 512 * 16000
    for i in range(24):
        lo, c, hi = f[i:i + 3]
        l = np.arange(np.floor(lo * 512 / 16000) + 1, np.floor(c * 512 / 16000) + 1).astype(int)
        r = np.arange(np.floor(c * 512 / 16000) + 1, min(np.floor(hi * 512 / 16000) + 1, 512)).astype(int)
        fb[i, l] = h[i] / (c - lo) * (hz[l] - lo); fb[i, r[:-1]] = h[i] / (hi - c) * (hi - hz[r[:-1]])
    n = (len(x16) - 400) // 160 + 1; fr = np.stack([x16[160 * j:160 * j + 400] for j in range(n)]).astype(np.float64)
    fr = fr - np.c_[fr[:, :1], fr[:, :-1]] * .97
    return np.log((np.abs(np.fft.rfft(fr * np.hanning(400), 512)) ** 2) @ fb.T)

def lena():
    here = HERE
    while here != '/':
        p = os.path.join(here, 'node_modules', 'audio-lena', 'lena.raw')
        if os.path.exists(p): return np.fromfile(p, '<f4')
        here = os.path.dirname(here)
    raise SystemExit('audio-lena not found: npm install in the package first')

if __name__ == '__main__':
    L = layers(fetch()); w = fold(L); w.tofile(os.path.join(HERE, 'guard.bin'))
    x = lena(); x = (x[:len(x) // 480 * 480] * 32768).astype(np.float32) / 32768
    x16 = np.convolve(x.astype(np.float64), lowpass())[:len(x)][::3]
    M = mspec(x16)[:, :21]; P = []
    for t in range(0, len(M) - 67, STEP):
        p = M[t:t + 68]; P.append(forward(L, (p - p.mean()) / p.std()).round(6).tolist())
    json.dump({ 'hdf5': SHA, 'bin': hashlib.sha256(w.tobytes()).hexdigest(), 'lena': P }, open(os.path.join(HERE, 'fixtures', 'guard.json'), 'w'))
    print(f'guard.bin: {w.size} values, {w.nbytes} bytes; fixtures/guard.json: {len(P)} patches of lena')
