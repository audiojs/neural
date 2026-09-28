# Python DeepFilterNet3 as the reference for test.js: deepfilternet 0.5.6, the checkpoint
# models/DeepFilterNet3.zip at Rikorose/DeepFilterNet d375b2d, df.enhance.enhance(pad=True) on the test
# input 'lena+noise' (as scripts/rnnoise-reference.mjs defines it, at float scale). Writes the input and
# the output to $AUDIO_NEURAL_CACHE/deepfilternet3/reference/.
#
#   python -m venv dfn && dfn/bin/pip install "torch==2.1.2" "torchaudio==2.1.2" "numpy<2" deepfilternet==0.5.6
#   dfn/bin/python scripts/deepfilter-reference.py [files.f32 ...]   (extra files: float32 at 48 kHz, enhanced next to them)
#   dfn/bin/python scripts/deepfilter-reference.py --vbdemand         (the Accuracy table's Python row: the VoiceBank+DEMAND
#                                                                      noisy test set into ~/.cache/audiojs/data/vbdemand/out/dfn3-py/)
#
# Python 3.8 to 3.11 (DeepFilterLib 0.5.6 has no newer wheels); numpy below 2.
import io, os, sys, glob, wave, zipfile, urllib.request, numpy as np, torch
from df.enhance import init_df, enhance

COMMIT = 'd375b2d8309e0935d165700c91da9de862a99c31'
CACHE = os.environ.get('AUDIO_NEURAL_CACHE') or os.path.expanduser('~/.cache/audiojs/neural')
OUT = os.path.join(CACHE, 'deepfilternet3', 'reference')

def lena():
    here = os.path.dirname(os.path.abspath(__file__))
    while here != '/':
        p = os.path.join(here, 'node_modules', 'audio-lena', 'lena.raw')
        if os.path.exists(p): return np.fromfile(p, dtype='<f4')
        here = os.path.dirname(here)
    raise SystemExit('audio-lena not found: npm install in the package first')

def lena_noise():
    # inputs()['lena+noise'] / 32768: lena's samples at int16 scale plus uniform noise from a 32-bit LCG
    x = lena(); n = len(x) // 480 * 480
    clean = (x[:n].astype(np.float64) * 32768).astype(np.float32)
    r, s = np.empty(n), 1
    for i in range(n):
        s = (s * 1664525 + 1013904223) % 4294967296
        r[i] = s / 4294967296 - .5
    return ((clean.astype(np.float64) + 3000 * r).astype(np.float32) / np.float32(32768)).astype(np.float32)

def model():
    base = os.path.join(CACHE, 'deepfilternet3', 'DeepFilterNet3')
    if not os.path.exists(os.path.join(base, 'config.ini')):
        url = f'https://raw.githubusercontent.com/Rikorose/DeepFilterNet/{COMMIT}/models/DeepFilterNet3.zip'
        zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(url).read())).extractall(os.path.dirname(base))
    return init_df(base, log_level='error')

def vbdemand(net, state):
    d = os.path.expanduser('~/.cache/audiojs/data/vbdemand'); out = os.path.join(d, 'out', 'dfn3-py')
    os.makedirs(out, exist_ok=True)
    for f in sorted(glob.glob(os.path.join(d, 'noisy_testset_wav', '*.wav'))):
        dst = os.path.join(out, os.path.basename(f)[:-4] + '.f32')
        if os.path.exists(dst): continue
        with wave.open(f) as w:
            assert w.getframerate() == 48000 and w.getsampwidth() == 2 and w.getnchannels() == 1
            x = np.frombuffer(w.readframes(w.getnframes()), dtype='<i2').astype(np.float32) / 32768
        enhance(net, state, torch.from_numpy(x)[None], pad=True)[0].numpy().astype(np.float32).tofile(dst)

if __name__ == '__main__':
    torch.set_num_threads(4)
    net, state, _ = model()
    if '--vbdemand' in sys.argv: vbdemand(net, state); sys.exit()
    os.makedirs(OUT, exist_ok=True)
    jobs = [(os.path.join(OUT, 'lena+noise.in.f32'), lena_noise())] + [(f, np.fromfile(f, dtype='<f4')) for f in sys.argv[1:]]
    for path, x in jobs:
        if path.startswith(OUT): x.tofile(path)
        y = enhance(net, state, torch.from_numpy(x)[None], pad=True)[0].numpy().astype(np.float32)
        y.tofile(path.replace('.in.f32', '.out.f32') if path.endswith('.in.f32') else path[:-4] + '.dfn3.f32')
        print(os.path.basename(path), len(x), 'samples')
