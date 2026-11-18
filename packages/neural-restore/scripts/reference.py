"""The JS pipeline's fixtures, from PyTorch: fixtures/in.f32 (10.5 s at 48 kHz, two runs of infer.py's chunks: three of
compare.py's degraded VoiceBank pairs back to back), fixtures/spec.f32 (dsp.stft of its first 4096 samples, re then
im, bin-major) and fixtures/out.f32 (infer.restore of it by the checkpoint).

    python reference.py CKPT [--pairs DIR]       DIR: a compare.py directory (default ~/.cache/audiojs/data/restore/eval-290k)
"""
import argparse, os, sys, numpy as np, soundfile as sf, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from dsp import SR, stft
from infer import Model, restore

a = argparse.ArgumentParser(); a.add_argument('ckpt'); a.add_argument('--pairs', default=os.path.expanduser('~/.cache/audiojs/data/restore/eval-290k'))
o = a.parse_args(); out = os.path.join(HERE, '..', 'fixtures'); os.makedirs(out, exist_ok=True)
x = np.concatenate([sf.read(os.path.join(o.pairs, f'{k:02d}-in.wav'), dtype='float32')[0] for k in (0, 1, 2)])[: int(10.5 * SR)]
x = torch.from_numpy(np.ascontiguousarray(x))
X = stft(x[None, :4096])[0]
spec = torch.cat([X.real, X.imag]).numpy().astype(np.float32)
with torch.no_grad(): y = restore(Model(o.ckpt, torch.device('cpu')), x)
for name, v in (('in', x.numpy()), ('spec', spec), ('out', y.numpy())): v.astype(np.float32).tofile(os.path.join(out, f'{name}.f32'))
print('fixtures:', len(x) / SR, 's in,', spec.shape, 'spectrum,', len(y), 'samples out')
