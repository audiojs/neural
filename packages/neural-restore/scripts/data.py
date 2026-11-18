"""Clean voices, rooms and noises on disk → training pairs.

    python data.py scan OUT.json DIR [DIR...] [--kind speech|sing] [--lang en|fr|ru|..] [--skip REGEX]
        every audio file under the DIRs (paths matching --skip left out: held-out speakers): its rate, length and
        the bandwidth it carries (dsp.bandwidth), one manifest; a corpus at 16 or 24 kHz is kept, its band edge
        recorded, so the losses stop at it (dsp.band)
    python data.py pairs OUT_DIR N --clean A.json[:W][,B.json] [--rirs DIR] [--noises DIR] [--seg 4]
        N training pairs as the trainer would draw them, <k>-in.wav and <k>-ref.wav: listen before training

Pairs(manifests, rirs, noises, seg) draws a crop of `seg` seconds from a random file (by length, a manifest's
share scaled by its `:W`), at 48 kHz mono, at a random level, through degrade.Chain, and yields (degraded, clean,
bandwidth_hz). An IterableDataset: each worker seeds itself. Rirs(dir) holds every response of a directory in
memory at 48 kHz (aligned to the direct peak, 2.5 s at most, continued to the full band, unit energy); Noises(dir)
reads a random crop of a random file on demand, so a bank of hours costs no memory.
"""
import glob, json, os, random, re, sys
from multiprocessing import Pool
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import IterableDataset, get_worker_info
from dsp import SR, bandwidth, resample, rms_db
from degrade import Chain, extend_rir, extend_noise

EXT = ('.wav', '.flac', '.ogg', '.mp3', '.opus')

def files(d):
    return sorted(p for p in glob.glob(os.path.join(d, '**', '*'), recursive=True) if p.lower().endswith(EXT))

def read(path, start=0, frames=-1):
    x, sr = sf.read(path, start=start, frames=frames, dtype='float32', always_2d=True)
    return torch.from_numpy(x.mean(1)), sr

def _item(p):
    try:
        info = sf.info(p)
        x, sr = read(p, frames=min(info.frames, info.samplerate * 30))
    except Exception as e:
        print('skip', p, e, file=sys.stderr); return None
    bw = bandwidth(resample(x, sr), SR) if sr != SR else bandwidth(x, SR)
    return dict(path=p, sr=sr, frames=info.frames, bw=round(bw))

def scan(out, dirs, kind='speech', lang='en', skip=None):
    items = json.load(open(out)) if os.path.exists(out) else []
    seen = {i['path'] for i in items}
    todo = [p for d in dirs for p in files(d) if p not in seen and not (skip and re.search(skip, p))]
    print(f'{len(todo)} files to scan, {len(items)} already in {out}', file=sys.stderr)
    with Pool() as pool:
        for k, r in enumerate(pool.imap(_item, todo, 32)):
            if r is None: continue
            items.append(dict(r, kind=kind, lang=lang))
            if k % 500 == 0:
                print(k, r['path'], r['sr'], round(r['frames'] / r['sr'], 1), 's', r['bw'], 'Hz', file=sys.stderr)
                json.dump(items, open(out, 'w'))
    json.dump(items, open(out, 'w'))
    print(f'{len(items)} files, {sum(i["frames"] / i["sr"] for i in items) / 3600:.1f} h, {sum(i["frames"] / i["sr"] for i in items if i["bw"] >= 20000) / 3600:.1f} h full band', file=sys.stderr)
    return items

class Rirs:
    def __init__(self, d, limit=None):
        self.items = []
        for p in (files(d)[:limit] if d else []):
            try: x, sr = read(p)
            except Exception: continue
            bw = min(bandwidth(x, sr), sr / 2)
            x = resample(x, sr)
            k = int(x.abs().argmax()); x = x[max(0, k - int(0.002 * SR)): k + int(2.5 * SR)]
            if len(x) < SR // 10: continue
            x = extend_rir(x, bw)
            self.items.append(x / (x.pow(2).sum().sqrt() + 1e-9))

class Noises:
    def __init__(self, d, limit=None):
        self.items = []
        for p in (files(d)[:limit] if d else []):
            try: i = sf.info(p)
            except Exception: continue
            if i.frames >= i.samplerate // 2: self.items.append([p, i.samplerate, i.frames, None])
        self.w = np.array([i[2] / i[1] for i in self.items], dtype=np.float64)
        if len(self.w): self.w /= self.w.sum()
    def __len__(self): return len(self.items)
    def crop(self, n):
        for _ in range(8):
            it = self.items[np.random.choice(len(self.items), p=self.w)]
            p, sr, frames, bw = it
            m = int(n * sr / SR) + 1; start = random.randint(0, max(0, frames - m))
            try: x, _ = read(p, start, min(m, frames - start))
            except Exception: continue
            if bw is None:  # the file's band edge, measured once per worker
                try: y, _ = read(p, 0, min(frames, sr * 30)); bw = it[3] = min(bandwidth(y, sr), sr / 2)
                except Exception: bw = it[3] = sr / 2
            x = resample(x, sr)
            if len(x) < n: x = x.repeat(n // len(x) + 1)
            x = x[:n]
            if x.pow(2).mean() < 1e-10: continue
            return extend_noise(x, bw)
        return None

def manifests(spec):
    """'a.json:2,b.json' → [(path, weight)]"""
    out = []
    for m in spec.split(','):
        path, _, w = m.partition(':'); out.append((path, float(w or 1)))
    return out

class Pairs(IterableDataset):
    def __init__(self, manifests, rirs=(), noises=None, seg=4.0, p=None, level=(-35, -12), seed=0):
        self.items, w = [], []
        for path, k in manifests:
            for i in json.load(open(path)):
                if i['frames'] >= i['sr'] * 0.5: self.items.append(i); w.append(k * i['frames'] / i['sr'])
        self.w = np.array(w); self.w /= self.w.sum()
        self.chain = Chain(rirs, noises, p)
        self.n, self.level, self.seed = int(seg * SR), level, seed

    def crop(self):
        while True:
            i = self.items[np.random.choice(len(self.items), p=self.w)]
            n = int(self.n * i['sr'] / SR) + 1
            start = random.randint(0, max(0, i['frames'] - n))
            try: x, sr = read(i['path'], start, min(n, i['frames'] - start))
            except Exception: continue
            x = resample(x, sr)
            if len(x) < self.n: x = torch.nn.functional.pad(x, (0, self.n - len(x)))
            x = x[: self.n]
            if rms_db(x) > -45: return x, float(min(i['bw'], sr / 2))

    def __iter__(self):
        w = get_worker_info()
        s = self.seed + (w.id if w else 0) + 1000 * int.from_bytes(os.urandom(2), 'little')
        random.seed(s); np.random.seed(s % 2 ** 32); torch.manual_seed(s)
        while True:
            x, bw = self.crop()
            x = x * 10 ** ((random.uniform(*self.level) - float(rms_db(x))) / 20)
            y = self.chain(x)
            peak = y.abs().max()
            if peak > 0.99: x, y = x * 0.99 / peak, y * 0.99 / peak
            yield y, x, torch.tensor(bw)

if __name__ == '__main__':
    import argparse
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('cmd', choices=['scan', 'pairs']); a.add_argument('out'); a.add_argument('dirs', nargs='*')
    a.add_argument('--kind', default='speech', choices=['speech', 'sing']); a.add_argument('--lang', default='en'); a.add_argument('--skip')
    a.add_argument('--clean'); a.add_argument('--rirs'); a.add_argument('--noises'); a.add_argument('--seg', type=float, default=4.0)
    o = a.parse_args()
    if o.cmd == 'scan':
        scan(o.out, o.dirs, o.kind, o.lang, o.skip)
    else:
        n = int(o.dirs[0]) if o.dirs else 10
        pairs = Pairs(manifests(o.clean), Rirs(o.rirs).items if o.rirs else (), Noises(o.noises) if o.noises else None, o.seg)
        os.makedirs(o.out, exist_ok=True)
        for k, (y, x, bw) in zip(range(n), pairs):
            sf.write(os.path.join(o.out, f'{k}-in.wav'), y.numpy(), SR, subtype='FLOAT')
            sf.write(os.path.join(o.out, f'{k}-ref.wav'), x.numpy(), SR, subtype='FLOAT')
            print(k, f'bandwidth {int(bw)} Hz')
