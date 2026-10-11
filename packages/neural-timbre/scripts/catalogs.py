#!/usr/bin/env python3
"""Build the DX7 catalogs from stored reference features, on MPS: the bulk path.

    python scripts/catalogs.py --run NAME [--model encoder.int8.onnx] [--encoding int8-base64] [--dir DIR] [--only factory,all]

Reads the reference rows of the refs (training voices), dev and heldout feature sets that
scripts/factory.js wrote, embeds them with the run's encoder (weights rounded to int8 per row when
--model is the int8 graph, as scripts/export.py rounds them), averages each voice's audible
references (a note below 4 int16 steps, which some key scalings give, is skipped) and writes ~/.cache/audiojs/neural/timbre/catalogs/dx7-{factory,dev,heldout,all}.json in the format
timbre.js catalog() writes and readCatalog() checks, bound to --model's SHA-256 and the features'.
scripts/catalog.js is the same builder through @audio/neural-runtime, rendering on the fly, for
small catalogs; on dx7-factory the two agree (README, Catalogs). The package ships dx7-factory and
dx7-all at int4: --encoding int4-base64 --dir catalogs --only factory,all.
"""
import argparse, base64, hashlib, json, os, sys
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).parent))
from train import Store, DATA, voices as load_voices  # noqa: E402
from export import load, round_int8, OUT  # noqa: E402

FACTORY = ('yamaha-factory', 'yamaha-vrc')


def cartridge(f):
    """'factory/rom1a.syx' -> 'ROM1A'; other cartridges keep their path under the source folder, as scripts/catalog.js."""
    if f.startswith('factory/') or f.startswith('vrc/'): return Path(f).stem.upper()
    return f.split('/', 1)[1]


def pack(values, bits):
    """Signed integers `bits` apiece in a little-endian bit stream, as timbre.js pack()."""
    v = np.asarray(values, np.int64).ravel() & ((1 << bits) - 1)
    return np.packbits(((v[:, None] >> np.arange(bits)) & 1).astype(np.uint8).ravel(), bitorder='little').tobytes()


def quantize(vectors, bits=8):
    """Per-row scale max|v| / (2^(bits-1) - 1), values rounded half away from zero as Math.round does for these magnitudes."""
    top = 2 ** (bits - 1) - 1
    scales = np.maximum(np.abs(vectors).max(1), 1e-30) / top
    scales = np.array([float(f'{s:.7g}') for s in scales])  # toPrecision(7), as catalog() stores them
    q = np.floor(vectors / scales[:, None] + 0.5).astype(np.int64)
    return q, scales


def write(path, entries, vectors, binding, info, encoding='int8-base64'):
    bits = {'int8-base64': 8, 'int4-base64': 4}[encoding]
    q, scales = quantize(vectors, bits)
    data = {'version': 1, 'kind': 'timbre-catalog', **binding, 'dimensions': vectors.shape[1], **info, 'entries': entries,
            'vectors': {'encoding': encoding, 'shape': list(vectors.shape), 'scales': scales.tolist(), 'data': base64.b64encode(pack(q, bits)).decode()}}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, separators=(',', ':')))
    print(f'{path}: {len(entries)} voices, {path.stat().st_size / 1e6:.2f} MB')


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--model', default='encoder.int8.onnx', help='the graph the catalogs bind to, in --out')
    ap.add_argument('--out', default=str(OUT))
    ap.add_argument('--encoding', default='int8-base64')
    ap.add_argument('--dir', help='where the catalogs go, default <out>/catalogs')
    ap.add_argument('--only', default='factory,dev,heldout,all', help='the catalogs to write')
    a = ap.parse_args()
    torch.set_num_threads(4)
    device = 'mps'
    torch.mps.set_per_process_memory_fraction(min(1.0, 3e9 / torch.mps.recommended_max_memory()))
    out = Path(a.out)
    graph, ck = load(a.run)
    if 'int8' in a.model: round_int8(graph)
    graph = graph.to(device).eval()
    feats = ck['features']
    canonical = json.dumps(feats, sort_keys=True, separators=(',', ':'))  # JSON.stringify of timbre.js's canonical()
    binding = {'encoderSha256': hashlib.sha256((out / a.model).read_bytes()).hexdigest(), 'featuresSha256': hashlib.sha256(canonical.encode()).hexdigest()}
    meta = load_voices()
    vec, loaded = {}, set()
    for name, split in (('refs', 'train'), ('dev', 'dev'), ('heldout', 'heldout')):
        if not (DATA / f'features/{name}.json').exists(): print(f'{name}: no feature set, its catalogs are skipped'); continue
        loaded.add(split)
        store = Store(name, ck['kind'])
        rows = [i for i, r in enumerate(store.rows) if r['kind'] == 'ref' and store.audible[i]]
        owners = np.array([store.rows[i]['voice'] for i in rows])
        for b in range(0, len(rows), 256):
            x = torch.from_numpy(store.read(rows[b:b + 256])).to(device).float()[:, None]
            e, _ = graph(x)
            e = e.cpu().numpy()
            for o, v in zip(owners[b:b + 256], e): vec.setdefault(int(o), []).append(v)
        print(f'{name}: {len(rows)} references', flush=True)
    ids = sorted(vec)
    print(f'{len(ids)} voices; {sum(len(v) < 10 for v in vec.values())} with fewer than ten audible references', flush=True)
    mean = F.normalize(torch.tensor(np.stack([np.mean(vec[i], 0) for i in ids])), dim=1).numpy()
    row = {v: k for k, v in enumerate(ids)}
    info = lambda which: {'name': f'DX7 {which}', 'synth': 'Yamaha DX7 (@audio/synth-dx7, MSFA)', 'references': 'MIDI notes 36, 48, 60, 72, 84 at velocities 64 and 112, held 1 s, 44.1 kHz',
                          'sources': 'names, cartridges and vectors only; each source and its stated terms: the @audio/neural-timbre README, Catalogs'}
    entry = lambda i: {'name': meta[i]['name'], 'cartridge': cartridge(meta[i]['cartridge']), 'index': meta[i]['index'] + 1, 'source': meta[i]['source'], 'algorithm': meta[i]['algorithm'], 'id': meta[i]['id']}
    for which, keep in (('factory', lambda v: v['source'] in FACTORY), ('dev', lambda v: v['split'] == 'dev'), ('heldout', lambda v: v['split'] == 'heldout'), ('all', lambda v: True)):
        if which not in a.only.split(','): continue
        chosen = [i for i in ids if keep(meta[i])]
        if not {v['split'] for v in meta if keep(v)} <= loaded: print(f'dx7-{which}: references missing, skipped'); continue
        write(Path(a.dir or out / 'catalogs') / f'dx7-{which}.json', [entry(i) for i in chosen], mean[[row[i] for i in chosen]], binding, info(which), a.encoding)


if __name__ == '__main__':
    main()
