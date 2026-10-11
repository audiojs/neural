#!/usr/bin/env python3
"""Train the timbre encoder on DX7 voices: supervised contrastive, on MPS.

    python scripts/train.py --run NAME [--kind cqt] [--steps 30000] [--limit VOICES]

Reads the feature sets scripts/factory.js writes (<data>/features/{train,dev}.json and
.<kind>.f16) and writes ~/.cache/audiojs/neural/timbre/runs/NAME/{best.pt,progress.json}.

Each step holds 12 voices, each with one of its 8 nearest training voices by parameters (the
near-copies retrieval must tell apart, gpu-font's pairing of each face with a near design), 4
views apiece. Loss: supervised contrastive over the batch's views (Khosla et al., NeurIPS 2020,
L_out, temperature 0.1), a margin softmax of each view against a proxy per training voice
(CosFace: Wang et al., CVPR 2018; scale 30 and margin 0.15 as gpu-font's style encoder), and the
voice's algorithm (1-32) read linearly from the unit embedding (weight 0.2), the DX7's operator
topology. The checkpoint kept is the best development top-1, averaged over clean and augmented
queries against a catalog of development voices.
"""
import argparse, json, math, os, threading, queue, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

DATA = Path(os.environ.get('TIMBRE_DATA', Path.home() / '.cache/audiojs/data/dx7'))
RUNS = Path(os.environ.get('AUDIO_NEURAL_CACHE', Path.home() / '.cache/audiojs/neural')) / 'timbre/runs'
TEMPERATURE, SCALE, MARGIN, HEAD = 0.1, 30.0, 0.15, 0.2
SILENT = 2 ** -13  # dry peak below 4 int16 steps: nothing to recognize


class Encoder(nn.Module):
    """Log-feature patch [B, 1, bands, frames] -> unit embedding [B, dims]. Four conv stages with 2x2
    max pooling, a 1x1 conv down to `reduce` channels, then mean and max over time of each
    (channel, band) cell and two linear layers. Frequency keeps its position to the end: brightness
    is identity, not noise."""
    def __init__(self, bands=96, dims=128, widths=(32, 64, 128, 256), reduce=96, hidden=512):
        super().__init__()
        layers, c = [], 1
        for i, w in enumerate(widths):
            for _ in range(1 if i == 0 else 2):
                layers += [nn.Conv2d(c, w, 3, padding=1, bias=False), nn.BatchNorm2d(w), nn.ReLU(inplace=True)]
                c = w
            layers.append(nn.MaxPool2d(2))
        layers += [nn.Conv2d(c, reduce, 1, bias=False), nn.BatchNorm2d(reduce), nn.ReLU(inplace=True)]
        self.features = nn.Sequential(*layers)
        cells = reduce * (bands // 2 ** len(widths))
        self.head = nn.Sequential(nn.Linear(2 * cells, hidden), nn.BatchNorm1d(hidden), nn.ReLU(inplace=True), nn.Linear(hidden, dims))
        self.config = {'bands': bands, 'dims': dims, 'widths': list(widths), 'reduce': reduce, 'hidden': hidden}

    def forward(self, x):
        h = self.features(x)                       # [B, C, F', T']
        h = h.flatten(1, 2)                        # [B, C*F', T']
        h = torch.cat([h.mean(2), h.amax(2)], 1)   # time pooled
        return F.normalize(self.head(h), dim=1)


class Store:
    """A feature set on disk, read row by row with pread: the float16 file stays out of memory."""
    def __init__(self, name, kind):
        meta = json.loads((DATA / f'features/{name}.json').read_text())
        f = meta['features'][kind]
        self.shape, self.size = (f['bands'], f['frames']), f['bands'] * f['frames'] * 2
        self.rows = [dict(zip(meta['keys'], r)) for r in meta['rows']]
        self.fd = os.open(DATA / f'features/{name}.{kind}.f16', os.O_RDONLY)
        stats = np.fromfile(DATA / f'features/{name}.stats.f32', np.float32).reshape(-1, 2)
        self.audible = stats[:, 0] >= SILENT

    def read(self, index):
        out = np.empty((len(index), *self.shape), np.float16)
        for k, i in enumerate(index):
            out[k] = np.frombuffer(os.pread(self.fd, self.size, int(i) * self.size), np.float16).reshape(self.shape)
        return out


def voices():
    return json.loads((DATA / 'voices.json').read_text())['voices']


def canonical(patches):
    """The sound-key bytes of scripts/voices.js: engine-ignored parameters zeroed."""
    p = patches[:, :145].copy()
    for op in range(6): p[:, op * 21 + 14] = 0
    p[:, 140] = 0; p[:, 136] = 0
    lfo = (p[:, 139] == 0) | (p[:, 143] == 0); p[lfo, 137:144] = 0
    flat = (p[:, 130] == p[:, 131]) & (p[:, 131] == p[:, 132]) & (p[:, 132] == p[:, 133]); p[flat, 126:130] = 0
    return p


def neighbours(ids, k=8, device='mps'):
    """The k nearest training voices of each by differing parameters, computed on the GPU in blocks."""
    patches = np.fromfile(DATA / 'patches.u8', np.uint8).reshape(-1, 155)
    p = torch.from_numpy(canonical(patches[ids]).astype(np.int16)).to(device)
    out = []
    for i in range(0, len(p), 64):
        d = (p[i:i + 64, None, :] != p[None, :, :]).sum(2).float()
        d[torch.arange(len(d)), torch.arange(i, i + len(d))] = 1e9
        out.append(d.topk(k, largest=False).indices.cpu())
    return torch.cat(out).numpy()


class Batches:
    """Background reader: `anchors` voices a step, each with one parameter neighbour, `views` views each."""
    def __init__(self, store, voice_rows, neigh, anchors, views, seed):
        self.store, self.voice_rows, self.neigh, self.anchors, self.views = store, voice_rows, neigh, anchors, views
        self.rng = np.random.default_rng(seed); self.q = queue.Queue(maxsize=6); self.stop = False
        self.thread = threading.Thread(target=self.run, daemon=True); self.thread.start()

    def one(self):
        n = len(self.voice_rows); chosen = []
        seen = set()
        for a in self.rng.permutation(n):
            if len(chosen) >= 2 * self.anchors: break
            b = int(self.rng.choice(self.neigh[a]))
            if a in seen or b in seen: continue
            chosen += [a, b]; seen |= {a, b}
        rows, labels = [], []
        for v in chosen:
            r = self.voice_rows[v]
            rows += list(self.rng.choice(r, self.views, replace=len(r) < self.views)); labels += [v] * self.views
        return torch.from_numpy(self.store.read(rows)), torch.tensor(labels)

    def run(self):
        while not self.stop: self.q.put(self.one())

    def __next__(self): return self.q.get()


def supcon(z, labels, t=TEMPERATURE):
    logits = (z @ z.T) / t
    self_mask = torch.eye(len(z), dtype=torch.bool, device=z.device)
    logits = logits.masked_fill(self_mask, -1e4)
    log_prob = logits - torch.logsumexp(logits, 1, keepdim=True)
    pos = (labels[:, None] == labels[None, :]) & ~self_mask
    return (-(log_prob * pos).sum(1) / pos.sum(1).clamp(min=1)).mean()


@torch.no_grad()
def embed(model, store, rows, device, batch=128):
    model.eval(); out = []
    for i in range(0, len(rows), batch):
        x = torch.from_numpy(store.read(rows[i:i + batch])).to(device).float()[:, None]
        out.append(model(x).float().cpu())
    return torch.cat(out)


@torch.no_grad()
def evaluate(model, store, device):
    """Catalog: each development voice's mean reference embedding. Queries: one clean and one
    augmented note per voice at a note and velocity no reference uses."""
    rows = store.rows; ok = store.audible
    refs = [i for i, r in enumerate(rows) if r['kind'] == 'ref' and ok[i]]
    e = embed(model, store, refs, device); owners = np.array([rows[i]['voice'] for i in refs])
    ids = np.unique(owners); pos = {v: k for k, v in enumerate(ids)}
    cat = torch.zeros(len(ids), e.shape[1]).index_add_(0, torch.tensor([pos[o] for o in owners]), e)
    cat = F.normalize(cat, dim=1)
    report = {}
    for cond in ('clean', 'chain'):
        q = [i for i, r in enumerate(rows) if r['kind'] == 'query' and r['cond'] == cond and ok[i] and r['voice'] in pos]
        qe = embed(model, store, q, device); truth = torch.tensor([pos[rows[i]['voice']] for i in q])
        rank = (qe @ cat.T).argsort(1, descending=True)
        hit = rank == truth[:, None]
        report[cond] = {'top1': hit[:, :1].any(1).float().mean().item(), 'top5': hit[:, :5].any(1).float().mean().item(), 'n': len(q)}
    report['catalog'] = len(ids)
    report['select'] = (report['clean']['top1'] + report['chain']['top1']) / 2
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--kind', default='cqt')
    ap.add_argument('--train', default='train', help='feature set to train on (scripts/factory.js --name)')
    ap.add_argument('--steps', type=int, default=30000)
    ap.add_argument('--limit', type=int, default=0, help='train on the first N training voices of the set')
    ap.add_argument('--anchors', type=int, default=12, help='voices a step, each with a parameter neighbour: 12 x 2 x 4 views = 96, what 3 GB of MPS memory holds with room')
    ap.add_argument('--views', type=int, default=4)
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--proxy', type=float, default=1.0, help='weight of the proxy loss, 0 for supervised contrastive alone')
    ap.add_argument('--check', type=int, default=2500)
    ap.add_argument('--seed', type=int, default=1)
    a = ap.parse_args()

    torch.set_num_threads(4); torch.manual_seed(a.seed)
    device = 'mps'
    torch.mps.set_per_process_memory_fraction(min(1.0, 3e9 / torch.mps.recommended_max_memory()))
    out = RUNS / a.run; out.mkdir(parents=True, exist_ok=True)

    train, dev = Store(a.train, a.kind), Store('dev', a.kind)
    meta = voices()
    by_voice = {}
    for i, r in enumerate(train.rows):
        if train.audible[i]: by_voice.setdefault(r['voice'], []).append(i)
    ids = sorted(v for v, rs in by_voice.items() if len(rs) >= 2)
    if a.limit: ids = ids[:a.limit]
    voice_rows = [np.array(by_voice[v]) for v in ids]
    algorithm = torch.tensor([meta[v]['algorithm'] - 1 for v in ids])
    neigh = neighbours(np.array(ids), device=device)
    print(f'{len(ids)} training voices, {sum(map(len, voice_rows))} views; dev {len(dev.rows)} rows', flush=True)

    bands = train.shape[0]
    model = Encoder(bands).to(device)
    heads = nn.Linear(128, 32).to(device)
    proxies = nn.Parameter(torch.randn(len(ids), 128, device=device) * 0.01)
    params = sum(p.numel() for p in model.parameters())
    print(f'encoder {params} parameters', flush=True)
    opt = torch.optim.AdamW([{'params': model.parameters(), 'lr': a.lr, 'weight_decay': 1e-4},
                             {'params': heads.parameters(), 'lr': a.lr, 'weight_decay': 1e-4},
                             {'params': [proxies], 'lr': 10 * a.lr, 'weight_decay': 0}])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 500) * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps))))
    batches = Batches(train, voice_rows, neigh, a.anchors, a.views, a.seed)
    history, best, t0, running = [], -1, time.time(), {}

    def check(step):
        nonlocal best
        torch.mps.empty_cache()
        rep = evaluate(model, dev, device); rep['step'] = step; rep['seconds'] = round(time.time() - t0)
        torch.mps.empty_cache()
        history.append(rep)
        if rep['select'] > best:
            best = rep['select']
            torch.save({'state': {k: v.cpu() for k, v in model.state_dict().items()}, 'config': model.config,
                        'heads': {k: v.cpu() for k, v in heads.state_dict().items()}, 'step': step, 'kind': a.kind,
                        'features': json.loads((DATA / f'features/{a.train}.json').read_text())['features'][a.kind]}, out / 'best.pt')
        (out / 'progress.json').write_text(json.dumps({'args': vars(a), 'voices': len(ids), 'parameters': params, 'best': best, 'history': history}, indent=1))
        print(f"check {step}: clean top1 {rep['clean']['top1']:.3f} top5 {rep['clean']['top5']:.3f} | chain top1 {rep['chain']['top1']:.3f} top5 {rep['chain']['top5']:.3f} "
              f"({rep['catalog']} voices, {rep['seconds']}s, {torch.mps.driver_allocated_memory() / 2**30:.1f} GB)", flush=True)

    try:
        for step in range(1, a.steps + 1):
            model.train(); heads.train()
            x, labels = next(batches)
            x = x.to(device, non_blocking=True).float()[:, None]; labels = labels.to(device)
            with torch.autocast('mps', dtype=torch.bfloat16):
                z = model(x).float()
            con = supcon(z, labels)
            cos = z @ F.normalize(proxies, dim=1).T
            onehot = F.one_hot(labels, len(ids)).bool()
            prox = F.cross_entropy(SCALE * (cos - MARGIN * onehot), labels) if a.proxy else z.sum() * 0
            alg = F.cross_entropy(heads(z) * 10, algorithm.to(device)[labels])
            loss = con + a.proxy * prox + HEAD * alg
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            for k, v in (('supcon', con), ('proxy', prox), ('algorithm', alg)): running[k] = running.get(k, 0) + v.detach()
            if step % 250 == 0:
                print(f'step {step}: ' + ' '.join(f'{k} {float(v) / 250:.3f}' for k, v in running.items()) + f' ({time.time() - t0:.0f}s)', flush=True); running = {}
                torch.mps.empty_cache()
            if step % a.check == 0 or step == a.steps: check(step)
    finally:
        batches.stop = True


if __name__ == '__main__':
    main()
