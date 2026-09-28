# Training on the feature shards of scripts/factory.js, on Apple MPS (or CUDA, or the CPU).
#
# Losses, per step:
#   supervised (synthetic renders, exact f0): cross-entropy to a Gaussian over bins (σ = 25 cents, as
#     CREPE: Kim et al., ICASSP 2018) at the labelled f0, and binary cross-entropy for voicing
#   equivariance (PESTO: Riou et al., ISMIR 2023): two crops of one frame, k bins apart, must give
#     posteriors whose projections φ(p) = Σ_j α^j p_j, α = 2^(1/36), differ by α^k; Huber on the log ratio
#     in bins
#   consistency (unlabelled recordings): the posterior of the degraded view must match the clean view's,
#     shifted by the crops' offset, where the clean recording is voiced (pYIN, sure frames only)
#   teacher (optional, --teacher): pYIN's pitch and voicing on the clean recording as targets for the
#     degraded view, sure frames only
# Every crop is transposed at random by up to ±MARGIN bins (the stored spectra extend that far).
#
#   python scripts/train.py --out <dir> [--steps 30000] [--C 16] [--blocks 1,2,4] [--real real] ...
import os, sys, json, time, math, argparse, glob
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import Net, decode, BINS, LOW, PER

DATA = os.environ.get('NP_DATA', os.path.expanduser('~/.cache/audiojs/data/neural-pitch'))
SPAN, MARGIN, TOP = 318, 15, 80
ALPHA = math.log(2) / (12 * PER)          # log α


class Shards:
    """Frames of <name>-*.{x,z}.f16 and .y.f32 as memory maps, read a batch at a time."""
    def __init__(self, name, views=('x',)):
        meta = json.load(open(os.path.join(DATA, name + '.json')))
        self.parts = []
        for w, n in enumerate(meta['frames']):
            v = {k: np.memmap(os.path.join(DATA, f'{name}-{w}.{k}.f16'), np.float16, 'r', shape=(n, SPAN)) for k in views}
            self.parts.append((v, np.fromfile(os.path.join(DATA, f'{name}-{w}.y.f32'), np.float32).reshape(n, 4)))
        self.n = np.array([p[1].shape[0] for p in self.parts])
        self.start = np.concatenate([[0], np.cumsum(self.n)])
        self.y = np.concatenate([p[1] for p in self.parts])
        self.views = views

    def __len__(self): return int(self.start[-1])

    def get(self, idx):
        idx = np.sort(idx)
        part = np.searchsorted(self.start, idx, side='right') - 1
        out = {k: np.empty((idx.size, SPAN), np.float32) for k in self.views}
        for p in np.unique(part):
            sel = part == p
            local = idx[sel] - self.start[p]
            for k in self.views: out[k][sel] = self.parts[p][0][k][local]
        return out, self.y[idx]


def crop(db, s):
    """dB spectra (B, SPAN), shifts (B,) → model input (B, BINS) in 0…1 and level (B,)."""
    i = (MARGIN + s)[:, None] + torch.arange(BINS, device=db.device)[None]
    x = db.gather(1, i)
    level = x.amax(1)
    return ((x - level[:, None]).clamp_min(-TOP) / TOP + 1), level


def usable(y):
    """Frames whose labels hold. The first shards carry two recipe bugs since fixed in data.js: a pitch
    fall that kept falling after its note (f0 under 25 Hz on 0.13% of frames), and tonewheel clips with
    a 1⅗' drawbar labelled unvoiced throughout (a harmonic tolerance absolute, not relative)."""
    f0 = y[:, 0]
    return ~((f0 > 0) & (f0 < 25)) & ~((y[:, 2] == 7) & (f0 == 0))


def target_bin(f0, s):
    return PER * (12 * torch.log2(f0.clamp_min(1e-3) / 440) + 69 - LOW) - s


def gauss(b, sigma=0.75):
    j = torch.arange(BINS, device=b.device)[None].float()
    g = torch.exp(-0.5 * ((j - b[:, None]) / sigma) ** 2)
    return g / g.sum(1, keepdim=True).clamp_min(1e-12)


def shift(p, k):
    """p (B, K) moved k (B,) bins up (p'[j] = p[j − k]), zeros in, renormalized."""
    j = torch.arange(BINS, device=p.device)[None] - k[:, None]
    q = p.gather(1, j.clamp(0, BINS - 1)) * ((j >= 0) & (j < BINS))
    return q / q.sum(1, keepdim=True).clamp_min(1e-12)


def logphi(logp):
    return torch.logsumexp(logp + ALPHA * torch.arange(BINS, device=logp.device)[None], 1)


def huber(e, d=1.0):
    a = e.abs()
    return torch.where(a < d, 0.5 * a * a, d * (a - 0.5 * d))


@torch.no_grad()
def evaluate(net, data, dev, n=60000, real=False):
    net.eval()
    idx = np.random.default_rng(0).choice(len(data), min(n, len(data)), replace=False)
    v, y = data.get(idx)
    y = torch.from_numpy(y).to(dev)
    res = {}
    for k in data.views:
        xs, lv = crop(torch.from_numpy(v[k]).to(dev), torch.zeros(len(idx), dtype=torch.long, device=dev))
        logits, vl = [], []
        for a in range(0, len(idx), 8192):
            l, q = net(xs[a:a + 8192], lv[a:a + 8192]); logits.append(l); vl.append(q)
        p = torch.softmax(torch.cat(logits), 1); vp = torch.sigmoid(torch.cat(vl))
        f = 440 * 2 ** ((decode(p) / PER + LOW - 69) / 12)
        ok = usable(y)
        voiced, sure = (y[:, 0] > 0) & ok, (y[:, 1] > 0) & ok
        m = voiced & sure
        c = 1200 * torch.log2(f[m] / y[m, 0])
        res[k + '_rpa'] = (c.abs() < 50).float().mean().item()
        res[k + '_rca'] = (((c + 600) % 1200 - 600).abs() < 50).float().mean().item()
        unv = ~voiced & ok
        res[k + '_vacc'] = ((vp > 0.5) == voiced)[m | unv].float().mean().item()
        res[k + '_vr'] = (vp[voiced] > 0.5).float().mean().item()
        res[k + '_vfa'] = (vp[unv] > 0.5).float().mean().item() if unv.any() else float('nan')
    net.train()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--synth', default='train'); ap.add_argument('--dev', default='dev')
    ap.add_argument('--real', default=''); ap.add_argument('--realdev', default='')
    ap.add_argument('--steps', type=int, default=30000); ap.add_argument('--batch', type=int, default=1024)
    ap.add_argument('--lr', type=float, default=3e-3); ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--C', type=int, default=16); ap.add_argument('--T', type=int, default=4)
    ap.add_argument('--blocks', default='1,2,4'); ap.add_argument('--H', type=int, default=16)
    ap.add_argument('--D1', type=int, default=72); ap.add_argument('--D2', type=int, default=144)
    ap.add_argument('--equiv', type=float, default=0.1); ap.add_argument('--cons', type=float, default=0.5)
    ap.add_argument('--teacher', type=float, default=0.0)
    ap.add_argument('--seed', type=int, default=1)
    a = ap.parse_args()

    torch.set_num_threads(4)
    torch.manual_seed(a.seed)
    dev = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    if dev.type == 'mps': torch.mps.set_per_process_memory_fraction(min(1.0, 3e9 / torch.mps.recommended_max_memory()))
    os.makedirs(a.out, exist_ok=True)

    synth, sdev = Shards(a.synth), Shards(a.dev)
    real = Shards(a.real, ('x', 'z')) if a.real else None
    rdev = Shards(a.realdev, ('x', 'z')) if a.realdev else None
    net = Net(C=a.C, T=a.T, blocks=[int(b) for b in a.blocks.split(',')], D1=a.D1, D2=a.D2, H=a.H).to(dev)
    params = sum(p.numel() for p in net.parameters())
    print(json.dumps({**vars(a), 'params': params, 'macs': net.macs(), 'frames': len(synth), 'real_frames': len(real) if real else 0, 'device': str(dev)}), flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.05)
    rng = np.random.default_rng(a.seed)
    t0, best, log = time.time(), -1, []

    for step in range(1, a.steps + 1):
        v, y = synth.get(rng.integers(0, len(synth), a.batch))
        db, y = torch.from_numpy(v['x']).to(dev), torch.from_numpy(y).to(dev)
        B = db.shape[0]
        s1 = torch.randint(-MARGIN, MARGIN + 1, (B,), device=dev)
        s2 = (s1 + torch.randint(-MARGIN, MARGIN + 1, (B,), device=dev)).clamp(-MARGIN, MARGIN)
        x1, l1 = crop(db, s1); x2, l2 = crop(db, s2)
        z1, v1 = net(x1, l1); z2, v2 = net(x2, l2)
        voiced, ok = y[:, 0] > 0, usable(y)
        b1 = target_bin(y[:, 0], s1)
        m = voiced & ok & (y[:, 1] > 0) & (b1 > 1) & (b1 < BINS - 2)
        sup = -(gauss(b1[m]) * F.log_softmax(z1[m], 1)).sum(1).mean() if m.any() else z1.sum() * 0
        voi = F.binary_cross_entropy_with_logits(v1[ok], voiced[ok].float())
        lp1, lp2 = F.log_softmax(z1, 1), F.log_softmax(z2, 1)
        eqv = huber((logphi(lp2) - logphi(lp1)) / ALPHA - (s1 - s2).float())[voiced & ok].mean()
        loss = sup + voi + a.equiv * eqv
        stats = {'sup': sup, 'voi': voi, 'eqv': eqv}

        if real is not None:
            v, y = real.get(rng.integers(0, len(real), a.batch // 2))
            dx, dz, y = torch.from_numpy(v['x']).to(dev), torch.from_numpy(v['z']).to(dev), torch.from_numpy(y).to(dev)
            B = dx.shape[0]
            s1 = torch.randint(-MARGIN, MARGIN + 1, (B,), device=dev); s2 = torch.randint(-MARGIN, MARGIN + 1, (B,), device=dev)
            xa, la = crop(dx, s1); xb, lb = crop(dz, s2)
            za, va = net(xa, la); zb, vb = net(xb, lb)
            sure, tv = y[:, 1] > 0, y[:, 0] > 0
            m = sure & tv
            pa = torch.softmax(za, 1)
            lpb = F.log_softmax(zb, 1)
            cons = -(shift(pa.detach(), s1 - s2)[m] * lpb[m]).sum(1).mean() if m.any() else zb.sum() * 0
            eqr = huber((logphi(lpb) - logphi(F.log_softmax(za, 1))) / ALPHA - (s1 - s2).float())[m].mean() if m.any() else zb.sum() * 0
            loss = loss + a.cons * cons + a.equiv * eqr
            stats.update(cons=cons, eqr=eqr)
            if a.teacher:
                bt = target_bin(y[:, 0], s2)
                mt = m & (bt > 1) & (bt < BINS - 2)
                tch = -(gauss(bt[mt]) * lpb[mt]).sum(1).mean() if mt.any() else zb.sum() * 0
                tvo = F.binary_cross_entropy_with_logits(vb[sure], tv[sure].float())
                loss = loss + a.teacher * (tch + tvo)
                stats.update(tch=tch, tvo=tvo)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sched.step()

        if step % 500 == 0:
            print(json.dumps({'step': step, 't': round(time.time() - t0), **{k: round(v.item(), 4) for k, v in stats.items()}}), flush=True)
        if step % 2500 == 0 or step == a.steps:
            r = evaluate(net, sdev, dev)
            if rdev is not None: r.update({'real_' + k: v for k, v in evaluate(net, rdev, dev).items()})
            score = r['x_rpa'] + r['x_vacc'] + (r.get('real_z_rpa', 0) + r.get('real_z_vacc', 0))
            r.update(step=step, t=round(time.time() - t0), score=score)
            log.append(r)
            print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}), flush=True)
            if score > best:
                best = score
                torch.save({'state': net.state_dict(), 'cfg': net.cfg, 'step': step, 'dev': r, 'args': vars(a), 'params': params}, os.path.join(a.out, 'best.pt'))
    torch.save({'state': net.state_dict(), 'cfg': net.cfg, 'step': a.steps, 'args': vars(a), 'params': params}, os.path.join(a.out, 'last.pt'))
    json.dump({'params': params, 'macs': net.macs(), 'seconds': time.time() - t0, 'log': log, 'args': vars(a)}, open(os.path.join(a.out, 'train.json'), 'w'), indent=1)
    print('done', round(time.time() - t0), 's', flush=True)


if __name__ == '__main__':
    main()
