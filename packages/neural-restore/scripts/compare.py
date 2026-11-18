"""A checkpoint against what exists, on validation pairs drawn as the trainer draws them (VoiceBank's test speakers through
the chain, seed 12345): the degraded input, deepfilter() at its default limit, deepfilter(0), auto('speech') (made by
baselines.mjs with `audio`), the checkpoint, and the reference. Each output at the reference's level, both through the
pair's band limit; LSD, SI-SDR, PESQ (wideband), DNSMOS SIG / BAK / OVRL (bench/speech.py), bandwidth.

    python compare.py CKPT DIR [--n 24] [--rirs DIR] [--noises DIR] [--val val-vb.json]     writes DIR/<k>-{in,ref,restore}.wav, then scores
    node baselines.mjs DIR                                                                   the baselines into DIR (run from audio's repo)
"""
import argparse, os, sys, torch, numpy as np, soundfile as sf
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.environ.get('AUDIO_BENCH', os.path.join(HERE, *['..'] * 5, 'audio', 'bench')))   # audio's bench/speech.py: PESQ, DNSMOS
from data import Pairs, Rirs, Noises, manifests
from dsp import to_level, band, SR, bandwidth
from model import Net
from train import lsd, sisdr

def make(ckpt, out, n, rirs, noises, val):
    torch.manual_seed(0); os.makedirs(out, exist_ok=True)
    pairs = Pairs(manifests(val), Rirs(rirs).items if rirs else (), Noises(noises) if noises else None, 4.0, seed=12345)
    ck = torch.load(ckpt, map_location='cpu'); net = Net(**ck['cfg']); net.load_state_dict(ck['net']); net.eval()
    it = iter(pairs)
    with torch.no_grad():
        for i in range(n):
            x, y, bw = next(it)
            if os.path.exists(f'{out}/{i:02d}-ref.wav'): x, y = load(f'{out}/{i:02d}-in.wav'), load(f'{out}/{i:02d}-ref.wav')   # the same pairs for every checkpoint
            xin, g = to_level(x[None]); yh = (net(xin) / g[:, None])[0]
            for name, v in (('in', x), ('restore', yh), ('ref', y)): sf.write(f'{out}/{i:02d}-{name}.wav', v.numpy(), SR, subtype='FLOAT')
            open(f'{out}/{i:02d}-bw.txt', 'w').write(str(float(bw)))

def load(p): return torch.from_numpy(sf.read(p, dtype='float32', always_2d=True)[0].mean(1))

def score(out, n):
    import speech
    from pesq import pesq
    systems = [s for s in ('in', 'dfn18', 'dfn0', 'auto', 'restore', 'ref') if os.path.exists(f'{out}/00-{s}.wav')]
    rows = {s: [] for s in systems}
    for i in range(n):
        bw = torch.tensor([float(open(f'{out}/{i:02d}-bw.txt').read())]); refb = band(load(f'{out}/{i:02d}-ref.wav')[None], bw)[0]
        r16 = speech.to16(refb.numpy().astype('float64'), SR)
        for s in systems:
            v = load(f'{out}/{i:02d}-{s}.wav'); m = min(len(v), len(refb)); v, rb = v[:m], refb[:m]
            v = v * (rb.pow(2).mean() / (v.pow(2).mean() + 1e-12)).sqrt()
            vb = band(v[None], bw)[0]; v16 = speech.to16(vb.numpy().astype('float64'), SR)
            r = dict(lsd=lsd(rb[None], vb[None]), sisdr=sisdr(rb[None], vb[None]), pesq=pesq(16000, r16[:len(v16)], v16, 'wb'), bw=bandwidth(vb))
            r['sig'], r['bak'], r['ovrl'] = speech.dnsmos(vb.numpy().astype('float64'), SR); rows[s].append(r)
    print(f'{"system":8s} {"LSD":>5s} {"SI-SDR":>7s} {"PESQ":>5s} {"SIG":>5s} {"BAK":>5s} {"OVRL":>5s} {"band":>6s}')
    for s in systems:
        m = {k: np.mean([r[k] for r in rows[s]]) for k in rows[s][0]}
        print(f'{s:8s} {m["lsd"]:5.2f} {m["sisdr"]:7.1f} {m["pesq"]:5.2f} {m["sig"]:5.2f} {m["bak"]:5.2f} {m["ovrl"]:5.2f} {m["bw"]:6.0f}')

if __name__ == '__main__':
    a = argparse.ArgumentParser(); a.add_argument('ckpt'); a.add_argument('dir'); a.add_argument('--n', type=int, default=24)
    D = os.environ.get('RESTORE_DATA', os.path.expanduser('~/.cache/audiojs/data/restore'))
    a.add_argument('--rirs', default=f'{D}/rirs'); a.add_argument('--noises', default=f'{D}/noise'); a.add_argument('--val', default=f'{D}/val-vb.json')
    o = a.parse_args(); make(o.ckpt, o.dir, o.n, o.rirs, o.noises, o.val); score(o.dir, o.n)
