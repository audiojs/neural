# Wide-band PESQ, and STOI, of each system scripts/floor.mjs rendered under OUT (and of the noisy input), against the
# clean take, both at 16 kHz (scipy.signal.resample_poly, 1:3, as scripts/accuracy.py scores): OUT/pesq.json,
# { system: { take: { pesq, stoi, ... } } }. From the repository's root:
#   python3 -I packages/neural-denoise/scripts/floor.py OUT [stoi] [full] [systems=a,b]
# PESQ: ITU-T P.862.2 wideband MOS-LQO (python-pesq); STOI: Taal, Hendriks, Heusdens, Jensen 2011 (pystoi, extended=False).
# full: also SI-SDR and DNSMOS SIG, BAK, OVRL at 16 kHz by scripts/accuracy.py's own sisdr() and dnsmos(), the Accuracy
# section's scoring, in place of floor.mjs score's (SI-SDR at 48 kHz; @audio/neural-quality at the output's rate).
# systems=a,b: those alone (input among them by name), merged into an OUT/pesq.json already there
import sys, os, json
import numpy as np
from scipy.signal import resample_poly
from pesq import pesq

root, with_stoi, full = sys.argv[1], 'stoi' in sys.argv[2:], 'full' in sys.argv[2:]
only = next((a[8:].split(',') for a in sys.argv[2:] if a.startswith('systems=')), None)
if with_stoi: from pystoi import stoi
if full:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from accuracy import sisdr, dnsmos
takes = [t['id'] for t in json.load(open(os.path.join(root, 'set.json')))]
rd = lambda p: resample_poly(np.fromfile(p, dtype=np.float32).astype(np.float64), 1, 3)
systems = ['input'] + sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)) and d not in ('set', 'y'))
if only: systems = [s for s in systems if s in only]
old = json.load(open(os.path.join(root, 'pesq.json'))) if only and os.path.exists(os.path.join(root, 'pesq.json')) else {}
out = {}
for s in systems:
    out[s] = {}
    for t in takes:
        x = rd(os.path.join(root, 'set', t + '.clean.f32'))
        y = rd(os.path.join(root, 'set', t + '.noisy.f32') if s == 'input' else os.path.join(root, s, t + '.f32'))
        n = min(len(x), len(y)); x, y = x[:n], y[:n]
        r = {'pesq': float(pesq(16000, x, y, 'wb'))}
        if with_stoi: r['stoi'] = float(stoi(x, y, 16000, extended=False))
        if full:
            r['sisdr'] = sisdr(x, y)
            r['sig'], r['bak'], r['ovrl'] = dnsmos(y)
        out[s][t] = r
    print(s, len(out[s]), 'takes', file=sys.stderr)
json.dump({**old, **out}, open(os.path.join(root, 'pesq.json'), 'w'))
