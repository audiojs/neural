# A checkpoint of train.py → weights.js (half-precision weights, base64, in model.js's order) and
# fixtures/parity.json: inputs from the synthetic development shards with the network's outputs from
# PyTorch on the CPU in float32, computed with the exported (half-precision) weights.
#   python scripts/export.py <checkpoint.pt>
import os, sys, json, base64, hashlib
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import Net, BINS
from train import Shards, crop

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def order(net):
    n = len(net.dw)
    names = ['stem.weight', 'stem.bias'] + [f'dw.{i}.weight' for i in range(n)] + [f'dw.{i}.bias' for i in range(n)] \
        + [f'pw.{i}.weight' for i in range(n)] + [f'pw.{i}.bias' for i in range(n)] \
        + ['proj.weight', 'proj.bias', 'comb', 'bias', 'head1.weight', 'head1.bias', 'head2.weight', 'head2.bias']
    sd = net.state_dict()
    assert set(names) == set(sd), set(sd) ^ set(names)
    return names, sd


if __name__ == '__main__':
    ck = torch.load(sys.argv[1], map_location='cpu')
    net = Net(**{k: v for k, v in ck['cfg'].items()})
    net.load_state_dict(ck['state'])
    names, sd = order(net)
    flat = np.concatenate([sd[k].detach().numpy().ravel() for k in names]).astype(np.float16)
    # the model PyTorch checks against is the exported one
    at = 0
    with torch.no_grad():
        for k in names:
            n = sd[k].numel()
            sd[k].copy_(torch.from_numpy(flat[at:at + n].astype(np.float32)).view_as(sd[k]))
            at += n
    net.eval()
    data = base64.b64encode(flat.tobytes()).decode()
    cfg = {**ck['cfg'], 'K': BINS}
    sha = hashlib.sha256(flat.tobytes()).hexdigest()
    with open(os.path.join(ROOT, 'weights.js'), 'w') as f:
        f.write('// Weights of the network in model.js, half precision, base64: trained by scripts/train.py on\n')
        f.write('// audiojs synth renders and VocalSet (CC BY 4.0); exported by scripts/export.py. MIT.\n')
        f.write(f'// {flat.size} weights, sha256 {sha}\n')
        f.write('export const WEIGHTS = ' + json.dumps({'config': cfg, 'data': data}) + '\n')

    # parity fixture: 16 development frames, varied (voiced and not, several rates)
    dev = Shards('dev')
    rng = np.random.default_rng(7)
    idx = rng.choice(len(dev), 16, replace=False)
    v, y = dev.get(idx)
    x, level = crop(torch.from_numpy(v['x']), torch.zeros(16, dtype=torch.long))
    x = x.half().float()                      # stored at half precision
    with torch.no_grad():
        logits, vl = net(x, level)
    fx = {
        'weights': sha,
        'input': base64.b64encode(x.numpy().astype(np.float16).tobytes()).decode(),
        'level': level.tolist(),
        'logits': base64.b64encode(logits.numpy().astype(np.float32).tobytes()).decode(),
        'voicing': torch.sigmoid(vl).tolist(),
        'frames': 16, 'bins': BINS, 'torch': torch.__version__
    }
    os.makedirs(os.path.join(ROOT, 'fixtures'), exist_ok=True)
    json.dump(fx, open(os.path.join(ROOT, 'fixtures', 'parity.json'), 'w'))
    print(json.dumps({'weights': int(flat.size), 'sha256': sha, 'bytes': int(flat.nbytes), 'step': ck.get('step'), 'dev': ck.get('dev')}))
