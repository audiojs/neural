#!/usr/bin/env python3
"""Export a trained encoder to ONNX for @audio/neural-runtime, with parity fixtures.

    python scripts/export.py --run NAME [--out ~/.cache/audiojs/neural/timbre]

Writes to --out:
  encoder.onnx        float32: features [N, 1, bands, frames] -> embedding [N, dims] (unit),
                      algorithm [N, 32] (logits of the DX7 algorithm head)
  encoder.int8.onnx   the same graph, every conv and linear weight rounded per output row to int8
                      (gpu-font's per-row weight rounding) and restored by DequantizeLinear
  parity.json         fixture patches from the development set (float16, base64), their PyTorch
                      embeddings for each graph, and the top-5 of each against a PyTorch catalog of
                      development voices: test.js checks ONNX in JS against them
Batch norm is folded into the convolutions first, so both graphs compute what training evaluated.
"""
import argparse, base64, json, os, sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).parent))
from train import Encoder, Store, RUNS, DATA  # noqa: E402

OUT = Path(os.environ.get('AUDIO_NEURAL_CACHE', Path.home() / '.cache/audiojs/neural')) / 'timbre'


class Graph(nn.Module):
    """The deployed function: unit embedding and algorithm logits."""
    def __init__(self, encoder, heads):
        super().__init__(); self.encoder, self.heads = encoder, heads

    def forward(self, features):
        e = self.encoder(features)
        return e, self.heads(e) * 10


def fold(model):
    """Conv + BatchNorm -> one conv; Linear + BatchNorm1d -> one linear. Eval-mode identities."""
    model = model.eval()
    seq, out, i = list(model.features), [], 0
    while i < len(seq):
        m = seq[i]
        if isinstance(m, nn.Conv2d) and i + 1 < len(seq) and isinstance(seq[i + 1], nn.BatchNorm2d):
            out.append(nn.utils.fuse_conv_bn_eval(m, seq[i + 1])); i += 2
        else: out.append(m); i += 1
    model.features = nn.Sequential(*out)
    lin, bn, *rest = list(model.head)
    model.head = nn.Sequential(nn.utils.fuse_linear_bn_eval(lin, bn), *rest)
    return model


def load(run):
    ck = torch.load(RUNS / run / 'best.pt', map_location='cpu', weights_only=False)
    c = ck['config']
    enc = Encoder(c['bands'], c['dims'], tuple(c['widths']), c['reduce'], c['hidden']); enc.load_state_dict(ck['state'])
    heads = nn.Linear(c['dims'], 32); heads.load_state_dict(ck['heads'])
    return Graph(fold(enc), heads.eval()).eval(), ck


@torch.no_grad()
def round_int8(graph):
    """Every conv and linear weight rounded per output row to int8 in place: the PyTorch twin of quantize()."""
    for m in graph.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            w = m.weight.data; rows = w.reshape(w.shape[0], -1); s = rows.abs().amax(1).clamp(min=1e-12) / 127
            m.weight.data = (torch.round(rows / s[:, None]).clamp(-127, 127) * s[:, None]).reshape(w.shape)
    return graph


def quantize(path, out):
    """Every float initializer of rank >= 2 feeding a Conv, Gemm or MatMul becomes int8 rows (axis 0 for
    Conv and transB Gemm, the output axis for MatMul) plus a per-row scale, dequantized in-graph."""
    import onnx
    from onnx import helper, numpy_helper
    m = onnx.load(path); g = m.graph
    inits = {i.name: i for i in g.initializer}
    users = {}
    for n in g.node:
        for k, name in enumerate(n.input): users.setdefault(name, []).append((n, k))
    for name, init in list(inits.items()):
        w = numpy_helper.to_array(init)
        if w.ndim < 2 or w.dtype != np.float32: continue
        uses = users.get(name, [])
        if not uses or any(n.op_type not in ('Conv', 'Gemm', 'MatMul') for n, _ in uses): continue
        n0 = uses[0][0]
        axis = 1 if n0.op_type == 'MatMul' or (n0.op_type == 'Gemm' and not any(a.name == 'transB' and a.i for a in n0.attribute)) else 0
        moved = np.moveaxis(w, axis, 0); rows = moved.reshape(moved.shape[0], -1)
        scale = np.maximum(np.abs(rows).max(1), 1e-12) / 127
        q = np.clip(np.round(rows / scale[:, None]), -127, 127).astype(np.int8).reshape(moved.shape)
        q = np.moveaxis(q, 0, axis)
        g.initializer.remove(init)
        g.initializer.extend([numpy_helper.from_array(q, name + '_q'), numpy_helper.from_array(scale.astype(np.float32), name + '_scale'),
                              numpy_helper.from_array(np.zeros_like(scale, np.int8), name + '_zero')])
        g.node.insert(0, helper.make_node('DequantizeLinear', [name + '_q', name + '_scale', name + '_zero'], [name], axis=axis))
    onnx.checker.check_model(m)
    onnx.save(m, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--out', default=str(OUT))
    ap.add_argument('--fixtures', type=int, default=32)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    graph, ck = load(a.run)
    bands, frames = ck['features']['bands'], ck['features']['frames']
    x = torch.zeros(2, 1, bands, frames)
    onnx_path = out / 'encoder.onnx'
    torch.onnx.export(graph, (x,), str(onnx_path), input_names=['features'], output_names=['embedding', 'algorithm'],
                      dynamic_axes={'features': {0: 'n'}, 'embedding': {0: 'n'}, 'algorithm': {0: 'n'}}, opset_version=17, dynamo=False)
    import onnx
    m = onnx.load(onnx_path)
    for k, v in {'features': json.dumps(ck['features'], sort_keys=True), 'run': a.run, 'step': str(ck['step'])}.items():
        m.metadata_props.append(onnx.StringStringEntryProto(key=k, value=v))
    onnx.save(m, onnx_path)
    quantize(str(onnx_path), str(out / 'encoder.int8.onnx'))

    # fixtures: queries and references of development voices, and a catalog of those voices
    dev = Store('dev', ck['kind']); rows = dev.rows
    voices = sorted({r['voice'] for r in rows})[:64]
    refs = [i for i, r in enumerate(rows) if r['kind'] == 'ref' and r['voice'] in voices and dev.audible[i]]
    queries = [i for i, r in enumerate(rows) if r['kind'] == 'query' and r['voice'] in voices and dev.audible[i]][:a.fixtures]
    import onnxruntime as ort
    fixture = {'features': ck['features'], 'rows': queries, 'voices': voices, 'patches': base64.b64encode(dev.read(queries).tobytes()).decode()}
    with torch.no_grad():
        for name in ('encoder.onnx', 'encoder.int8.onnx'):
            # the int8 graph's PyTorch twin: the same weights rounded per row
            g = graph if name == 'encoder.onnx' else round_int8(load(a.run)[0])
            run = lambda rows: torch.cat([g(torch.from_numpy(dev.read(rows[b:b + 64])).float()[:, None])[0] for b in range(0, len(rows), 64)])
            qe, re_ = run(queries), run(refs)
            owners = torch.tensor([voices.index(rows[i]['voice']) for i in refs])
            cat = F.normalize(torch.zeros(len(voices), re_.shape[1]).index_add_(0, owners, re_), dim=1)
            top5 = (qe @ cat.T).argsort(1, descending=True)[:, :5]
            sess = ort.InferenceSession(str(out / name), providers=['CPUExecutionProvider'])
            oe = sess.run(['embedding'], {'features': dev.read(queries).astype(np.float32)[:, None]})[0]
            print(f'{name}: {(out / name).stat().st_size / 1e6:.2f} MB, python onnxruntime vs torch max |diff| {np.abs(oe - qe.numpy()).max():.2e}')
            fixture[name] = {'embeddings': qe.numpy().round(7).tolist(), 'catalog': cat.numpy().round(7).tolist(), 'top5': top5.tolist()}
    (out / 'parity.json').write_text(json.dumps(fixture))
    print('wrote', onnx_path, out / 'encoder.int8.onnx', out / 'parity.json')


if __name__ == '__main__':
    main()
