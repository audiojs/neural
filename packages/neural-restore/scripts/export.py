"""The trained net as ONNX, between the STFT and the iSTFT the JS side computes.

    python export.py runs/a/net-0400000.pt --out ~/.cache/audiojs/neural/restore [--fp16] [--verify]

  input   spec  [1, 2, 1025, T]   STFT of one channel at 48 kHz (n_fft 2048, hop 512, Hann, centered, reflect), re and im
  output  spec  [1, 2, 1025, T]   the restored STFT
The frame axis is dynamic; the model hears its input at −20 dBFS RMS (dsp.to_level), the JS side undoes the gain.
--fp16 writes a second file with float16 weights (half the download; computed in float32 by onnxruntime's
WebGPU as neural-separate's mel-roformer is). --verify runs onnxruntime on a random segment against torch.
"""
import argparse, os
import numpy as np
import torch
from model import Net, Graph, params
from dsp import BINS

def load(path, dev='cpu'):
    ck = torch.load(path, map_location=dev)
    net = Net(**ck['cfg']); net.load_state_dict(ck['net']); return net.eval()

def half(src, dst, least=1024):
    """Weights stored as float16, computed in float32: every float initializer of `least` elements or more becomes a
    float16 one and a Cast back to float where it was. Half the download, and the arithmetic stays the export's
    (onnxconverter_common's whole-graph conversion breaks on the attention's own Casts and changes the arithmetic)."""
    import onnx
    from onnx import numpy_helper, helper, TensorProto
    m = onnx.load(src); g = m.graph; casts = []
    for t in list(g.initializer):
        if t.data_type != TensorProto.FLOAT: continue
        a = numpy_helper.to_array(t)
        if a.size < least: continue
        h = numpy_helper.from_array(a.astype(np.float16), t.name + '_fp16')
        g.initializer.remove(t); g.initializer.append(h)
        casts.append(helper.make_node('Cast', [h.name], [t.name], to=TensorProto.FLOAT, name=t.name + '_cast'))
    for c in reversed(casts): g.node.insert(0, c)
    onnx.checker.check_model(m); onnx.save(m, dst)

def main():
    a = argparse.ArgumentParser()
    a.add_argument('ckpt'); a.add_argument('--out', default=os.path.join(os.environ.get('AUDIO_NEURAL_CACHE') or os.path.expanduser('~/.cache/audiojs/neural'), 'restore'))
    a.add_argument('--fp16', action='store_true'); a.add_argument('--verify', action='store_true'); a.add_argument('--frames', type=int, default=376)
    o = a.parse_args()
    os.makedirs(o.out, exist_ok=True)
    net = load(o.ckpt); g = Graph(net).eval()
    print(f'{params(net) / 1e6:.2f}M parameters')
    x = torch.randn(1, 2, BINS, o.frames) * 0.01
    path = os.path.join(o.out, 'restore.onnx')
    torch.onnx.export(g, (x,), path, input_names=['spec'], output_names=['spec_out'], dynamic_axes={'spec': {3: 'T'}, 'spec_out': {3: 'T'}}, opset_version=17, dynamo=False)
    print('wrote', path, os.path.getsize(path) // 1024, 'KB')
    if o.fp16:
        p16 = os.path.join(o.out, 'restore.fp16.onnx'); half(path, p16); print('wrote', p16, os.path.getsize(p16) // 1024, 'KB')
    if o.verify:
        import onnxruntime as ort
        for p in [path] + ([p16] if o.fp16 else []):
            s = ort.InferenceSession(p, providers=['CPUExecutionProvider'])
            for T in (o.frames, 97):
                x = torch.randn(1, 2, BINS, T) * 0.01
                with torch.no_grad(): ref = g(x).numpy()
                out = s.run(None, {'spec': x.numpy()})[0]
                err = np.abs(out - ref).max(); snr = 10 * np.log10((ref ** 2).sum() / ((out - ref) ** 2).sum())
                print(f'{os.path.basename(p)} T={T}: max |Δ| {err:.2e}, {snr:.1f} dB from torch')

if __name__ == '__main__':
    main()
