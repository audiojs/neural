#!/usr/bin/env python3
"""Export TIGER's Divide and Remaster model (Xu, Li, Chen, Hu, "TIGER: Time-frequency Interleaved Gain Extraction and
Reconstruction for Efficient Speech Separation", ICLR 2025) to the ONNX contract @audio/neural-separate expects for
modelType 'complex': the networks between their STFT and iSTFT, which JS computes.

    pip install torch onnx onnxruntime safetensors
    python3 export-tiger.py --verify

TIGERDNR is three band-split TIGER models of 1.4 M parameters, each separating three sources, of which it keeps one:
the dialogue of the first, the effects of the second, the music of the third. The weights are the authors'
(huggingface.co/JusperLee/TIGER-DnR, whose model card states license: apache-2.0; the code, JusperLee/TIGER, is MIT),
fetched at a pinned revision with the model code at a pinned commit. Its paper, DnR test (v1), SI-SDR: music 7.4,
speech 15.5, effects 6.5 dB.

One mono 12 s segment (529200 samples at 44.1 kHz, 1034 frames) per run, the three models in one graph:
  input    mix_spec    [1, 2, 1025, 1034]    STFT, re and im as channels
  output   stems_spec  [1, 6, 1025, 1034]    dialogue, music, effects, each re and im
The STFT is TIGER's own: n_fft 2048, hop 512, Hann, unnormalized, centered with reflect padding. Three changes to the
graph, none to its function: F.adaptive_avg_pool1d (the exporter takes only output sizes that divide the input's)
becomes a cumulative sum read at torch's windows, UConvBlock's global sum starts at its first term instead of a
zeros tensor (exported, a constant as large as the features), and each GroupNorm(1, C) reduces its statistics one
axis at a time (export-scnet.py's; exported whole, each was an InstanceNormalization with constants as large as its
batch).

Writes <out-dir>/tiger.onnx (29 MB), weights embedded. Default out-dir: $AUDIO_NEURAL_CACHE/tiger, else
~/.cache/audiojs/neural/tiger. --verify compares the graph with the three TIGER.forward on one segment (in PyTorch on a
CPU, minutes), and writes the reference separation test.js compares the JS pipeline against: upstream's
TIGERDNR.wav_chunk_inference (12 s segments every 4 s, unweighted, zero-padded) over reference.py's 20 s mix, each
channel apart, its segments run through the verified graph. Last run: torch 2.14, onnxruntime 1.30.
"""
import argparse
import os
import sys
import types
import importlib.util
import json
import urllib.request
from pathlib import Path

COMMIT = "9f18d4a10a7137e1ce8052cfb62215179f1287b6"
REPO = f"https://raw.githubusercontent.com/JusperLee/TIGER/{COMMIT}/look2hear/"
REVISION = "b7a59560bbca10febbcd46fb01600f868e587f57"
HF = f"https://huggingface.co/JusperLee/TIGER-DnR/resolve/{REVISION}/"
RATE, N_FFT, HOP, SEGMENT = 44100, 2048, 512, 529200
# TIGERDNR.forward: each model's kept source, in the order the graph gives them
KEEP = (("dialog", 2), ("music", 0), ("effect", 1))


def fetch(url, path):
    if not path.exists():
        print(f"fetching {url}")
        urllib.request.urlretrieve(url, str(path) + ".part")
        os.replace(str(path) + ".part", path)
    return path


def module(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    sys.modules[name] = m
    s.loader.exec_module(m)
    return m


def load(src):
    """TIGERDNR with its weights: tiger_dnr.py and the two layer modules it reads, without the package's training stack
    (its layers/__init__ and base_model import lightning, rich and huggingface_hub)."""
    import torch
    from safetensors.torch import load_file

    for f in ("models/tiger_dnr.py", "layers/activations.py", "layers/normalizations.py"):
        fetch(REPO + f, src / Path(f).name)
    for f in ("config.json", "model.safetensors"):
        fetch(HF + f, src / f)
    for p in ("look2hear", "look2hear.layers", "look2hear.models"):
        sys.modules[p] = types.ModuleType(p)
        sys.modules[p].__path__ = []
    sys.dont_write_bytecode = True
    layers = sys.modules["look2hear.layers"]
    layers.activations = module("look2hear.layers.activations", src / "activations.py")
    layers.normalizations = module("look2hear.layers.normalizations", src / "normalizations.py")
    base = types.ModuleType("look2hear.models.base_model")

    class BaseModel(torch.nn.Module):
        def __init__(self, sample_rate, in_chan=1):
            super().__init__()

    base.BaseModel = BaseModel
    sys.modules["look2hear.models.base_model"] = base
    tiger = module("look2hear.models.tiger_dnr", src / "tiger_dnr.py")
    model = tiger.TIGERDNR(**json.loads((src / "config.json").read_text()))
    model.load_state_dict(load_file(str(src / "model.safetensors")))
    return model.eval(), tiger


def aap(x, output_size):
    """F.adaptive_avg_pool1d, windows floor(i L / n) to ceil((i + 1) L / n) as torch's, by a cumulative sum read at the
    windows' ends"""
    import torch
    import torch.nn.functional as F

    L, n = int(x.shape[-1]), int(output_size[0] if isinstance(output_size, (list, tuple)) else output_size)
    a = torch.tensor([(i * L) // n for i in range(n)])
    b = torch.tensor([-((-(i + 1) * L) // n) for i in range(n)])
    c = F.pad(torch.cumsum(x, -1), (1, 0))
    return (c.index_select(-1, b) - c.index_select(-1, a)) / (b - a).to(x.dtype)


def uconv(self, x):
    """UConvBlock.forward, its global sum started at the first pooled level (0 + a is a)"""
    residual = x
    output = [self.spp_dw[0](self.proj_1x1(x))]
    for k in range(1, self.depth):
        output.append(self.spp_dw[k](output[-1]))
    g = None
    for fea in output:
        p = aap(fea, output[-1].shape[-1])
        g = p if g is None else g + p
    g = self.globalatt(g)
    fused = [self.loc_glo_fus[i](output[i], g) for i in range(self.depth)]
    expanded = None
    for i in range(self.depth - 2, -1, -1):
        expanded = self.last_layer[i](fused[i], fused[i - 1] if i == self.depth - 2 else expanded)
    return self.res_conv(expanded) + residual


def group_norm(mod):
    """GroupNorm(1, C), its mean and variance reduced one axis at a time"""
    import torch

    def mean(x):
        for d in range(x.dim() - 1, 0, -1):
            x = x.mean(d, keepdim=True)
        return x

    def forward(x):
        mu = mean(x)
        var = mean((x - mu) ** 2)
        shape = (1, -1) + (1,) * (x.dim() - 2)
        return (x - mu) / torch.sqrt(var + mod.eps) * mod.weight.view(shape) + mod.bias.view(shape)

    return forward


def stft(x):
    """TIGER.forward's STFT: (B, L) -> (B, 2, F, T)"""
    import torch

    z = torch.stft(x, N_FFT, HOP, window=torch.hann_window(N_FFT), return_complex=True)
    return torch.stack([z.real, z.imag], 1)


def istft(y, length):
    """(B, 2·S, F, T) -> (B, S, length), as TIGER.forward's iSTFT"""
    import torch

    B, C, F, T = y.shape
    z = torch.complex(y[:, 0::2], y[:, 1::2]).reshape(-1, F, T)
    return torch.istft(z, N_FFT, HOP, window=torch.hann_window(N_FFT), length=length).reshape(B, C // 2, length)


class Core:
    """The three TIGER.forward between their STFT and iSTFT, each its kept source."""

    @staticmethod
    def build(model, tiger):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        # the patches, on the module the model's code reads them from
        tiger.F = types.SimpleNamespace(**{k: getattr(F, k) for k in dir(F) if not k.startswith("_")}, )
        tiger.F.adaptive_avg_pool1d = aap
        tiger.UConvBlock.forward = uconv
        for m in model.modules():
            if isinstance(m, nn.GroupNorm) and m.num_groups == 1:
                m.forward = group_norm(m)

        def one(t, src, spec):
            B, _, _, T = spec.shape
            re, im = spec[:, 0], spec[:, 1]
            feats, bands, i0 = [], [], 0
            for i, bw in enumerate(t.band_width):
                bw = int(bw)
                bands.append((i0, bw))
                feats.append(t.BN[i](spec[:, :, i0:i0 + bw].reshape(B, bw * 2, T)))
                i0 += bw
            x = t.separator(torch.stack(feats, 1)).view(B, t.nband, t.feature_dim, T)
            outr, outi = [], []
            for i, (i0, bw) in enumerate(bands):
                o = t.mask[i](x[:, i]).view(B, 2, 2, t.num_output, bw, T)
                mk = o[:, 0] * torch.sigmoid(o[:, 1])
                mr, mi = mk[:, 0], mk[:, 1]
                mr = (mr - (mr.sum(1, keepdim=True) - 1) / t.num_output)[:, src]
                mi = (mi - mi.sum(1, keepdim=True) / t.num_output)[:, src]
                r, j = re[:, i0:i0 + bw], im[:, i0:i0 + bw]
                outr.append(r * mr - j * mi)
                outi.append(r * mi + j * mr)
            return torch.stack([torch.cat(outr, 1), torch.cat(outi, 1)], 1)

        class C(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, spec):
                return torch.cat([one(getattr(self.m, name), src, spec) for name, src in KEEP], 1)

        return C(model).eval()


def export(core, path, opset):
    import torch

    spec = stft(0.1 * torch.randn(1, SEGMENT))
    with torch.no_grad():
        torch.onnx.export(core, (spec,), str(path), input_names=["mix_spec"], output_names=["stems_spec"],
                          opset_version=opset, dynamo=False)
    print(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB), segment {SEGMENT} samples, spec {list(spec.shape)}")


def session(path):
    import onnxruntime as ort

    o = ort.SessionOptions()
    o.enable_cpu_mem_arena = False
    o.enable_mem_pattern = False
    return ort.InferenceSession(str(path), o, providers=["CPUExecutionProvider"])


def verify(model, path, tol):
    """One segment: the graph's sources through the iSTFT against each TIGER.forward's kept source (unpatched)."""
    import math
    import numpy as np
    import torch

    s = session(path)
    t = torch.arange(SEGMENT) / RATE
    x = (0.2 * sum(torch.sin(2 * math.pi * f * t) / k for k, f in enumerate((110, 220, 440, 880), 1)) + 0.05 * torch.randn(SEGMENT))[None]
    got = istft(torch.from_numpy(s.run(None, {"mix_spec": stft(x).numpy()})[0]), SEGMENT).numpy()
    with torch.no_grad():
        want = np.stack([getattr(model, name)(x)[:, src].numpy() for name, src in KEEP], 1)
    rel = np.abs(got - want).max() / max(np.abs(want).max(), 1e-12)
    print(f"  verify {path.name}: max|diff|/max|y| = {rel:.2e} " + ("OK" if rel < tol else "FAIL"))
    if rel >= tol:
        raise SystemExit(f"{path}: verification failed ({rel:.2e} >= {tol})")
    return s


def reference(model, s, out_dir):
    """upstream's wav_chunk_inference over reference.py's 20 s mix, each channel apart, its segments through the graph"""
    import torch

    sys.path.insert(0, str(Path(__file__).parent))
    import reference as ref

    def graph(k):
        # TIGER.forward's contract for wav_chunk_inference: (B, L) -> (B, 3, L), the kept source at its index
        def run(seg):
            y = istft(torch.from_numpy(s.run(None, {"mix_spec": stft(seg.reshape(-1, seg.shape[-1])).numpy()})[0]), seg.shape[-1])
            out = torch.zeros(y.shape[0], 3, y.shape[-1])
            out[:, KEEP[k][1]] = y[:, k]
            return out
        return run

    def run(mix):
        with torch.no_grad():
            x = torch.from_numpy(mix)[:, None]  # each channel a batch of one
            out = {}
            for k, name in enumerate(("dialogue", "music", "effects")):
                y = torch.stack([model.wav_chunk_inference(graph(k), x[c:c + 1])[KEEP[k][1], 0] for c in range(len(mix))])
                out[name] = y.numpy()
            return out

    ref.write(out_dir, run, 20.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=None, help="default: $AUDIO_NEURAL_CACHE/tiger or ~/.cache/audiojs/neural/tiger")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--verify", action="store_true", help="compare with TIGER.forward; write the reference separation")
    args = ap.parse_args()

    cache = Path(os.environ.get("AUDIO_NEURAL_CACHE") or Path.home() / ".cache" / "audiojs" / "neural")
    out_dir = Path(args.out_dir) if args.out_dir else cache / "tiger"
    out_dir.mkdir(parents=True, exist_ok=True)
    src = out_dir / "src"
    src.mkdir(exist_ok=True)
    try:
        model, tiger = load(src)
        ref_model, _ = load(src) if args.verify else (None, None)
    except ImportError as e:
        raise SystemExit(f"torch and safetensors are required: pip install torch onnx onnxruntime safetensors ({e})")
    path = out_dir / "tiger.onnx"
    export(Core.build(model, tiger), path, args.opset)
    if args.verify:
        s = verify(ref_model, path, 1e-4)
        reference(ref_model, s, out_dir)
    print(f"done: tiger in {out_dir}")


if __name__ == "__main__":
    sys.exit(main())
