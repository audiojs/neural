#!/usr/bin/env python3
"""Export Mel-Band RoFormer (Wang, Lu, Won, Chen, Song, "Mel-Band RoFormer for Music Source Separation", ISMIR LBD
2023) to the ONNX contract @audio/neural-separate expects for modelType 'complex': the network between its STFT and
iSTFT, which JS computes, written for the GPU.

    pip install torch onnx onnxruntime pyyaml einops beartype rotary-embedding-torch librosa
    python3 export-roformer.py --model mel-roformer --verify

The model is Kimberley Jensen's vocals Mel-Band RoFormer (huggingface.co/KimberleyJSN/melbandroformer, MIT), with the
config ZFTurbo/Music-Source-Separation-Training (MIT) runs it by (configs/KimberleyJensen/
config_vocals_mel_band_roformer_kj.yaml) and that repository's models/bs_roformer code, both at a pinned commit.

One 8 s segment (352800 samples at 44.1 kHz, 801 frames) per run:
  input    mix_spec    [1, 4, 1025, 801]     STFT, complex as channels (L re, L im, R re, R im)
  output   stems_spec  [1, 4, 1025, 801]     the vocals, per channel, re and im
The STFT is the model's own: n_fft 2048, hop 441, Hann window of 2048, unnormalized, centered with reflect padding.

The graph computes what MelBandRoformer.forward computes between its STFT and iSTFT, rearranged so a GPU runs it in few
and large operations; none of it changes the function (--verify: against the PyTorch forward):
  - the 60 bands' input projections and mask MLPs run as batched matrix products, bands of near widths grouped and
    zero-padded to the widest of their group (10 groups, none wider than 1.3 times its narrowest), not 60
    small products each;
  - each RMSNorm's scale (√d·γ) is folded into the weights of the linear layer it feeds; the attention's query scale
    into the query weights; its gates' projection joins the q, k, v projection;
  - the rotary embedding's interleaved pairs become halves (the same permutation of q's and k's head features leaves
    q·k as it was), so the rotation is two products with tables of cos and sin and one with a fixed 64×64 matrix;
  - the masks of overlapping mel bands are averaged by two gathers and a weight, not a scatter, the DC bin's weight 0
    (zero_dc);
  - no tensor passes 100 MB: each transformer runs on slices of its sequences, the mask MLP by group of bands
    (onnxruntime's WebGPU Softmax went wrong past WebGPU's default 128 MiB storage binding).

Writes <out-dir>/mel-roformer.onnx (928 MB), and with --fp16 mel-roformer.fp16.onnx (float16 weights, float32 compute,
half the download). Products computed in float16 measured slower on WebGPU (3.2 against 2.6 s a segment): not offered.
Default out-dir: $AUDIO_NEURAL_CACHE/<model>, else ~/.cache/audiojs/neural/<model>. --verify compares the stems rebuilt
from onnxruntime's output with MelBandRoformer.forward on the same segment and writes the reference separation test.js
compares the JS pipeline against (scripts/reference.py, chunked as separate.js chunks: step 0.5, Kim's num_overlap 2).
Last run: torch 2.14, onnxruntime 1.31.
"""
import argparse
import gc
import math
import os
import sys
import urllib.request
from pathlib import Path

REPO = "https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/84b1eac0887756b4f1a9d7a1ff49105939749ed2/"
MODELS = {  # checkpoint URL, its SHA-256, config path in REPO
    "mel-roformer": ("https://huggingface.co/KimberleyJSN/melbandroformer/resolve/ac9b0614ab3cd7f77219e18ba494dfd93956c348/MelBandRoformer.ckpt",
                     "87201f4d31afb5bc79993230fc49446918425574db48c01c405e44f365c7559e",
                     "configs/KimberleyJensen/config_vocals_mel_band_roformer_kj.yaml"),
}
SEGMENT = 352800
STEP = 0.5


def fetch(url, path):
    if not path.exists():
        print(f"fetching {url}")
        urllib.request.urlretrieve(url, str(path) + ".part")
        os.replace(str(path) + ".part", path)
    return path


def load(src, name="mel-roformer"):
    """MelBandRoformer with its weights: the model code, config and checkpoint fetched into `src` once."""
    import hashlib
    import torch
    import yaml

    pkg = src / "roformer_code" / "models" / "bs_roformer"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg.parent / "__init__.py").touch()
    (pkg / "__init__.py").touch()
    for f in ("mel_band_roformer.py", "attend.py"):
        fetch(REPO + "models/bs_roformer/" + f, pkg / f)
    sys.path.insert(0, str(src / "roformer_code"))
    sys.dont_write_bytecode = True
    from models.bs_roformer.mel_band_roformer import MelBandRoformer

    class Loader(yaml.SafeLoader): pass  # the config's resolutions are a !!python/tuple
    Loader.add_constructor("tag:yaml.org,2002:python/tuple", lambda l, n: tuple(l.construct_sequence(n)))
    url, sha, config = MODELS[name]
    cfg = yaml.load(fetch(REPO + config, src / Path(config).name).read_text(), Loader=Loader)
    ckpt = fetch(url, src / Path(url).name)
    h = hashlib.sha256()
    with open(ckpt, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    if h.hexdigest() != sha:
        raise SystemExit(f"{ckpt}: SHA-256 {h.hexdigest()}, expected {sha}")
    model = MelBandRoformer(**cfg["model"])
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(state.get("state_dict", state) if isinstance(state, dict) else state)
    return model.eval(), cfg


def stft(model, x):
    """The model's STFT: (B, C, L) -> (B, 2C, F, T), complex as channels."""
    import torch

    B, C, L = x.shape
    z = torch.view_as_real(torch.stft(x.reshape(-1, L), **model.stft_kwargs, window=model.stft_window_fn(), return_complex=True))
    return z.permute(0, 3, 1, 2).reshape(B, 2 * C, z.shape[1], z.shape[2])


def istft(model, y, length):
    """(B, 2·S·C, F, T) -> (B, S, C, L)"""
    import torch

    B, _, Fr, T = y.shape
    y = torch.view_as_complex(y.reshape(-1, 2, Fr, T).permute(0, 2, 3, 1).contiguous())
    x = torch.istft(y, **model.stft_kwargs, window=model.stft_window_fn(), length=length)
    return x.reshape(B, -1, model.audio_channels, length)


def groups(widths, slack=1.3):
    """Contiguous runs of bands whose widest is at most `slack` times their narrowest: [(first, last + 1, widest)]"""
    out, b0 = [], 0
    for b in range(1, len(widths) + 1):
        if b == len(widths) or max(widths[b0:b + 1]) > slack * min(widths[b0:b + 1]):
            out.append((b0, b, max(widths[b0:b])))
            b0 = b
    return out


class Core:
    """MelBandRoformer.forward between its STFT and its iSTFT, for the GPU (the module docstring says how)."""

    @staticmethod
    def build(model, T):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        f64 = torch.float64
        Fq = model.num_bands_per_freq.numel()                       # 1025
        S = model.audio_channels
        fi = model.freq_indices.tolist()                            # (f·S + s) per band's pair, bands in order
        nf = model.num_freqs_per_band.tolist()
        widths = [2 * n * S for n in nf]                            # a band's input: its (f, s) pairs, re and im
        G = groups(widths)
        lin = torch.matmul
        # every tensor under WebGPU's default storage binding (128 MiB, maxStorageBufferBindingSize): onnxruntime
        # 1.30's WebGPU Softmax over a larger one came out wrong one run in two (36 dB from the CPU's at 307 MB, right
        # at 123 MB), so a transformer runs on as many sequences at a time as keep its largest, the attention's
        # scores (heads × n × n) or its feed-forward's hidden layer (n × 4·dim), under 100 MB
        BUDGET = 100e6

        def sliced(f, x):
            per = max(f.biggest(x.shape[1]), 1)
            size = max(1, int(BUDGET // per))
            return torch.cat([f(x[i:i + size]) for i in range(0, x.shape[0], size)]) if size < x.shape[0] else f(x)

        rot = lambda n, emb: emb(emb.get_seq_pos(n, device="cpu", dtype=torch.float32))  # (n, dh) interleaved

        def rms(m):
            return (m.gamma.detach().double() * m.scale)

        class Mod(nn.Module):
            def __init__(s):
                super().__init__()
                # the STFT as columns: E[t, (f·S + s)·2 + c], and a zero column past them (index Fq·S·2)
                pos = []                                            # each band's input columns, in its order
                o = 0
                for n in nf:
                    pos.append([fi[k] * 2 + c for k in range(o, o + n * S) for c in range(2)])
                    o += n * S
                zero = Fq * S * 2
                bs = model.band_split.to_features
                s.split_idx, s.split_w, s.split_b = [], nn.ParameterList(), nn.ParameterList()
                for b0, b1, W in G:
                    idx = torch.full((b1 - b0, W), zero, dtype=torch.long)
                    w = torch.zeros(b1 - b0, W, bs[0][1].out_features, dtype=f64)
                    bias = torch.zeros(b1 - b0, 1, w.shape[2], dtype=f64)
                    for j, b in enumerate(range(b0, b1)):
                        norm, lin = bs[b][0], bs[b][1]
                        idx[j, :widths[b]] = torch.tensor(pos[b])
                        w[j, :widths[b]] = (lin.weight.detach().double() * rms(norm)[None]).T
                        bias[j, 0] = lin.bias.detach().double()
                    s.register_buffer(f"si{len(s.split_idx)}", idx)
                    s.split_idx.append(f"si{len(s.split_idx)}")
                    s.split_w.append(nn.Parameter(w.float(), requires_grad=False))
                    s.split_b.append(nn.Parameter(bias.float(), requires_grad=False))
                # transformers: time then frequency, per depth
                s.blocks = nn.ModuleList()
                for time, freq in model.layers:
                    s.blocks.append(nn.ModuleList([Block(time, T), Block(freq, len(nf))]))
                # mask estimator (one stem): its MLP's linear layers before the last batched over bands (their shapes
                # alike), the last by group, then GLU
                me = model.mask_estimators[0].to_freqs
                lin = [[m for m in me[b][0] if isinstance(m, nn.Linear)] for b in range(len(nf))]
                s.mw = nn.ParameterList([nn.Parameter(torch.stack([l[i].weight.detach().T for l in lin]), requires_grad=False) for i in range(len(lin[0]) - 1)])
                s.mb = nn.ParameterList([nn.Parameter(torch.stack([l[i].bias.detach() for l in lin])[:, None], requires_grad=False) for i in range(len(lin[0]) - 1)])
                s.m2_w, s.m2_b = nn.ParameterList(), nn.ParameterList()
                col, cols = 0, {}
                for b0, b1, W in G:
                    H = lin[0][-1].in_features
                    w = torch.zeros(b1 - b0, H, 2 * W)
                    bias = torch.zeros(b1 - b0, 1, 2 * W)
                    for j, b in enumerate(range(b0, b1)):
                        l2 = lin[b][-1]
                        n = widths[b]
                        w[j, :, :n], w[j, :, W:W + n] = l2.weight.detach()[:n].T, l2.weight.detach()[n:].T
                        bias[j, 0, :n], bias[j, 0, W:W + n] = l2.bias.detach()[:n], l2.bias.detach()[n:]
                        for e in range(n):
                            cols.setdefault(pos[b][e], []).append(col + j * W + e)
                    s.m2_w.append(nn.Parameter(w, requires_grad=False))
                    s.m2_b.append(nn.Parameter(bias, requires_grad=False))
                    col += (b1 - b0) * W
                # each output column's sources among the masks (at most two bands overlap), the zero column for none,
                # and its weight: 1 over the bands that cover it, 0 at DC
                i1 = torch.full((zero,), col, dtype=torch.long)
                i2 = torch.full((zero,), col, dtype=torch.long)
                wt = torch.zeros(zero)
                for o, src in cols.items():
                    assert len(src) <= 2, "three bands over one frequency"
                    i1[o] = src[0]
                    i2[o] = src[-1] if len(src) == 2 else col
                    wt[o] = 0 if o // (S * 2) == 0 and model.zero_dc else 1 / len(src)
                s.register_buffer("i1", i1)
                s.register_buffer("i2", i2)
                s.register_buffer("wt", wt)
                s.Fq, s.S = Fq, S

            def forward(s, mix_spec):
                Fq, S = s.Fq, s.S
                T_ = mix_spec.shape[3]
                # (1, 2S, F, T) -> E (T, F·S·2): [t, (f·S + s)·2 + c]
                E = mix_spec.reshape(S, 2, Fq, T_).permute(3, 2, 0, 1).reshape(T_, Fq * S * 2)
                Ez = torch.cat([E, torch.zeros(T_, 1, dtype=E.dtype)], 1)
                xs = []
                for k, name in enumerate(s.split_idx):
                    idx = getattr(s, name)                          # (nb, W)
                    x = Ez[:, idx].permute(1, 0, 2)                 # (nb, T, W)
                    x = x / x.norm(dim=-1, keepdim=True).clamp(min=1e-12)
                    xs.append(lin(x, s.split_w[k]) + s.split_b[k])
                x = torch.cat(xs, 0)                                # (B, T, D)
                for time, freq in s.blocks:
                    x = sliced(time, x)                             # sequences over time, one per band
                    x = sliced(freq, x.transpose(0, 1)).transpose(0, 1)  # over bands, one per frame
                ms = []
                for k, (lo, hi, W) in enumerate(G):
                    h = x[lo:hi]
                    for w, b in zip(s.mw, s.mb):
                        h = torch.tanh(lin(h, w[lo:hi]) + b[lo:hi])  # (nb, T, H)
                    y = lin(h, s.m2_w[k]) + s.m2_b[k]               # (nb, T, 2W)
                    m = y[..., :W] * torch.sigmoid(y[..., W:])      # GLU
                    ms.append(m.permute(1, 0, 2).reshape(T_, (hi - lo) * W))
                M = torch.cat(ms + [torch.zeros(T_, 1)], 1)         # (T, columns + 1)
                mask = (M[:, s.i1] + M[:, s.i2]) * s.wt             # (T, F·S·2)
                e = E.reshape(T_, Fq, S, 2)
                mk = mask.reshape(T_, Fq, S, 2)
                er, ei, mr, mi = e[..., 0], e[..., 1], mk[..., 0], mk[..., 1]
                out = torch.stack([er * mr - ei * mi, er * mi + ei * mr], -1)   # (T, F, S, 2)
                return out.permute(2, 3, 1, 0).reshape(1, 2 * S, Fq, T_)

        class Block(nn.Module):
            """One Transformer of the model (its layers and output norm) over sequences of n: x (batch, n, D)."""

            def __init__(s, tr, n):
                super().__init__()
                s.layers = nn.ModuleList()
                for attn, ff in tr.layers:
                    s.layers.append(nn.ModuleList([Attn(attn, n), FF(ff)]))
                s.out = nn.Parameter(rms(tr.norm).float(), requires_grad=False)

            def biggest(s, n):
                """bytes of the largest tensor one sequence of n makes"""
                a, ff = s.layers[0]
                return 4 * max(a.h * n * n, n * ff.w1.shape[1], n * a.qkvg.shape[1])

            def forward(s, x):
                for attn, ff in s.layers:
                    x = attn(x) + x
                    x = ff(x) + x
                return x / x.norm(dim=-1, keepdim=True).clamp(min=1e-12) * s.out

        class Attn(nn.Module):
            def __init__(s, a, n):
                super().__init__()
                h, dh = a.heads, a.to_qkv.out_features // (3 * a.heads)
                g = rms(a.norm)[None]
                wq, wk, wv = (a.to_qkv.weight.detach().double() * g).reshape(3, h, dh, -1)
                # interleaved pairs -> halves, the same permutation of q's and k's features
                perm = torch.cat([torch.arange(0, dh, 2), torch.arange(1, dh, 2)])
                wq, wk = wq[:, perm] * a.scale, wk[:, perm]
                wg = a.to_gates.weight.detach().double() * g
                s.qkvg = nn.Parameter(torch.cat([wq.reshape(h * dh, -1), wk.reshape(h * dh, -1), wv.reshape(h * dh, -1), wg]).T.float(), requires_grad=False)
                s.gb = nn.Parameter(a.to_gates.bias.detach().float(), requires_grad=False)
                s.wo = nn.Parameter(a.to_out[0].weight.detach().T.float(), requires_grad=False)
                freqs = rot(n, a.rotary_embed).double()[:, perm]  # (n, dh)
                s.register_buffer("cos", freqs.cos().float())
                s.register_buffer("sin", freqs.sin().float())
                R = torch.zeros(dh, dh)                             # x @ R: (-x[half:], x[:half])
                R[torch.arange(dh // 2) + dh // 2, torch.arange(dh // 2)] = -1
                R[torch.arange(dh // 2), torch.arange(dh // 2) + dh // 2] = 1
                s.register_buffer("R", R)
                s.h, s.dh = h, dh

            def forward(s, x):
                Bt, n, _ = x.shape
                h, dh = s.h, s.dh
                x = x / x.norm(dim=-1, keepdim=True).clamp(min=1e-12)
                y = lin(x, s.qkvg)
                q, k, v = (y[..., i * h * dh:(i + 1) * h * dh].reshape(Bt, n, h, dh).transpose(1, 2) for i in range(3))
                gates = torch.sigmoid(y[..., 3 * h * dh:] + s.gb)  # (Bt, n, h)
                q = q * s.cos + torch.matmul(q, s.R) * s.sin
                k = k * s.cos + torch.matmul(k, s.R) * s.sin
                o = torch.matmul(torch.softmax(torch.matmul(q, k.transpose(-1, -2)), -1), v)  # (Bt, h, n, dh)
                o = o * gates.transpose(1, 2)[..., None]
                return lin(o.transpose(1, 2).reshape(Bt, n, h * dh), s.wo)

        class FF(nn.Module):
            def __init__(s, ff):
                super().__init__()
                norm, l1, _, _, l2, _ = ff.net
                s.w1 = nn.Parameter((l1.weight.detach().double() * rms(norm)[None]).T.float(), requires_grad=False)
                s.b1 = nn.Parameter(l1.bias.detach().float(), requires_grad=False)
                s.w2 = nn.Parameter(l2.weight.detach().T.float(), requires_grad=False)
                s.b2 = nn.Parameter(l2.bias.detach().float(), requires_grad=False)

            def forward(s, x):
                x = x / x.norm(dim=-1, keepdim=True).clamp(min=1e-12)
                return lin(F.gelu(lin(x, s.w1) + s.b1), s.w2) + s.b2

        return Mod().eval()


def frames(model):
    import torch

    return stft(model, torch.zeros(1, model.audio_channels, SEGMENT)).shape[3]


def export(model, path, opset):
    import torch

    T = frames(model)
    core = Core.build(model, T)
    spec = stft(model, torch.randn(1, model.audio_channels, SEGMENT))
    with torch.no_grad():
        torch.onnx.export(core, (spec,), str(path), input_names=["mix_spec"], output_names=["stems_spec"],
                          opset_version=opset, dynamo=False)
    print(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB), segment {SEGMENT} samples, spec {list(spec.shape)}, groups {groups([2 * n * model.audio_channels for n in model.num_freqs_per_band.tolist()])}")
    return core


def engine(model, out_dir, name):
    """The GPU form's tensors for the package's own WebGPU engine (roformer.js): <name>.engine.bin, each tensor in turn
    (weights float16, tables float32, indices int32), and <name>.engine.json naming them, their type, shape and offset,
    with the band groups and the sizes the engine reads."""
    import json
    import numpy as np
    import torch

    T = frames(model)
    core = Core.build(model, T)
    tensors, blob, off = {}, bytearray(), 0
    for n, t in list(core.named_parameters()) + list(core.named_buffers()):
        a = t.detach().numpy()
        kind = "int32" if a.dtype.kind in "iu" else "float16" if n.rsplit(".", 1)[-1] in ("qkvg", "wo", "w1", "w2") or n.startswith(("split_w.", "mw.", "m2_w.")) else "float32"
        b = a.astype(kind).tobytes()
        pad = -len(blob) % 16
        blob += bytes(pad)
        tensors[n] = {"type": kind, "shape": list(a.shape), "offset": len(blob)}
        blob += b
    nf = model.num_freqs_per_band.tolist()
    meta = {
        "model": name, "frames": T, "freqs": core.Fq, "channels": core.S, "dim": model.layers[0][0].layers[0][0].to_qkv.in_features,
        "heads": model.layers[0][0].layers[0][0].heads, "depth": len(model.layers), "bands": len(nf),
        "groups": [list(g) for g in groups([2 * n * core.S for n in nf])], "widths": [2 * n * core.S for n in nf],
        "tensors": tensors,
    }
    (out_dir / f"{name}.engine.bin").write_bytes(bytes(blob))
    (out_dir / f"{name}.engine.json").write_text(json.dumps(meta))
    print(f"wrote {out_dir / (name + '.engine.bin')} ({len(blob) / 1e6:.1f} MB) and its .json, {len(tensors)} tensors")


def fp16(path, out):
    """float16 weights, float32 compute (scripts/compact.py --as fp16)."""
    sys.path.insert(0, str(Path(__file__).parent))
    import compact

    compact.compact(path, out, "fp16")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


def session(path):
    import onnxruntime as ort

    o = ort.SessionOptions()
    o.enable_cpu_mem_arena = False
    o.enable_mem_pattern = False
    return ort.InferenceSession(str(path), o, providers=["CPUExecutionProvider"])


def mixes():
    import torch

    t = torch.arange(SEGMENT) / 44100
    tone = sum(torch.sin(2 * math.pi * f * t) / k for k, f in enumerate((110, 220, 440, 880, 1760), 1))
    return (("noise", 0.1 * torch.randn(1, 2, SEGMENT)), ("tones", 0.2 * torch.stack([tone, 0.8 * tone])[None]))


def verify(model, run, label, tol):
    """The vocals rebuilt from run(spec) against MelBandRoformer.forward: noise and tones."""
    import numpy as np
    import torch

    for name, mix in mixes():
        with torch.no_grad():
            want = model(mix).numpy()
            got = istft(model, torch.as_tensor(run(stft(model, mix))), SEGMENT)[:, 0].numpy()
        want = want[..., :got.shape[-1]]
        rel = np.abs(got - want).max() / max(np.abs(want).max(), 1e-12)
        print(f"  verify {label} {name}: max|diff|/max|y| = {rel:.2e} " + ("OK" if rel < tol else "FAIL"))
        if rel >= tol:
            raise SystemExit(f"{label}: verification failed ({rel:.2e} >= {tol})")


def reference(model, out_dir):
    """The reference separation: MelBandRoformer.forward on segments as scripts/reference.py's chunked() cuts them
    (step 0.5, no standardization: Kim's config normalizes nothing), over a 20 s mix with a noise floor."""
    import torch

    sys.path.insert(0, str(Path(__file__).parent))
    import reference as ref

    def run(seg):
        with torch.no_grad():
            return model(torch.from_numpy(seg)[None])[0][None].numpy()   # (1 stem, C, L)

    # under the mix a noise floor 60 dB down: each band normalizes its input, and a silent band's is its rounding
    ref.write(out_dir, lambda mix: {"vocals": ref.chunked(mix, run, SEGMENT, STEP)[0]}, 20.0, 1e-3)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="mel-roformer", choices=list(MODELS))
    ap.add_argument("--out-dir", default=None, help="default: $AUDIO_NEURAL_CACHE/<model> or ~/.cache/audiojs/neural/<model>")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--fp16", action="store_true", help="also write <model>.fp16.onnx (float16 weights, float32 compute)")
    ap.add_argument("--verify", action="store_true", help="compare with MelBandRoformer.forward; write the reference separation")
    ap.add_argument("--engine", action="store_true", help="also write <model>.engine.bin/.json: the tensors for roformer.js, the package's WebGPU engine")
    args = ap.parse_args()

    cache = Path(os.environ.get("AUDIO_NEURAL_CACHE") or Path.home() / ".cache" / "audiojs" / "neural")
    out_dir = Path(args.out_dir) if args.out_dir else cache / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    src = out_dir / "src"
    src.mkdir(exist_ok=True)
    try:
        model, _ = load(src, args.model)
    except ImportError as e:
        raise SystemExit(f"required: pip install torch onnx onnxruntime pyyaml einops beartype rotary-embedding-torch librosa ({e})")
    path = out_dir / f"{args.model}.onnx"
    core = export(model, path, args.opset)
    if args.verify:
        import torch

        with torch.no_grad():
            verify(model, lambda spec: core(spec), "the GPU form in torch", 1e-4)
        # onnxruntime against torch: on the tones 1.7e-4, where torch's own float32 stands 4.6e-3 from its float64
        s = session(path)
        verify(model, lambda spec: s.run(None, {"mix_spec": spec.numpy()})[0], path.name, 1e-3)
        del s
        gc.collect()
        reference(model, out_dir)
    if args.fp16:
        fp16(path, out_dir / f"{args.model}.fp16.onnx")
        if args.verify:
            s = session(out_dir / f"{args.model}.fp16.onnx")
            verify(model, lambda spec: s.run(None, {"mix_spec": spec.numpy()})[0], f"{args.model}.fp16.onnx", 2e-2)
    if args.engine:
        engine(model, out_dir, args.model)
    print(f"done: {args.model} in {out_dir}")


if __name__ == "__main__":
    sys.exit(main())
