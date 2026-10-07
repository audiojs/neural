#!/usr/bin/env python3
"""Export SCNet (Tong et al., "SCNet: Sparse Compression Network for Music Source
Separation", ICASSP 2024) to the ONNX contract @audio/neural-separate expects for
modelType 'complex': the network without its STFT and iSTFT, which JS computes.

    pip install torch onnx onnxruntime pyyaml
    python3 export-scnet.py --model scnet-large --verify
    python3 export-scnet.py --model scnet --verify

The models are SCNet-large and SCNet as their author trained them on MUSDB18-HQ
(starrytong/SCNet, MIT; the author confirms the weights are MIT too and may be
redistributed converted, ONNX included: github.com/starrytong/SCNet/issues/35), in
the layout of ZFTurbo/Music-Source-Separation-Training (MIT), whose releases host
them with their configs (v1.0.9 SCNet-large_starrytong_fixed.ckpt, 41.2 M
parameters; v.1.0.6 scnet_checkpoint_musdb18.ckpt, 10.1 M); the model code is that
repository's models/scnet, fetched at a pinned commit. MUSDB18 test, its own
measure: SCNet-large vocals 10.94, drums 11.15, bass 9.38, other 7.31 dB SDR; SCNet
9.90, 10.44, 8.89, 6.89.

One 11 s segment (485100 samples at 44.1 kHz, padded as SCNet.forward pads it to
486400, 476 frames) per run:
  input    mix_spec    [1, 4, 2049, 476]     STFT, complex as channels (L re, L im, R re, R im)
  output   stems_spec  [1, 16, 2049, 476]    per source (drums, bass, other, vocals) and channel, re and im
The STFT is SCNet's own: n_fft 4096, hop 1024, no window (torch.stft's default,
ones), normalized (1/sqrt(4096)), centered with reflect padding.

Writes <out-dir>/<model>.onnx (scnet-large 169 MB, scnet 43 MB) and, with --fp16,
<model>.fp16.onnx (half that: float16 weights, float32 compute, scripts/compact.py's), weights embedded. Default out-dir:
$AUDIO_NEURAL_CACHE/<model>, else ~/.cache/audiojs/neural/<model>. --verify compares the stems rebuilt from the ONNX
output with SCNet.forward on the same segment, and writes the reference
separation test.js compares the JS pipeline against (scripts/reference.py),
chunked as scripts/reference.py's `chunked` does. Last run: torch 2.14,
onnxruntime 1.30.
"""
import argparse
import gc
import math
import os
import sys
import urllib.request
from pathlib import Path

REPO = "https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/84b1eac0887756b4f1a9d7a1ff49105939749ed2/"
RELEASE = "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/"
MODELS = {  # release, checkpoint, config
    "scnet-large": ("v1.0.9", "SCNet-large_starrytong_fixed.ckpt", "config_musdb18_scnet_large_starrytong.yaml"),
    "scnet": ("v.1.0.6", "scnet_checkpoint_musdb18.ckpt", "config_musdb18_scnet.yaml"),
}
SEGMENT = 485100


def fetch(url, path):
    if not path.exists():
        print(f"fetching {url}")
        urllib.request.urlretrieve(url, str(path) + ".part")
        os.replace(str(path) + ".part", path)
    return path


def load(src, name="scnet-large"):
    """SCNet or SCNet-large with its weights: the model code and checkpoint fetched into `src` once."""
    import torch
    import yaml

    pkg = src / "scnet_code" / "scnet"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").touch()
    for f in ("scnet.py", "separation.py"):
        fetch(REPO + "models/scnet/" + f, pkg / f)
    sys.path.insert(0, str(src / "scnet_code"))
    sys.dont_write_bytecode = True
    from scnet.scnet import SCNet

    class Loader(yaml.SafeLoader): pass  # the config's augmentation settings hold a !!python/tuple
    Loader.add_constructor("tag:yaml.org,2002:python/tuple", lambda l, n: tuple(l.construct_sequence(n)))
    release, ckpt, config = MODELS[name]
    cfg = yaml.load(fetch(f"{RELEASE}{release}/{config}", src / config).read_text(), Loader=Loader)
    model = SCNet(**cfg["model"])
    model.load_state_dict(torch.load(fetch(f"{RELEASE}{release}/{ckpt}", src / ckpt), map_location="cpu", weights_only=True))
    return model.eval()


def stft(model, x):
    """SCNet.forward's padding and STFT: (B, C, L) -> (B, 2C, F, T), and the padding."""
    import torch
    import torch.nn.functional as F

    hop = model.hop_length
    pad = hop - x.shape[-1] % hop
    if (x.shape[-1] + pad) // hop % 2 == 0:
        pad += hop
    x = F.pad(x, (0, pad))
    B, C, L = x.shape
    z = torch.view_as_real(torch.stft(x.reshape(-1, L), **model.stft_config, return_complex=True))
    return z.permute(0, 3, 1, 2).reshape(B, 2 * C, z.shape[1], z.shape[2]), pad


def istft(model, y, pad):
    """SCNet.forward's tail: (B, 2·S·C, F, T) -> (B, S, C, L)."""
    import torch

    B, _, Fr, T = y.shape
    y = torch.view_as_complex(y.reshape(-1, 2, Fr, T).permute(0, 2, 3, 1).contiguous())
    x = torch.istft(y, **model.stft_config)
    return x.reshape(B, len(model.sources), model.audio_channels, -1)[..., :-pad]


def dft(x, inverse):
    """FeatureConversion's rfft and irfft over time (norm "ortho") as products with cos and sin matrices: no ONNX
    exporter of torch 2.14 takes aten::fft_rfft, and a MatMul runs on every backend. x: (B, C, F, T), T even."""
    import torch

    T = x.shape[3] if not inverse else 2 * (x.shape[3] - 1)
    t, k = torch.arange(T, dtype=torch.float64), torch.arange(T // 2 + 1, dtype=torch.float64)
    a = 2 * math.pi * k[:, None] * t[None] / T                     # (K, T)
    if not inverse:
        c, s = (torch.cos(a).T / math.sqrt(T)).to(x.dtype), (-torch.sin(a).T / math.sqrt(T)).to(x.dtype)
        return torch.cat([x @ c, x @ s], dim=1)                     # (re, im) on channels, as torch.cat([x.real, x.imag], 1)
    w = torch.full((T // 2 + 1, 1), 2.0, dtype=torch.float64)
    w[0] = w[-1] = 1                                                # irfft: DC and Nyquist once, their imaginary parts unread
    c, s = (w * torch.cos(a) / math.sqrt(T)).to(x.dtype), (-w * torch.sin(a) / math.sqrt(T)).to(x.dtype)
    h = x.shape[1] // 2
    return x[:, :h] @ c + x[:, h:] @ s


def group_norm(mod):
    """GroupNorm(1, C) with its mean and variance reduced one axis at a time, the last first. Exported as it is,
    onnxruntime 1.30 sums a whole (C, F, T) group in float32 (10 million values in the dual-path layers): the stems
    came out 52 to 55 dB from torch's; reduced by axis, 110 dB, as torch's own float32 against float64."""
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


class Core:
    """SCNet.forward between its STFT and its iSTFT."""

    @staticmethod
    def build(model):
        import torch.nn as nn
        from collections import deque

        for f in model.separation_net.feature_conversion:
            f.forward = (lambda inv: lambda x: dft(x, inv))(f.inverse)
        for m in model.modules():
            if isinstance(m, nn.GroupNorm) and m.num_groups == 1:
                m.forward = group_norm(m)

        class C(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, x):
                m, skips = self.m, deque()
                for sd in m.encoder:
                    x, skip, lengths, orig = sd(x)
                    skips.append((skip, lengths, orig))
                x = m.separation_net(x)
                for fusion, su in m.decoder:
                    skip, lengths, orig = skips.pop()
                    x = su(fusion(x, skip), lengths, orig)
                return x

        return C(model).eval()


def export(model, path, opset):
    import torch

    spec, _ = stft(model, torch.randn(1, model.audio_channels, SEGMENT))
    with torch.no_grad():
        torch.onnx.export(Core.build(model), (spec,), str(path), input_names=["mix_spec"], output_names=["stems_spec"],
                          opset_version=opset, dynamo=False)
    print(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB), segment {SEGMENT} samples, spec {list(spec.shape)}")


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


def verify(model, path, tol):
    """Stems rebuilt from onnxruntime's output against SCNet.forward: noise and tones."""
    import numpy as np
    import torch

    s = session(path)
    t = torch.arange(SEGMENT) / 44100
    tone = sum(torch.sin(2 * math.pi * f * t) / k for k, f in enumerate((110, 220, 440, 880, 1760), 1))
    for name, mix in (("noise", 0.1 * torch.randn(1, 2, SEGMENT)), ("tones", 0.2 * torch.stack([tone, 0.8 * tone])[None])):
        with torch.no_grad():
            want = model(mix).numpy()
            spec, pad = stft(model, mix)
            got = istft(model, torch.from_numpy(s.run(None, {"mix_spec": spec.numpy()})[0]), pad).numpy()
        rel = np.abs(got - want).max() / max(np.abs(want).max(), 1e-12)
        print(f"  verify {path.name} {name}: max|diff|/max|y| = {rel:.2e} " + ("OK" if rel < tol else "FAIL"))
        if rel >= tol:
            raise SystemExit(f"{path}: verification failed ({rel:.2e} >= {tol})")
    del s
    gc.collect()


def reference(model, out_dir):
    """The reference separation: SCNet.forward on segments as scripts/reference.py's normalized() and chunked() cut
    them, over a 20 s mix (longer than two segments less a step: its ends reflected out)."""
    import torch

    sys.path.insert(0, str(Path(__file__).parent))
    import reference as ref

    def run(seg):
        with torch.no_grad():
            return model(torch.from_numpy(seg)[None])[0].numpy()

    ref.write(out_dir, lambda mix: dict(zip(model.sources, ref.normalized(mix, lambda x: ref.chunked(x, run, SEGMENT)))), 20.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="scnet-large", choices=list(MODELS))
    ap.add_argument("--out-dir", default=None, help="default: $AUDIO_NEURAL_CACHE/<model> or ~/.cache/audiojs/neural/<model>")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--fp16", action="store_true", help="also write <model>.fp16.onnx (float16 weights, float32 compute)")
    ap.add_argument("--verify", action="store_true", help="compare with SCNet.forward; write the reference separation")
    args = ap.parse_args()

    cache = Path(os.environ.get("AUDIO_NEURAL_CACHE") or Path.home() / ".cache" / "audiojs" / "neural")
    out_dir = Path(args.out_dir) if args.out_dir else cache / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    src = out_dir / "src"
    src.mkdir(exist_ok=True)
    try:
        model = load(src, args.model)
    except ImportError as e:
        raise SystemExit(f"torch and pyyaml are required: pip install torch onnx onnxruntime pyyaml ({e})")
    path = out_dir / f"{args.model}.onnx"
    export(model, path, args.opset)
    if args.verify:
        verify(model, path, 1e-4)
        reference(model, out_dir)
    if args.fp16:
        fp16(path, out_dir / f"{args.model}.fp16.onnx")
        if args.verify:
            verify(model, out_dir / f"{args.model}.fp16.onnx", 2e-2)
    print(f"done: {args.model} in {out_dir}")


if __name__ == "__main__":
    sys.exit(main())
