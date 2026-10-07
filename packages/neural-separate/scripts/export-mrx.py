#!/usr/bin/env python3
"""Export MRX (Petermann, Wichern, Wang, Le Roux, "The Cocktail Fork Problem: Three-Stem Audio Separation for
Real-World Soundtracks", ICASSP 2022) to the ONNX contract @audio/neural-separate expects for modelType
'multires': the network between its three STFTs and their inverses, which JS computes.

    pip install torch onnx onnxruntime pyloudnorm
    python3 export-mrx.py --verify

The model is MERL's own, trained on Divide and Remaster with the SNR loss so its stems keep the mixture's level
(merlresearch/cocktail-fork-separation, checkpoints/default_mrx_pre_trained_weights.pth, 30.5 M parameters; the
repository's .reuse/dep5 licenses the four checkpoints MIT, Copyright 2023 Mitsubishi Electric Research Laboratories);
its code and checkpoint are fetched at a pinned commit. Its README, DnR test, average SI-SDR: speech 12.5, music 4.2,
sound effects 5.7 dB (the mixture 1.0, -6.8, -5.0).

The graph sees each channel's magnitude spectrogram at three resolutions, scaled as torch.stft(normalized=True)
scales it (1/sqrt(n)): Hann windows of 1024, 2048 and 8192 samples, hop 256, centered with reflect padding, at
44.1 kHz. B (channels, encoded apart) and T (frames) are free:
  inputs   mag_1024 [B, 513, T]   mag_2048 [B, 1025, T]   mag_8192 [B, 4097, T]
  outputs  mask_1024 [B, 3, 513, T]   mask_2048 [B, 3, 1025, T]   mask_8192 [B, 3, 4097, T]
the sources in MRX's order: music, speech, sfx. A source is the sum over resolutions of the iSTFT of its mask times
that resolution's spectrogram (the scale 1/sqrt(n) and its inverse cancel: the masks are real).

Writes <out-dir>/mrx.onnx (122 MB), weights embedded. Default out-dir: $AUDIO_NEURAL_CACHE/mrx, else
~/.cache/audiojs/neural/mrx. --verify compares the sources rebuilt from the ONNX output with MRX.forward, and writes
the reference separation test.js compares the JS pipeline against: separate.separate_soundtrack, as upstream's
separate.py runs it (the input at -27 LUFS by pyloudnorm, the stems scaled back), on scripts/reference.py's mix.
Last run: torch 2.14, onnxruntime 1.30.
"""
import argparse
import os
import sys
import urllib.request
from pathlib import Path

COMMIT = "19b3de827ebc4bfb014570cf92dd32b4ee3b6921"
REPO = f"https://raw.githubusercontent.com/merlresearch/cocktail-fork-separation/{COMMIT}/"
LFS = f"https://media.githubusercontent.com/media/merlresearch/cocktail-fork-separation/{COMMIT}/"
CKPT = "checkpoints/default_mrx_pre_trained_weights.pth"
WINDOWS, HOP, RATE = (1024, 2048, 8192), 256, 44100


def fetch(url, path):
    if not path.exists():
        print(f"fetching {url}")
        urllib.request.urlretrieve(url, str(path) + ".part")
        os.replace(str(path) + ".part", path)
    return path


def load(src):
    """MRX with its weights, and upstream's separate module: code and checkpoint fetched into `src` once."""
    import torch

    for f in ("mrx.py", "separate.py", "consistency.py", "dnr_dataset.py"):
        fetch(REPO + f, src / f)
    sys.path.insert(0, str(src))
    sys.dont_write_bytecode = True
    from mrx import MRX

    model = MRX().eval()
    model.load_state_dict(torch.load(fetch(LFS + CKPT, src / Path(CKPT).name), map_location="cpu", weights_only=True))
    return model


def mags(x):
    """x (B, N) -> the three normalized magnitude spectrograms (B, F, T) and complex spectrograms MRX._stft makes"""
    import mrx

    specs = [mrx._stft(x, n, HOP) for n in WINDOWS]
    return [s.abs() for s in specs], specs


class Core:
    """MRX.forward between its STFTs and its iSTFTs: magnitudes in, masks out."""

    @staticmethod
    def build(model):
        import torch
        import torch.nn as nn

        class C(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, m1, m2, m3):
                m = self.m
                enc = []
                for mag, layer in zip((m1, m2, m3), m._encoders):
                    B, F, T = mag.shape
                    e = layer.layer(mag.transpose(1, 2).reshape(-1, F))           # _EncoderBlock, its abs() done
                    enc.append(torch.tanh(e.reshape(B, T, -1)))
                cross = m._cross_net(enc)
                out = []
                for r, mag in enumerate((m1, m2, m3)):
                    B, F, T = mag.shape
                    masks = [dec[r](cross).transpose(-1, -2) for dec in m._decoders]  # (B, F, T) per source
                    out.append(torch.stack(masks, 1))
                return tuple(out)

        return C(model).eval()


def export(model, path, opset):
    import torch

    m, _ = mags(0.1 * torch.randn(2, RATE * 4))
    names = [f"mag_{n}" for n in WINDOWS]
    outs = [f"mask_{n}" for n in WINDOWS]
    axes = {**{k: {0: "B", 2: "T"} for k in names}, **{k: {0: "B", 3: "T"} for k in outs}}
    with torch.no_grad():
        torch.onnx.export(Core.build(model), tuple(m), str(path), input_names=names, output_names=outs,
                          dynamic_axes=axes, opset_version=opset, dynamo=False)
    print(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB)")


def session(path):
    import onnxruntime as ort

    o = ort.SessionOptions()
    o.enable_cpu_mem_arena = False
    o.enable_mem_pattern = False
    return ort.InferenceSession(str(path), o, providers=["CPUExecutionProvider"])


def verify(model, path, tol):
    """Sources rebuilt from onnxruntime's masks against MRX.forward: noise and tones, mono and stereo."""
    import math
    import numpy as np
    import torch
    import mrx

    s = session(path)
    t = torch.arange(RATE * 3) / RATE
    tone = sum(torch.sin(2 * math.pi * f * t) / k for k, f in enumerate((110, 220, 440, 880, 1760), 1))
    for name, x in (("noise", 0.1 * torch.randn(1, RATE * 3)), ("tones", 0.2 * torch.stack([tone, 0.8 * tone]))):
        with torch.no_grad():
            want = model(x).numpy()
            m, specs = mags(x)
            masks = s.run(None, {f"mag_{n}": v.numpy() for n, v in zip(WINDOWS, m)})
            got = sum(mrx._istft(torch.from_numpy(k) * z[:, None], HOP, x.shape[-1]) for k, z in zip(masks, specs))
            got = got.transpose(0, 1).numpy()
        rel = np.abs(got - want).max() / max(np.abs(want).max(), 1e-12)
        print(f"  verify {path.name} {name}: max|diff|/max|y| = {rel:.2e} " + ("OK" if rel < tol else "FAIL"))
        if rel >= tol:
            raise SystemExit(f"{path}: verification failed ({rel:.2e} >= {tol})")


def reference(model, out_dir):
    """The reference separation: upstream's separate_soundtrack (-27 LUFS in, back out) on reference.py's mix, 20 s
    (longer than the package's chunks are not: MRX hears the whole input at once, as upstream)."""
    import torch

    sys.path.insert(0, str(Path(__file__).parent))
    import reference as ref
    import separate

    def run(mix):
        with torch.no_grad():
            y = separate.separate_soundtrack(torch.from_numpy(mix.copy()), separation_model=model)
        return {"dialogue": y["speech"].numpy(), "music": y["music"].numpy(), "effects": y["sfx"].numpy()}

    ref.write(out_dir, run, 20.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=None, help="default: $AUDIO_NEURAL_CACHE/mrx or ~/.cache/audiojs/neural/mrx")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--verify", action="store_true", help="compare with MRX.forward; write the reference separation")
    args = ap.parse_args()

    cache = Path(os.environ.get("AUDIO_NEURAL_CACHE") or Path.home() / ".cache" / "audiojs" / "neural")
    out_dir = Path(args.out_dir) if args.out_dir else cache / "mrx"
    out_dir.mkdir(parents=True, exist_ok=True)
    src = out_dir / "src"
    src.mkdir(exist_ok=True)
    try:
        model = load(src)
    except ImportError as e:
        raise SystemExit(f"torch is required: pip install torch onnx onnxruntime pyloudnorm ({e})")
    path = out_dir / "mrx.onnx"
    export(model, path, args.opset)
    if args.verify:
        verify(model, path, 1e-4)
        reference(model, out_dir)
    print(f"done: mrx in {out_dir}")


if __name__ == "__main__":
    sys.exit(main())
