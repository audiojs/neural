#!/usr/bin/env python3
"""Export Demucs v4 (Hybrid Transformer Demucs) to the ONNX contract
@audio/neural-separate expects for modelType 'hybrid': the network without its
STFT and iSTFT, which ONNX export cannot carry. The split is sevagh's
demucs.onnx (github.com/sevagh/demucs.onnx, MIT); JS computes both transforms.

    pip install torch demucs onnx onnxruntime
    python3 export-htdemucs.py --model htdemucs --verify
    python3 export-htdemucs.py --model htdemucs_ft --verify   # four source-specialist graphs

One 7.8 s training segment (343980 samples at 44.1 kHz) per run:
  inputs   mix       [1, 2, 343980]         waveform
           mix_spec  [1, 4, 2048, 336]      STFT as complex-as-channels (L re, L im, R re, R im)
  outputs  stems_spec  [1, S, 4, 2048, 336] frequency branch, same layout
           stems_wave  [1, S, 2, 343980]    time branch
  stem = iSTFT(stems_spec) + stems_wave

Writes <out-dir>/<model>.onnx (htdemucs_ft: <out-dir>/<source>.onnx per
specialist), weights embedded. Default out-dir: $AUDIO_NEURAL_CACHE/<model>, else
~/.cache/audiojs/neural/<model>. Last run: torch 2.14, demucs 4.1.0,
onnxruntime 1.30.

How: HTDemucs.forward runs unmodified with its four transform methods swapped
on the instance: _spec and _magnitude return the given spectrogram, _mask keeps
the frequency-branch estimate, _ispec returns 0 so forward returns the time
branch alone. sevagh edits a vendored copy of htdemucs.py to the same effect;
patching keeps upstream's forward as the single source. --verify rebuilds the
stems from the ONNX outputs with demucs's own _mask and _ispec and compares them
with model(mix) on the same segment.

Weights: Demucs's checkpoints "are not covered by the MIT license, and are
provided only for scientific purposes" (the author,
github.com/facebookresearch/demucs/issues/327). Memory: an export peaks near
3.2 GB; htdemucs_ft exports and checks each specialist in its own process.
"""
import argparse
import gc
import os
import subprocess
import sys
from pathlib import Path


def build_core(model):
    import torch
    import torch.nn as nn

    class HybridCore(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, mix, mix_spec):
            m, box = self.m, {}
            m._spec = lambda x: mix_spec
            m._magnitude = lambda z: z
            m._mask = lambda z, x: box.setdefault("spec", x)
            m._ispec = lambda z, length=None, scale=0: torch.zeros((), dtype=mix.dtype)
            try:
                wave = m(mix)
            finally:
                for k in ("_spec", "_magnitude", "_mask", "_ispec"):
                    del m.__dict__[k]
            return box["spec"], wave

    return HybridCore(model).eval()


def cac_spec(model, mix):
    """The mix's STFT as HTDemucs sees it (its own _spec and _magnitude)."""
    return model._magnitude(model._spec(mix))


def export_one(model, out_path, opset, verify):
    import torch

    L = int(model.segment * model.samplerate)
    mix = torch.randn(1, model.audio_channels, L)
    with torch.no_grad():
        spec = cac_spec(model, mix)
    # traced with grad enabled: under no_grad, nn.MultiheadAttention takes its fused
    # fast path, aten::_native_multi_head_attention, which has no ONNX export
    torch.onnx.export(
        build_core(model),
        (mix, spec),
        str(out_path),
        input_names=["mix", "mix_spec"],
        output_names=["stems_spec", "stems_wave"],
        opset_version=opset,
        dynamo=False,
    )
    print(f"wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB), segment {L} samples")
    gc.collect()  # the traced graph: export peaks near 3 GB
    if verify:
        verify_export(model, out_path, L)


def verify_export(model, onnx_path, L):
    """Stems rebuilt from onnxruntime's outputs against model(mix): a random
    segment and a harmonic one, relative max |diff| < 1e-3 (float32 through a
    transformer; a transform mismatch would show at order 1)."""
    import math
    import numpy as np
    import onnxruntime as ort
    import torch

    opts = ort.SessionOptions()
    opts.enable_cpu_mem_arena = False  # give memory back between runs
    opts.enable_mem_pattern = False
    session = ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])
    t = torch.arange(L) / model.samplerate
    tone = sum(torch.sin(2 * math.pi * f * t) / k for k, f in enumerate((110, 220, 440, 880, 1760), 1))
    for name, mix in (("noise", 0.1 * torch.randn(1, 2, L)), ("tones", 0.2 * torch.stack([tone, 0.8 * tone])[None])):
        with torch.no_grad():
            expected = model(mix).numpy()
            spec = cac_spec(model, mix)
            s, w = session.run(None, {"mix": mix.numpy(), "mix_spec": spec.numpy()})
            actual = (model._ispec(model._mask(None, torch.from_numpy(s)), L) + torch.from_numpy(w)).numpy()
        rel = np.abs(actual - expected).max() / max(np.abs(expected).max(), 1e-12)
        ok = rel < 1e-3
        print(f"  verify {onnx_path.name} {name}: max|diff|/max|y| = {rel:.2e} " + ("OK" if ok else "FAIL"))
        if not ok:
            raise SystemExit(f"{onnx_path}: verification failed ({rel:.2e} >= 1e-3)")
    del session
    gc.collect()


def write_reference(bag, out_dir):
    """demucs.api.Separator's path: whole-input normalization, apply_model with split=True,
    overlap=0.25 and shifts=0 (the CLI's one random shift is not reproducible)."""
    import torch
    from demucs.apply import apply_model

    sys.path.insert(0, str(Path(__file__).parent))
    sys.dont_write_bytecode = True  # no __pycache__ beside the package's scripts
    import reference

    def separate(mix):
        wav = torch.from_numpy(mix)
        ref = wav.mean(0)
        mean, std = ref.mean(), ref.std() + 1e-8
        with torch.no_grad():
            est = apply_model(bag, ((wav - mean) / std)[None], shifts=0, split=True, overlap=0.25)[0] * std + mean
        return {s: est[k].numpy() for k, s in enumerate(bag.sources)}

    reference.write(out_dir, separate)


def bag_of(name):
    """(signature, source) per model of a bag of source specialists (htdemucs_ft), read off the
    bag's YAML in the demucs package without importing torch; None for a single model."""
    import importlib.util
    import yaml

    spec = importlib.util.find_spec("demucs")
    f = Path(spec.origin).parent / "remote" / f"{name}.yaml" if spec else None
    bag = yaml.safe_load(f.read_text()) if f and f.exists() else {}
    if len(bag.get("models", [])) < 2 or not bag.get("weights"):
        return None
    sources = ["drums", "bass", "other", "vocals"]  # demucs.pretrained.SOURCES, the weights' column order
    return [(sig, sources[max(range(len(w)), key=lambda k: w[k])]) for sig, w in zip(bag["models"], bag["weights"])]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="htdemucs", help="htdemucs | htdemucs_ft | htdemucs_6s")
    ap.add_argument("--out-dir", default=None, help="default: $AUDIO_NEURAL_CACHE/<model> or ~/.cache/audiojs/neural/<model>")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--verify", action="store_true", help="compare the rebuilt stems with model(mix), and write the "
                    "reference separation test.js compares against (scripts/reference.py)")
    ap.add_argument("--source", default=None, help=argparse.SUPPRESS)  # a bag's specialist, in its own process
    ap.add_argument("--reference", action="store_true", help=argparse.SUPPRESS)  # a bag's reference, in its own process
    ap.add_argument("--check", action="store_true", help=argparse.SUPPRESS)  # verify a specialist's export, in its own process
    args = ap.parse_args()

    cache = os.environ.get("AUDIO_NEURAL_CACHE") or Path.home() / ".cache" / "audiojs" / "neural"
    out_dir = Path(args.out_dir) if args.out_dir else Path(cache) / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        bag = bag_of(args.model)
    except ImportError:
        raise SystemExit("torch and demucs are required: pip install torch demucs onnx onnxruntime")
    if bag and not (args.reference or args.source):
        # a bag of source specialists: one process each, since an export peaks near 3 GB and is
        # not handed back; this one stays free of torch
        run = lambda *extra: subprocess.run([sys.executable, __file__, "--model", args.model, "--out-dir", str(out_dir), "--opset", str(args.opset), *extra], check=True)
        if args.verify:
            run("--reference")
        for _, src in bag:
            run("--source", src)
            if args.verify:
                run("--source", src, "--check")
        print(f"done: {args.model} in {out_dir}")
        return

    try:
        from demucs.pretrained import get_model
        from demucs.htdemucs import HTDemucs
    except ImportError:
        raise SystemExit("torch and demucs are required: pip install torch demucs onnx onnxruntime")

    def load(name):
        model = get_model(name)
        for m in getattr(model, "models", [model]):
            if not isinstance(m, HTDemucs):
                raise SystemExit(f"{name}: not a Hybrid Transformer Demucs model")
            m.eval()
        return model

    if args.reference:
        write_reference(load(args.model), out_dir)
    elif args.source:
        # the one specialist, by its signature: a single HTDemucs; exported, or checked in a fresh process
        model, path = load(dict((s, g) for g, s in bag)[args.source]), out_dir / f"{args.source}.onnx"
        if args.check:
            verify_export(model, path, int(model.segment * model.samplerate))
        else:
            export_one(model, path, args.opset, False)
    else:
        model = load(args.model)
        if args.verify:
            write_reference(model, out_dir)  # first, while memory is fresh
            gc.collect()
        export_one(getattr(model, "models", [model])[0], out_dir / f"{args.model}.onnx", args.opset, args.verify)
        print(f"done: {args.model} in {out_dir}")


if __name__ == "__main__":
    sys.exit(main())
