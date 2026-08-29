#!/usr/bin/env python3
"""Export open-unmix-pytorch checkpoints to the ONNX contract @audio/neural-separate
expects (modelType 'openunmix'): one graph per target, magnitude spectrogram
in, estimated magnitude spectrogram out, shape [1, C, F, T] — batch, channels,
frequency bins, time frames — row-major, T (frames) dynamic.

NOT EXECUTED IN THIS ENVIRONMENT. torch is not installed here (installing it
is a 500+ MB dependency this package doesn't otherwise need) and pretrained
weights require a network fetch. This script is reviewed carefully against
open-unmix-pytorch's actual source (model.py, utils.py, filtering.py,
__init__.py, commit as of 2026-08 — sigsep/open-unmix-pytorch, MIT) rather
than run — the --verify step below is exactly the check to run before
trusting an export.

    pip install torch openunmix onnx onnxruntime   # ~500 MB+, torch's CUDA/CPU wheel
    python3 export-openunmix.py --model umxhq --targets vocals,drums,bass,other \\
        --out-dir ./onnx --verify

Why this shape and not a literal port of OpenUnmix.forward's own docstring
order: the module's `forward(x)` docstring says
"x: input spectrogram of shape (nb_samples, nb_channels, nb_bins, nb_frames)"
and returns the identical shape — that IS already [1, C, F, T] batch-first,
so the wrapper below is a direct call-through, not a reshape. No permute is
needed on the Python side.

max_bin / bandwidth: umxhq restricts the *input* bandwidth to bins below
16 kHz (`max_bin = bandwidth_to_max_bin(rate, n_fft, 16000)`, 1487 of 2049
bins for n_fft=4096 @ 44.1 kHz — see openunmix/utils.py) before the LSTM
body (`x = x[..., :self.nb_bins]`), for efficiency (bins above 16 kHz carry
little musical energy). Contrary to the folk description ("zero-pads bins
above max_bin"), the final dense layer (`fc3`, out_features =
nb_output_bins * nb_channels = full 2049 bins) regresses the FULL bin range
from that reduced representation — it is a *learned extrapolation*, not a
literal zero-fill. Read model.py's OpenUnmix.forward before changing this
comment if the upstream implementation has since changed.
"""
import argparse
import sys


def build_wrapper(target_model):
    import torch.nn as nn

    class ExportWrapper(nn.Module):
        """Call-through — OpenUnmix.forward already takes/returns [1, C, F, T]
        (nb_samples, nb_channels, nb_bins, nb_frames); max_bin cropping and the
        full-band regression happen inside `target_model` itself."""

        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, magnitude):
            return self.model(magnitude)

    return ExportWrapper(target_model)


def build_combined_wrapper(target_models, targets):
    import torch
    import torch.nn as nn

    class CombinedWrapper(nn.Module):
        """All targets in one graph — output stacks a target axis after batch:
        [1, S, C, F, T]. Matches the JS side's `{ url, targets }` model option."""

        def __init__(self, models, names):
            super().__init__()
            self.models = nn.ModuleList(models)
            self.names = names

        def forward(self, magnitude):
            return torch.stack([m(magnitude) for m in self.models], dim=1)

    return CombinedWrapper([target_models[t] for t in targets], targets)


def load_targets(model_name, targets, checkpoint_dir):
    """Returns { target: nn.Module }, each already .eval()."""
    import openunmix

    if checkpoint_dir:
        # local checkpoint dir: <dir>/<target>.pth + <dir>/<target>.json, per
        # openunmix.utils.load_target_models's documented layout.
        models = openunmix.utils.load_target_models(
            targets=targets, model_str_or_path=checkpoint_dir, pretrained=True
        )
    else:
        spec_fn = getattr(openunmix, f"{model_name}_spec", None)
        if spec_fn is None:
            raise SystemExit(
                f"unknown --model '{model_name}' — expected umx, umxhq, umxl, umxse, "
                f"or --checkpoint-dir pointing at a local <target>.pth/.json pair"
            )
        models = spec_fn(targets=targets, pretrained=True)
    for m in models.values():
        m.eval()
    return models


def export_one(wrapped, out_path, n_channels, n_bins, opset, fp16, verify, orig_model=None):
    import torch

    dummy_frames = 8  # arbitrary — the frames axis is exported dynamic
    x = torch.randn(1, n_channels, n_bins, dummy_frames, dtype=torch.float32).abs()

    torch.onnx.export(
        wrapped,
        (x,),
        str(out_path),
        input_names=["magnitude"],
        output_names=["estimate"],
        dynamic_axes={"magnitude": {0: "batch", 3: "frames"}, "estimate": {0: "batch", 3: "frames"}},
        opset_version=opset,
        do_constant_folding=True,
    )
    print(f"wrote {out_path}")

    if fp16:
        # Post-convert rather than exporting a half model directly — LSTM's
        # ONNX export in fp16 has historically been unreliable across opset
        # versions; converting the verified fp32 graph is the safer path.
        from onnxconverter_common import float16
        import onnx

        m = onnx.load(str(out_path))
        m16 = float16.convert_float_to_float16(m, keep_io_types=True)
        fp16_path = out_path.with_suffix(".fp16.onnx")
        onnx.save(m16, str(fp16_path))
        print(f"wrote {fp16_path}")

    if verify:
        verify_export(wrapped if orig_model is None else orig_model, out_path, n_channels, n_bins)


def verify_export(torch_model, onnx_path, n_channels, n_bins):
    """Runs the exported graph through onnxruntime on random input and compares
    against the torch model directly — asserts max abs diff < 1e-4."""
    import numpy as np
    import onnxruntime as ort
    import torch

    for n_frames in (1, 8, 37):  # exercise the dynamic frames axis, incl. a short/odd size
        x = np.abs(np.random.randn(1, n_channels, n_bins, n_frames)).astype(np.float32)
        with torch.no_grad():
            expected = torch_model(torch.from_numpy(x)).numpy()

        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        (actual,) = session.run(None, {"magnitude": x})

        diff = np.abs(actual - expected).max()
        status = "OK" if diff < 1e-4 else "FAIL"
        print(f"  verify frames={n_frames}: max|Δ|={diff:.2e} [{status}]")
        if diff >= 1e-4:
            raise SystemExit(f"{onnx_path}: verification failed (max|Δ|={diff:.2e} ≥ 1e-4)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="umxhq", help="umx | umxhq | umxl | umxse (torchhub pretrained variant)")
    ap.add_argument("--checkpoint-dir", default=None, help="local dir with <target>.pth/.json instead of torchhub")
    ap.add_argument("--targets", default="vocals,drums,bass,other", help="comma-separated target names")
    ap.add_argument("--out-dir", default="./onnx", help="output directory")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--fp16", action="store_true", help="also write a float16 variant (needs onnxconverter-common)")
    ap.add_argument("--combined", action="store_true", help="also write one multi-target graph (targets.onnx, output [1,S,C,F,T])")
    ap.add_argument("--verify", action="store_true", help="run onnxruntime vs. torch on random input, assert max|Δ| < 1e-4")
    args = ap.parse_args()

    try:
        import torch  # noqa: F401
    except ImportError:
        raise SystemExit("torch is required to run this script: pip install torch openunmix onnx onnxruntime")

    from pathlib import Path

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = args.targets.split(",")

    target_models = load_targets(args.model, targets, args.checkpoint_dir)

    # nb_channels / nb_bins are fixed per checkpoint architecture (not exported
    # dynamic) — read them off the first loaded model.
    any_model = next(iter(target_models.values()))
    n_channels = any_model.fc1.in_features // any_model.nb_bins  # fc1: Linear(nb_bins*nb_channels, hidden)
    n_bins = any_model.nb_output_bins

    for name in targets:
        wrapped = build_wrapper(target_models[name])
        export_one(wrapped, out_dir / f"{name}.onnx", n_channels, n_bins, args.opset, args.fp16, args.verify, orig_model=wrapped)

    if args.combined:
        combined = build_combined_wrapper(target_models, targets)
        export_one(combined, out_dir / "targets.onnx", n_channels, n_bins, args.opset, args.fp16, args.verify, orig_model=combined)

    print(f"\ndone — {len(targets)} target graph(s) in {out_dir}" + (" + 1 combined graph" if args.combined else ""))
    print("JS side: model: { " + ", ".join(f"{t}: '{out_dir}/{t}.onnx'" for t in targets) + " }, modelType: 'openunmix'")
    if args.combined:
        print(f"     or: model: {{ url: '{out_dir}/targets.onnx', targets: {targets!r} }}, modelType: 'openunmix'")


if __name__ == "__main__":
    sys.exit(main())
