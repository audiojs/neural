#!/usr/bin/env python3
"""Export open-unmix-pytorch checkpoints to the ONNX contract @audio/neural-separate
expects (modelType 'openunmix'): one graph per target, magnitude spectrogram
in, estimated magnitude spectrogram out, shape [1, C, F, T] (batch, channels,
frequency bins, time frames), row-major, T dynamic.

    pip install torch openunmix onnx onnxruntime onnxscript
    python3 export-openunmix.py --model umxhq --verify

Writes <out-dir>/<target>.onnx, weights embedded (one self-contained file per
target, loadable from bytes). Default out-dir: $AUDIO_NEURAL_CACHE/<model>, else
~/.cache/audiojs/neural/<model>, where `separate(audio, { model: 'umxhq' })`
looks. Last run: torch 2.14, onnx 1.23, onnxruntime 1.30, openunmix 1.3.0.

Two things keep the frame axis dynamic, both found by running this:
- A wrapper, not OpenUnmix itself. OpenUnmix.forward reads the frame count off
  `x.data.shape`; `.data` leaves the traced graph, so the exporter records the
  dummy input's frame count as a constant. ExportWrapper restates the same
  forward with shapes taken from the traced tensor; --verify compares the ONNX
  graph against the original module, so the restatement is checked, not trusted.
- The TorchScript exporter (dynamo=False). The torch.export-based default bakes
  the LSTM's sequence length into a Reshape even with dynamic_shapes declared.

max_bin: umxhq crops the network input to bins below 16 kHz
(`bandwidth_to_max_bin(44100, 4096, 16000)`, 1487 of 2049 bins,
openunmix/utils.py), and fc3 regresses all 2049 output bins from that: a learned
extrapolation above 16 kHz, not a zero fill.
"""
import argparse
import os
import sys
from pathlib import Path


def build_wrapper(model):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class ExportWrapper(nn.Module):
        """OpenUnmix.forward (openunmix/model.py) with symbolic shapes. Eval mode:
        dropout and batch-norm statistics are the module's own."""

        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, magnitude):
            m = self.m
            x = magnitude.permute(3, 0, 1, 2)  # (frames, samples, channels, bins)
            mix = x
            frames, samples, channels = x.shape[0], x.shape[1], x.shape[2]
            x = x[..., : m.nb_bins]
            x = (x + m.input_mean) * m.input_scale
            x = m.bn1(m.fc1(x.reshape(-1, channels * m.nb_bins)))
            x = torch.tanh(x.reshape(frames, samples, m.hidden_size))
            x = torch.cat([x, m.lstm(x)[0]], -1)
            x = F.relu(m.bn2(m.fc2(x.reshape(-1, x.shape[-1]))))
            x = m.bn3(m.fc3(x)).reshape(frames, samples, channels, m.nb_output_bins)
            x = x * m.output_scale + m.output_mean
            return (F.relu(x) * mix).permute(1, 2, 3, 0)

    return ExportWrapper(model).eval()


def build_combined(models):
    import torch
    import torch.nn as nn

    class CombinedWrapper(nn.Module):
        """All targets in one graph, output [1, S, C, F, T]: the JS side's
        `{ url, targets }` model option."""

        def __init__(self, ms):
            super().__init__()
            self.ms = nn.ModuleList(ms)

        def forward(self, magnitude):
            return torch.stack([m(magnitude) for m in self.ms], dim=1)

    return CombinedWrapper(models).eval()


def load_targets(model_name, targets, checkpoint_dir):
    """{ target: OpenUnmix } in eval mode."""
    import openunmix

    if checkpoint_dir:
        # <dir>/<target>.pth + <dir>/<target>.json, openunmix.utils.load_target_models's layout
        models = openunmix.utils.load_target_models(targets=targets, model_str_or_path=checkpoint_dir, pretrained=True)
    else:
        spec_fn = getattr(openunmix, f"{model_name}_spec", None)
        if spec_fn is None:
            raise SystemExit(f"unknown --model '{model_name}': expected umx, umxhq, umxl, umxse, or --checkpoint-dir")
        models = spec_fn(targets=targets, pretrained=True)
    for m in models.values():
        m.eval()
    return models


def export_one(wrapped, reference, out_path, n_channels, n_bins, opset, fp16, verify):
    import torch

    x = torch.randn(1, n_channels, n_bins, 8).abs()
    torch.onnx.export(
        wrapped,
        (x,),
        str(out_path),
        input_names=["magnitude"],
        output_names=["estimate"],
        dynamic_axes={"magnitude": {3: "frames"}, "estimate": {3: "frames"}},
        opset_version=opset,
        dynamo=False,
    )
    print(f"wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    if verify:
        verify_export(reference, out_path, n_channels, n_bins, 1e-4)

    if fp16:
        # post-converted from the verified fp32 graph, float32 I/O kept
        import onnx
        from onnxruntime.transformers.float16 import convert_float_to_float16

        fp16_path = out_path.with_suffix(".fp16.onnx")
        onnx.save(convert_float_to_float16(onnx.load(str(out_path)), keep_io_types=True), str(fp16_path))
        print(f"wrote {fp16_path} ({fp16_path.stat().st_size / 1e6:.1f} MB)")
        if verify:
            verify_export(reference, fp16_path, n_channels, n_bins, None)


def verify_export(reference, onnx_path, n_channels, n_bins, tol):
    """onnxruntime vs the original torch module on random magnitudes. fp32: max
    |diff| relative to max |output| < tol. fp16 (tol None): report only."""
    import numpy as np
    import onnxruntime as ort
    import torch

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    for n_frames in (1, 8, 37, 300):  # the dynamic frames axis, incl. 1 and Open-Unmix's Wiener window
        x = np.abs(rng.standard_normal((1, n_channels, n_bins, n_frames))).astype(np.float32)
        with torch.no_grad():
            expected = reference(torch.from_numpy(x)).numpy()
        (actual,) = session.run(None, {"magnitude": x})
        rel = np.abs(actual - expected).max() / max(np.abs(expected).max(), 1e-12)
        ok = tol is None or rel < tol
        print(f"  verify {onnx_path.name} frames={n_frames}: max|diff|/max|y| = {rel:.2e}" + ("" if tol is None else " OK" if ok else " FAIL"))
        if not ok:
            raise SystemExit(f"{onnx_path}: verification failed ({rel:.2e} >= {tol})")


def write_reference(model_name, targets, out_dir):
    """openunmix's own Separator as `umx` runs it (niter=1, 300-frame Wiener windows), in float64:
    upstream's float32 EM alone moves the stems by -71 to -93 dB on this mix, which would hide
    any difference in the port."""
    import numpy as np
    import openunmix
    import torch

    sys.path.insert(0, str(Path(__file__).parent))
    sys.dont_write_bytecode = True  # no __pycache__ beside the package's scripts
    import reference

    sep = openunmix.utils.load_separator(model_name, targets=targets, niter=1, residual=False, wiener_win_len=300, device="cpu").eval().double()

    def separate(mix):
        with torch.no_grad():
            est = sep(torch.from_numpy(mix.astype(np.float64))[None])[0]
        return {t: est[k].numpy() for k, t in enumerate(sep.target_models)}

    reference.write(out_dir, separate)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="umxhq", help="umx | umxhq | umxl | umxse (umxl weights are CC BY-NC-SA 4.0)")
    ap.add_argument("--checkpoint-dir", default=None, help="local dir with <target>.pth/.json instead of the pretrained download")
    ap.add_argument("--targets", default=None, help="comma-separated targets (default: all the model has)")
    ap.add_argument("--out-dir", default=None, help="default: $AUDIO_NEURAL_CACHE/<model> or ~/.cache/audiojs/neural/<model>")
    ap.add_argument("--opset", type=int, default=18)
    ap.add_argument("--fp16", action="store_true", help="also write <target>.fp16.onnx (float16 weights, float32 I/O)")
    ap.add_argument("--combined", action="store_true", help="also write targets.onnx, one multi-target graph, output [1,S,C,F,T]")
    ap.add_argument("--verify", action="store_true", help="compare onnxruntime against the torch module (relative max|diff| < 1e-4), "
                    "and write the reference separation test.js compares against (scripts/reference.py)")
    args = ap.parse_args()

    try:
        import torch  # noqa: F401
    except ImportError:
        raise SystemExit("torch is required: pip install torch openunmix onnx onnxruntime onnxscript")

    cache = os.environ.get("AUDIO_NEURAL_CACHE") or Path.home() / ".cache" / "audiojs" / "neural"
    out_dir = Path(args.out_dir) if args.out_dir else Path(cache) / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = args.targets.split(",") if args.targets else None

    models = load_targets(args.model, targets, args.checkpoint_dir)
    targets = list(models)
    any_model = next(iter(models.values()))
    n_channels = any_model.fc1.in_features // any_model.nb_bins  # fc1: Linear(nb_bins * nb_channels, hidden)
    n_bins = any_model.nb_output_bins

    for name in targets:
        export_one(build_wrapper(models[name]), models[name], out_dir / f"{name}.onnx", n_channels, n_bins, args.opset, args.fp16, args.verify)

    if args.combined:
        combined = build_combined([build_wrapper(models[t]) for t in targets])
        reference = build_combined([models[t] for t in targets])
        export_one(combined, reference, out_dir / "targets.onnx", n_channels, n_bins, args.opset, args.fp16, args.verify)

    if args.verify and not args.checkpoint_dir:
        write_reference(args.model, targets, out_dir)

    print(f"\ndone: {len(targets)} target graph(s) in {out_dir}" + (" + targets.onnx" if args.combined else ""))
    print("JS: separate(audio, { model: { " + ", ".join(f"{t}: '{out_dir}/{t}.onnx'" for t in targets) + " } })")


if __name__ == "__main__":
    sys.exit(main())
