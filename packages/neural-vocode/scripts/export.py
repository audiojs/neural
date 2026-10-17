#!/usr/bin/env python3
"""Export Vocos (charactr/vocos-mel-24khz, MIT) for vocode.js: the log-mel spectrogram in, the complex STFT out.

    uv pip install vocos soundfile onnx onnxruntime onnxscript
    python3 export.py --verify                     # vocos.onnx, then: compact.py --model vocos --as fp16
    python3 export.py --reference voice.wav        # voice.24k.f32 and voice.vocos.f32, the PyTorch pipeline's

Vocos (Siuzdak, "Vocos: Closing the gap between time-domain and Fourier-based neural vocoders for high-quality audio
synthesis", ICLR 2024, arXiv:2306.00814) predicts each STFT frame's magnitude and phase from a mel spectrogram. The
graph is what Vocos.decode computes between its features and its iSTFT: the ConvNeXt backbone and the head's linear
layer, the magnitude exp'd and clipped at 100, the phase's cos and sin: re, im of [1, 513, T]. The features (torchaudio's
MelSpectrogram: 24 kHz, n_fft 1024, hop 256, 100 HTK mels, power 1, centered with reflect padding; log clipped at 1e-7)
and the iSTFT (torch.istft, Hann of 1024, centered) are vocode.js's own, in JS: the same on every runtime.

--verify: the ONNX graph against the PyTorch forward on the features of noise and tones, max |diff| over max |y|.
--reference: a sound's first channel resampled to 24 kHz (torchaudio), and Vocos's own decode of its features, planar float32: the
fixtures test.js compares vocode.js's mel and output with.

Default out-dir: $AUDIO_NEURAL_CACHE/vocos, else ~/.cache/audiojs/neural/vocos.
"""
import argparse
import os
from pathlib import Path

import torch

REPO, REVISION = "charactr/vocos-mel-24khz", "0feb3fdd929bcd6649e0e7c5a688cf7dd012ef21"


class Core(torch.nn.Module):
    def __init__(self, vocos):
        super().__init__()
        self.backbone, self.out = vocos.backbone, vocos.head.out

    def forward(self, mel):
        x = self.out(self.backbone(mel)).transpose(1, 2)
        mag, p = x.chunk(2, dim=1)
        mag = torch.clip(torch.exp(mag), max=1e2)
        return mag * torch.cos(p), mag * torch.sin(p)


def model():
    from vocos import Vocos
    return Vocos.from_pretrained(REPO, revision=REVISION).eval()


def export(v, path):
    mel = v.feature_extractor(torch.randn(1, 24000) * 0.1)
    torch.onnx.export(Core(v), (mel,), str(path), input_names=["mel"], output_names=["re", "im"], opset_version=17,
                      dynamic_axes={"mel": {2: "frames"}, "re": {2: "frames"}, "im": {2: "frames"}}, dynamo=False)


def verify(v, path):
    import numpy as np
    import onnxruntime as ort
    s, t = ort.InferenceSession(str(path)), torch.arange(48000) / 24000
    for name, x in [("noise", torch.randn(1, 48000) * 0.1), ("tones", 0.3 * torch.sin(2 * torch.pi * 220 * t)[None] + 0.1 * torch.sin(2 * torch.pi * 3100 * t)[None])]:
        mel = v.feature_extractor(x)
        with torch.no_grad(): want = Core(v)(mel)
        got = s.run(None, {"mel": mel.numpy()})
        for k, (w, g) in enumerate(zip(want, got)):
            rel = np.abs(w.numpy() - g).max() / np.abs(w.numpy()).max()
            print(f"  verify {name} {['re', 'im'][k]}: max|diff|/max|y| = {rel:.2e} " + ("OK" if rel < 1e-4 else "FAIL"))


def reference(v, wav, d):
    import soundfile
    import torchaudio
    x, sr = soundfile.read(wav, dtype="float32", always_2d=True)
    x = torchaudio.functional.resample(torch.from_numpy(x.T[:1].copy()), sr, 24000)
    with torch.no_grad(): y = v.decode(v.feature_extractor(x))
    stem = Path(wav).stem
    x[0].numpy().astype("float32").tofile(d / f"{stem}.24k.f32")
    y[0].numpy().astype("float32").tofile(d / f"{stem}.vocos.f32")
    print(f"wrote {d / stem}.24k.f32 and .vocos.f32 ({x.shape[1]} and {y.shape[1]} samples)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--reference", default=None, help="a sound file: write its 24 kHz samples and Vocos's decode of them")
    args = ap.parse_args()
    d = Path(args.out_dir or Path(os.environ.get("AUDIO_NEURAL_CACHE") or Path.home() / ".cache" / "audiojs" / "neural") / "vocos")
    d.mkdir(parents=True, exist_ok=True)
    v = model()
    if args.reference: return reference(v, args.reference, d)
    path = d / "vocos.onnx"
    export(v, path)
    print(f"wrote {path} ({path.stat().st_size / 1e6:.2f} MB)")
    if args.verify: verify(v, path)


if __name__ == "__main__":
    main()
