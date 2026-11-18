#!/usr/bin/env python3
"""DNSMOS P.835 as Microsoft's dnsmos_local.py computes it, on the waveforms scripts/inputs.mjs makes: fixtures/ for test.js.

From the package directory, after npm install at the repository's root (it runs node scripts/inputs.mjs):

    uv run --python 3.11 --with librosa==0.9.1 --with resampy==0.2.2 --with numba==0.59.1 --with numpy==1.26.4 \\
        --with onnxruntime==1.30.0 --with soundfile==0.14.0 --with pandas --with tqdm --with requests \\
        --with 'setuptools<70' python -I scripts/reference.py

Reads DNSMOS/dnsmos_local.py and DNSMOS/DNSMOS/sig_bak_ovr.onnx of microsoft/DNS-Challenge at COMMIT from <dir>
($AUDIO_NEURAL_CACHE/dnsmos, else ~/.cache/audiojs/neural/dnsmos), fetched there first where missing, each refused
unless of its SHA-256. Imports the script and calls it as its main() does, ComputeScore(model)(wav, 16000, False), on
each input written as a float32 WAV (soundfile reads it back to the same values), recording every call of the network.
librosa 0.9.1 and resampy 0.2.2, the releases of the day the commit was made (2022-05-24): the script calls
librosa.resample(aud, input_fs, fs) positionally, which librosa 0.10 and later refuse; 0.9's default resampler is
resampy's 'kaiser_best', whose filter resampy 0.3.0 (2022-06-29) replaced. Writes fixtures/reference.json: the versions, the inputs' SHA-256,
and per input SIG, BAK, OVRL and num_hops as ComputeScore returns them, each window's raw outputs (the ramps: each
window's first sample, from its value, instead); get_polyfit_val on some raw values; and, of lena at 44.1 and 8 kHz,
the length of librosa's 16 kHz signal and 0.5 s of it (fixtures/<input>.librosa16.f32, samples SLICE).
"""
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf

COMMIT = "82f1b17e7776a43eee395d0f45bae8abb700ad00"  # microsoft/DNS-Challenge, "Add support for personalized and improved models"
RAW = f"https://raw.githubusercontent.com/microsoft/DNS-Challenge/{COMMIT}/DNSMOS/"
SOURCES = {
    "dnsmos_local.py": (RAW + "dnsmos_local.py", "7195de211df8b1943570ee2e2d920099fc1bcca1eb950758e8ca390b2463be55"),
    "sig_bak_ovr.onnx": (RAW + "DNSMOS/sig_bak_ovr.onnx", "269fbebdb513aa23cddfbb593542ecc540284a91849ac50516870e1ac78f6edd"),
}
HERE = Path(__file__).resolve().parent.parent
SLICE = (80000, 88000)  # 0.5 s of the 16 kHz signal, 5 s in


def sha(b):
    return hashlib.sha256(b).hexdigest()


def fetch(dir):
    dir.mkdir(parents=True, exist_ok=True)
    for name, (url, want) in SOURCES.items():
        p = dir / name
        if not p.exists():
            with urllib.request.urlopen(url) as r:
                b = r.read()
            if sha(b) != want:
                sys.exit(f"{url}: SHA-256 {sha(b)}, expected {want}")
            p.write_bytes(b)
        if sha(p.read_bytes()) != want:
            sys.exit(f"{p}: SHA-256 {sha(p.read_bytes())}, expected {want}: not the file at {COMMIT}")


def main():
    import librosa, onnxruntime, resampy
    sys.dont_write_bytecode = True  # importing dnsmos_local.py leaves no __pycache__ in the neural cache
    cache = Path(os.environ.get("AUDIO_NEURAL_CACHE") or Path.home() / ".cache" / "audiojs" / "neural") / "dnsmos"
    fetch(cache)
    spec = importlib.util.spec_from_file_location("dnsmos_local", cache / "dnsmos_local.py")
    dl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dl)
    cs = dl.ComputeScore(str(cache / "sig_bak_ovr.onnx"))
    calls, run = [], cs.onnx_sess.run

    def recorded(*a):
        r = run(*a)
        x = a[1]["input_1"]
        calls.append(dict(first=float(x[0, 0]), length=int(x.shape[1]), raw=[float(v) for v in r[0][0]]))
        return r
    cs.onnx_sess.run = recorded

    seen = {}
    resample = dl.librosa.resample
    def kept(y, *a, **k):
        seen["y16"] = z = resample(y, *a, **k)
        return z
    dl.librosa.resample = kept

    out = dict(by="scripts/reference.py: dnsmos_local.py's ComputeScore, called as its main() calls it",
               commit=COMMIT, script=SOURCES["dnsmos_local.py"][1], model=SOURCES["sig_bak_ovr.onnx"][1],
               versions=dict(python=platform.python_version(), numpy=np.__version__, librosa=librosa.__version__,
                             resampy=resampy.__version__, onnxruntime=onnxruntime.__version__, soundfile=sf.__version__),
               slice=list(SLICE), inputs={})
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["node", str(HERE / "scripts" / "inputs.mjs"), tmp], check=True, cwd=HERE)
        rates = json.loads(Path(tmp, "rates.json").read_text())
        for name, sr in rates.items():
            b = Path(tmp, f"{name}.f32").read_bytes()
            x = np.frombuffer(b, dtype=np.float32)
            wav = Path(tmp, f"{name}.wav")
            sf.write(wav, x, sr, subtype="FLOAT")
            assert np.array_equal(sf.read(wav, dtype="float32")[0], x)
            calls.clear(); seen.clear()
            d = cs(str(wav), 16000, False)
            r = dict(samples=len(x), rate=sr, sha256=sha(b), num_hops=int(d["num_hops"]),
                     SIG=float(d["SIG"]), BAK=float(d["BAK"]), OVRL=float(d["OVRL"]))
            if name.startswith("ramp-"):
                # sample i of a ramp is i · 2^-21: a window's first value is the input sample it starts on
                r["windows"] = [round(c["first"] * 2 ** 21) for c in calls]
                r["lengths"] = sorted({c["length"] for c in calls})
                del r["SIG"], r["BAK"], r["OVRL"]
            else:
                r["raw"] = [c["raw"] for c in calls]
            if "y16" in seen:
                y = np.asarray(seen["y16"], dtype=np.float32)
                r["length16"] = len(y)
                (HERE / "fixtures" / f"{name}.librosa16.f32").write_bytes(y[SLICE[0]:SLICE[1]].tobytes())
            out["inputs"][name] = r
            print(name, {k: v for k, v in r.items() if k in ("num_hops", "SIG", "BAK", "OVRL")}, len(calls), "windows", flush=True)

    # the polynomials alone, on lena's raw outputs and on 1, 3 and 5, through get_polyfit_val
    raws = out["inputs"]["lena@16k"]["raw"] + [[1.0, 1.0, 1.0], [3.0, 3.0, 3.0], [5.0, 5.0, 5.0]]
    out["polyfit"] = [dict(raw=r, mos=[float(v) for v in cs.get_polyfit_val(*map(np.float32, r), False)]) for r in raws]
    (HERE / "fixtures" / "reference.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
