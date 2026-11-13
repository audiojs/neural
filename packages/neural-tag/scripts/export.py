#!/usr/bin/env python3
"""YAMNet (tensorflow/models, research/audioset/yamnet, Apache-2.0) for tag.js: log-mel patches in, 521 scores out.

From the package directory, after npm install at the repository's root (--verify and --reference run node):

    uv run --python 3.12 --with tensorflow==2.21.0 --with tf-keras==2.21.0 --with h5py --with onnx --with onnxruntime \\
        python -I scripts/export.py --verify --reference

Fetches upstream's yamnet.py, features.py, params.py and yamnet_class_map.csv at COMMIT, and yamnet.h5, each refused
unless of its SHA-256, into <dir>/src (<dir>: $AUDIO_NEURAL_CACHE/yamnet, else ~/.cache/audiojs/neural/yamnet). Writes:

- <dir>/yamnet.onnx, yamnet.fp16.onnx: the network yamnet.yamnet() defines, built from yamnet.h5's arrays (h5py and
  onnx, no TensorFlow): `patches` [n, 96, 64] float32 in (frames × mel bands, as features.py frames them), `scores`
  [n, 521] out (sigmoid). Each batch norm (center, no scale, epsilon 1e-4: params.py) folded into the convolution before
  it, in float64; 'same' padding as TensorFlow places it (an odd total puts the extra row and column after). fp16: the
  weights of over 1024 values stored as float16 and Cast to float32 in the graph, the rest float32 (neural-separate's
  compact.py --as fp16 rule): computed in float32, half the file.
- classes.js: yamnet_class_map.csv's ids and names, in output order.
- --verify: both graphs against the Keras network (upstream's yamnet.yamnet, yamnet.h5 loaded) on upstream's own
  patches of every reference input (features.py's pad_waveform and waveform_to_log_mel_spectrogram_patches): the
  largest |difference| of the scores.
- --reference: upstream's yamnet_frames_model run on each input scripts/inputs.mjs writes (lena also at a 2 s hop),
  into fixtures/: <input>.scores.f32 (every patch's scores), <input>.patches.f32 (the log-mel of the patches
  reference.json names), reference.json (the inputs' SHA-256, shapes, the Keras model's top 3).
"""
import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

import numpy as np

COMMIT = "c14bf9ad91962cf189f9f58db2132c06247fcd53"  # tensorflow/models, the last to touch research/audioset/yamnet
RAW = f"https://raw.githubusercontent.com/tensorflow/models/{COMMIT}/research/audioset/yamnet/"
SOURCES = {
    "yamnet.py": (RAW + "yamnet.py", "7ef3df32b7ecb782490b5a04d7581a23cfcf701dbf476bc4d03deefe22cdb040"),
    "features.py": (RAW + "features.py", "e6cd53f81d072c7c43be4c7fff2b9dd0c5ccc7d64f2fbcfc85b44013d6d2ed5e"),
    "params.py": (RAW + "params.py", "925bb1e62461016031f98aea09aeac28975dd516f5747513767de5d1b06b6145"),
    "yamnet_class_map.csv": (RAW + "yamnet_class_map.csv", "cdf24d193e196d9e95912a2667051ae203e92a2ba09449218ccb40ef787c6df2"),
    # the Keras weights upstream's README links (Last-Modified 2019-11-18)
    "yamnet.h5": ("https://storage.googleapis.com/audioset/yamnet.h5", "13c3308955bbfaef262f175ac9c40e47b134573a93984f009220dd7cc12a1744"),
}
HERE = Path(__file__).resolve().parent.parent
EPS = 1e-4
# yamnet.py's _YAMNET_LAYER_DEFS: (kind, stride, filters), every kernel 3 × 3
LAYERS = [("conv", 2, 32), ("sep", 1, 64), ("sep", 2, 128), ("sep", 1, 128), ("sep", 2, 256), ("sep", 1, 256),
          ("sep", 2, 512), ("sep", 1, 512), ("sep", 1, 512), ("sep", 1, 512), ("sep", 1, 512), ("sep", 1, 512),
          ("sep", 2, 1024), ("sep", 1, 1024)]
# the reference inputs whose patches' log-mel goes into the fixtures (the rest are checked through their scores):
# the first and the last, its zero-padded tail
FEATURES = {"lena": "first,last", "pluck": "last", "sweep": "last", "noise": "last", "beep": "first"}


def fetch(src):
    src.mkdir(parents=True, exist_ok=True)
    for name, (url, sha) in SOURCES.items():
        p = src / name
        if not p.exists(): urllib.request.urlretrieve(url, p)
        if hashlib.sha256(p.read_bytes()).hexdigest() != sha: raise SystemExit(f"{p}: SHA-256 is not {sha}")


def same(n, stride, k):
    # TensorFlow's 'same' padding of a k-tap kernel over n: (before, after), an odd total's extra one after
    total = max((-(-n // stride) - 1) * stride + k - n, 0)
    return total // 2, total - total // 2


def build(h5, half):
    """yamnet.h5's network as an ONNX model; half: weights over 1024 values stored as float16, Cast to float32."""
    import h5py
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    f, inits, nodes = h5py.File(h5, "r"), [], []
    arr = lambda layer, name: np.array(f[f"{layer}/{layer}/{name}:0"], np.float64)

    def const(name, a):
        if half and a.size > 1024:
            inits.append(numpy_helper.from_array(a.astype(np.float16), name + "_h"))
            nodes.append(helper.make_node("Cast", [name + "_h"], [name], to=TensorProto.FLOAT))
        else: inits.append(numpy_helper.from_array(a.astype(np.float32), name))
        return name

    def conv(x, layer, k, stride, size, group=1):
        # k in ONNX's layout [out, in / group, 3, 3]; its batch norm folded: k / √(var + ε), β − mean / √(var + ε)
        beta, mean, var = (arr(layer + "/bn", n) for n in ("beta", "moving_mean", "moving_variance"))
        s = 1 / np.sqrt(var + EPS)
        (t, b), (l, r) = same(size[0], stride, k.shape[2]), same(size[1], stride, k.shape[3])
        y = layer.replace("/", "_")
        nodes.append(helper.make_node("Conv", [x, const(y + "_w", k * s[:, None, None, None]), const(y + "_b", beta - mean * s)], [y],
                                      kernel_shape=list(k.shape[2:]), strides=[stride, stride], pads=[t, l, b, r], group=group))
        nodes.append(helper.make_node("Relu", [y], [y + "_r"]))
        return y + "_r"

    inits.append(numpy_helper.from_array(np.array([1], np.int64), "axis1"))
    nodes.append(helper.make_node("Unsqueeze", ["patches", "axis1"], ["x"]))
    x, size = "x", (96, 64)
    for i, (kind, stride, _) in enumerate(LAYERS, 1):
        if kind == "conv":
            x = conv(x, f"layer{i}/conv", arr(f"layer{i}/conv", "kernel").transpose(3, 2, 0, 1), stride, size)
        else:
            k = arr(f"layer{i}/depthwise_conv", "depthwise_kernel")  # [3, 3, c, 1]
            x = conv(x, f"layer{i}/depthwise_conv", k.transpose(2, 3, 0, 1), stride, size, group=k.shape[2])
        size = (-(-size[0] // stride), -(-size[1] // stride))
        if kind == "sep":
            x = conv(x, f"layer{i}/pointwise_conv", arr(f"layer{i}/pointwise_conv", "kernel").transpose(3, 2, 0, 1), 1, size)
    nodes += [helper.make_node("GlobalAveragePool", [x], ["pooled"]), helper.make_node("Flatten", ["pooled"], ["embedding"], axis=1),
              helper.make_node("Gemm", ["embedding", const("logits_w", arr("logits", "kernel")), const("logits_b", arr("logits", "bias"))], ["logits"]),
              helper.make_node("Sigmoid", ["logits"], ["scores"])]
    g = helper.make_graph(nodes, "yamnet", [helper.make_tensor_value_info("patches", TensorProto.FLOAT, ["n", 96, 64])],
                          [helper.make_tensor_value_info("scores", TensorProto.FLOAT, ["n", 521])], inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)], ir_version=8, producer_name="audiojs neural-tag export.py")
    for k, v in {"source": f"tensorflow/models@{COMMIT} research/audioset/yamnet", "weights": f"yamnet.h5 sha256 {SOURCES['yamnet.h5'][1]}",
                 "weights_stored": "float16 (over 1024 values), computed in float32" if half else "float32"}.items():
        m.metadata_props.append(onnx.StringStringEntryProto(key=k, value=v))
    onnx.checker.check_model(m)
    return m


def classes(src):
    with open(src / "yamnet_class_map.csv", newline="") as fh:
        rows = list(csv.reader(fh))[1:]
    assert [int(r[0]) for r in rows] == list(range(521))
    lines = ",\n".join(f"\t{{ id: {json.dumps(mid)}, name: {json.dumps(name)} }}" for _, mid, name in rows)
    (HERE / "classes.js").write_text(
        "// AudioSet's classes in YAMNet's output order: yamnet_class_map.csv (tensorflow/models at "
        f"{COMMIT[:7]}, Apache-2.0); the ids and\n// names are the AudioSet ontology's (Google, CC BY-SA 4.0). "
        "Written by scripts/export.py.\nexport default [\n" + lines + ",\n]\n")


def inputs(d):
    d.mkdir(parents=True, exist_ok=True)
    for f in d.glob("*.f32"): f.unlink()
    subprocess.run(["node", str(HERE / "scripts" / "inputs.mjs"), str(d)], check=True, cwd=HERE)
    return {p.stem: p for p in sorted(d.glob("*.f32"))}


def upstream(src):
    sys.path.insert(0, str(src))
    import params, features, yamnet  # upstream's own modules, at COMMIT
    return params, features, yamnet


def keras(src, hop=None):
    """upstream's waveform model and its core over patches, yamnet.h5 loaded into each"""
    import tf_keras
    params, features, yamnet = upstream(src)
    p = params.Params() if hop is None else params.Params(patch_hop_seconds=hop)
    frames = yamnet.yamnet_frames_model(p)
    frames.load_weights(str(src / "yamnet.h5"))
    x = tf_keras.layers.Input(shape=(p.patch_frames, p.patch_bands))
    core = tf_keras.Model(x, yamnet.yamnet(x, p)[0])
    core.load_weights(str(src / "yamnet.h5"))
    patches = lambda w: features.waveform_to_log_mel_spectrogram_patches(features.pad_waveform(w, p), p)[1].numpy()
    return frames, core, patches


def verify(d, src):
    import onnxruntime as ort
    _, core, patches = keras(src)
    P = np.concatenate([patches(np.fromfile(f, "<f4")) for f in inputs(d / "inputs").values()])
    want = core.predict(P, batch_size=64, verbose=0)
    for name in ("yamnet.onnx", "yamnet.fp16.onnx"):
        got = ort.InferenceSession(str(d / name), providers=["CPUExecutionProvider"]).run(None, {"patches": P})[0]
        print(f"  verify {name}: {len(P)} patches, max |score − Keras| {np.abs(got - want).max():.2e}, "
              f"mean {np.abs(got - want).mean():.2e}; top class differs in {(got.argmax(1) != want.argmax(1)).sum()}")


def reference(d, src):
    import tensorflow as tf
    names = [r[2] for r in csv.reader(open(src / "yamnet_class_map.csv", newline=""))][1:]
    out, fx, waves = {"commit": COMMIT, "weights": SOURCES["yamnet.h5"][1], "inputs": {}}, HERE / "fixtures", inputs(d / "inputs")
    fx.mkdir(exist_ok=True)
    for f in fx.glob("*.f32"): f.unlink()
    for hop in (None, 2.0):
        frames, _, patches = keras(src, hop)
        for name, f in waves.items():
            if hop and name != "lena": continue
            key = name if hop is None else f"{name}@{hop:g}"
            x = np.fromfile(f, "<f4")
            scores, _, _ = frames(tf.constant(x))
            s, P = scores.numpy().astype("<f4"), patches(x).astype("<f4")
            assert s.shape[0] == P.shape[0], (key, s.shape, P.shape)
            s.tofile(fx / f"{key}.scores.f32")
            pick = {"first": 0, "last": len(P) - 1}
            keep = sorted({pick[k] for k in FEATURES[name].split(",")}) if hop is None and name in FEATURES else []
            if keep: P[keep].tofile(fx / f"{key}.patches.f32")
            mean = s.mean(0)
            out["inputs"][key] = {"samples": int(x.size), "sha256": hashlib.sha256(x.tobytes()).hexdigest(), "hop": hop or 0.48,
                                  "patches": int(s.shape[0]), "features": keep,
                                  "top": [[int(i), names[i], round(float(mean[i]), 4)] for i in np.argsort(-mean)[:3]]}
            print(f"  {key}: {x.size} samples, {s.shape[0]} patches; Keras: " + ", ".join(f"{names[i]} {mean[i]:.3f}" for i in np.argsort(-mean)[:3]))
    (fx / "reference.json").write_text(json.dumps(out, indent=1) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--verify", action="store_true", help="both graphs against the Keras network on upstream's patches")
    ap.add_argument("--reference", action="store_true", help="the Keras model's scores and patches of the reference inputs, into fixtures/")
    args = ap.parse_args()
    d = Path(args.out_dir or Path(os.environ.get("AUDIO_NEURAL_CACHE") or Path.home() / ".cache" / "audiojs" / "neural") / "yamnet")
    src = d / "src"
    fetch(src)
    import onnx
    for half, name in ((False, "yamnet.onnx"), (True, "yamnet.fp16.onnx")):
        onnx.save(build(src / "yamnet.h5", half), str(d / name))
        print(f"wrote {d / name} ({(d / name).stat().st_size:,} bytes, SHA-256 {hashlib.sha256((d / name).read_bytes()).hexdigest()})")
    classes(src)
    print(f"wrote {HERE / 'classes.js'}")
    if args.verify: verify(d, src)
    if args.reference: reference(d, src)


if __name__ == "__main__":
    main()
