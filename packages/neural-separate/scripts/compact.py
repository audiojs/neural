#!/usr/bin/env python3
"""Compact an exported separation graph: its weights stored in fewer bits, its compute left in float32.

    pip install onnx onnxruntime numpy
    python3 compact.py --model scnet-large --verify             # scnet-large.int8.onnx
    python3 compact.py --model tiger --as fp16 --verify         # tiger.fp16.onnx

Reads <dir>/<model>.onnx (an export-*.py output; <dir> the neural cache's <model>/) and writes <model>.<as>.onnx beside
it. The weights are the initializers a Conv, ConvTranspose, MatMul, Gemm, LSTM or GRU takes as its weight, of over 1024
values (biases, norms and constants stay float32):
  int8   stored as int8 with a float32 scale per output channel (symmetric, the channel's largest magnitude to 127; an
         LSTM's or GRU's per gate row of each direction), Cast and multiplied by its scale in the graph; all of them when
         their rounding together costs --budget dB under the output or less (40; 0: no bound) on the calibration mix,
         else int8 for the weights cheapest per byte saved (each rounded alone, the rest float32) while their summed
         cost stays within the budget, float16 for the rest (separate weights' costs add: the sum predicts the whole
         to a fraction of a dB)
  fp16   stored as float16, Cast to float32 in the graph
  fp32   as they are
onnxruntime folds the Cast and the product at load (both of constants): the session holds float32 weights and computes
as the float32 graph does, on every execution provider. DequantizeLinear would not: onnxruntime 1.30 fuses it with the
MatMul it feeds into MatMulNBits, whose default accuracy level quantizes the activations to int8 on a CPU (MRX's int8
graph 38.4 dB from float32 so, 40.4 dB as stored here).

Every variant is also folded and named short, neither of which changes a value. With each input's shape fixed (SCNet's
and TIGER's graphs; MRX's takes any length and is left as it is), every value computable from the initializers and the
shapes alone is computed once and stored as the graph computes it (SCNet's DFT matrices, made in float64 from a Range,
which onnxruntime-web's WebGPU session neither places nor folds), but for a fill (ConstantOfShape: SCNet's LSTM zero
states), kept with its shape stored. Identical initializers are stored once; node and value names become base-36
counters (TIGER's 56,652 nodes carried 11.7 MB of names), the graph's inputs and outputs keep theirs.

--verify runs the variant and the float32 graph on the calibration mix (default <dir>/test.f32, scripts/reference.py's;
--calibrate another: planar float32 stereo at 44.1 kHz), cut and transformed as the package does it, and prints each
output's largest difference over its peak and its SNR against the float32 graph's (MRX's masks as the spectra they
mask).
"""
import argparse
import hashlib
import os
import sys
from pathlib import Path

import numpy as np

WEIGHT = {"Conv": (1,), "ConvTranspose": (1,), "MatMul": (1,), "Gemm": (1,), "LSTM": (1, 2), "GRU": (1, 2)}
MIN, SMALL = 1024, 1024


def users_of(g):
    users = {}
    for n in g.node:
        for k, name in enumerate(n.input): users.setdefault(name, []).append((n, k))
    return users


def weights(g):
    """{name: node} of the float32 initializers of over MIN values that every user takes as its weight."""
    from onnx import TensorProto

    users, out = users_of(g), {}
    for i in g.initializer:
        uses = users.get(i.name, [])
        if i.data_type != TensorProto.FLOAT or int(np.prod(i.dims)) <= MIN or not uses: continue
        if all(n.op_type in WEIGHT and k in WEIGHT[n.op_type] for n, k in uses) and len({n.op_type for n, _ in uses}) == 1:
            out[i.name] = uses[0][0]
    return out


def q8(node, w):
    """Symmetric int8 per output channel (an LSTM's or GRU's per gate row of each direction), the channel's largest
    magnitude to 127: q in w's layout and its scale, shaped to broadcast against it."""
    keep = (0, 1) if node.op_type in ("LSTM", "GRU") else ({"ConvTranspose": 1, "MatMul": w.ndim - 1}.get(node.op_type, 0),)
    if node.op_type == "Gemm": keep = (0 if any(x.name == "transB" and x.i for x in node.attribute) else 1,)
    scale = np.maximum(np.abs(w).max(axis=tuple(k for k in range(w.ndim) if k not in keep), keepdims=True), 1e-30) / 127
    return np.clip(np.round(w / scale), -127, 127).astype(np.int8), scale.astype(np.float32)


def rounded(node, w):
    """w through q8 and back, in float32: what the int8 graph computes with."""
    q, scale = q8(node, w)
    return q.astype(np.float32) * scale


def prepend(g, nodes):
    body = list(g.node)
    del g.node[:]
    g.node.extend(nodes + body)


def store(m, kinds):
    """Each weight as kinds[name] says: 'fp16' (Cast), 'int8' (Cast, Mul by its scale), else float32."""
    from onnx import helper, numpy_helper, TensorProto

    g = m.graph
    inits = {i.name: i for i in g.initializer}
    nodes = []
    for name, node in weights(g).items():
        kind = kinds.get(name)
        if kind not in ("fp16", "int8"): continue
        w = numpy_helper.to_array(inits[name])
        g.initializer.remove(inits[name])
        if kind == "fp16":
            g.initializer.append(numpy_helper.from_array(w.astype(np.float16), name + "_h"))
            nodes.append(helper.make_node("Cast", [name + "_h"], [name], to=TensorProto.FLOAT))
            continue
        q, scale = q8(node, w)
        g.initializer.extend([numpy_helper.from_array(q, name + "_q"), numpy_helper.from_array(scale, name + "_s")])
        nodes += [helper.make_node("Cast", [name + "_q"], [name + "_c"], to=TensorProto.FLOAT), helper.make_node("Mul", [name + "_c", name + "_s"], [name])]
    prepend(g, nodes)
    return m


def static(g):
    return all(d.dim_value > 0 for i in g.input for d in i.type.tensor_type.shape.dim)


def fold(m):
    """Values computable from the initializers and the input shapes alone: stored when small, else made at load."""
    import onnx
    import onnxruntime as ort
    from onnx import helper, numpy_helper, shape_inference

    g = m.graph
    if not static(g): return dedupe(m)
    const, cnodes = {i.name for i in g.initializer}, []
    for n in g.node:
        ins = [x for x in n.input if x]
        if n.op_type in ("Constant", "Shape") or (ins and all(x in const for x in ins) and not n.op_type.startswith("Random")):
            cnodes.append(n)
            const.update(n.output)
    computed = [o for n in cnodes for o in n.output]
    types = {v.name: v.type.tensor_type.elem_type for v in shape_inference.infer_shapes(m).graph.value_info}
    for n in cnodes:
        if n.op_type == "Constant": types[n.output[0]] = next(a.t.data_type for a in n.attribute if a.name == "value")
    probe = onnx.ModelProto()
    probe.CopyFrom(m)
    probe.graph.output.extend(helper.make_tensor_value_info(x, types[x], None) for x in computed)
    s = session(probe.SerializeToString())
    shapes = {i.name: [d.dim_value for d in i.type.tensor_type.shape.dim] for i in g.input}
    vals = dict(zip(computed, s.run(computed, {k: np.zeros(v, np.float32) for k, v in shapes.items()})))
    # the same from another input, or the graph holds a shape that depends on the data: left as it is
    other = s.run(computed, {k: np.random.default_rng(0).standard_normal(v).astype(np.float32) for k, v in shapes.items()})
    if any(not np.array_equal(vals[x], y, equal_nan=vals[x].dtype.kind == "f") for x, y in zip(computed, other)): return dedupe(m)
    # what the rest of the graph reads, stored; a fill (ConstantOfShape) kept, its shape stored
    producer = {o: n for n in cnodes for o in n.output}
    inner = {id(n) for n in cnodes}
    keep, stored = set(), {}
    want = [x for n in g.node if id(n) not in inner for x in n.input] + [o.name for o in g.output]
    while want:
        x = want.pop()
        if x not in producer or x in stored or id(producer[x]) in keep: continue
        if producer[x].op_type == "ConstantOfShape" and vals[x].size > SMALL:
            keep.add(id(producer[x]))
            want += list(producer[x].input)
        else: stored[x] = vals[x]
    body = [n for n in g.node if id(n) not in inner or id(n) in keep]
    del g.node[:]
    g.node.extend(body)
    g.initializer.extend(numpy_helper.from_array(np.asarray(v), x) for x, v in stored.items())
    return dedupe(m)


def dedupe(m):
    """Identical initializers stored once; those no node reads dropped."""
    from onnx import numpy_helper

    g = m.graph
    seen, alias = {}, {}
    for i in list(g.initializer):
        a = numpy_helper.to_array(i)
        key = hashlib.sha256(str((a.dtype, a.shape)).encode() + a.tobytes()).hexdigest()
        if key in seen:
            alias[i.name] = seen[key]
            g.initializer.remove(i)
        else: seen[key] = i.name
    for n in g.node:
        for k, x in enumerate(n.input):
            if x in alias: n.input[k] = alias[x]
    read = {x for n in g.node for x in n.input} | {o.name for o in g.output}
    for i in [i for i in g.initializer if i.name not in read]: g.initializer.remove(i)
    return m


def shorten(m):
    """Node and value names as base-36 counters; the inputs and outputs keep theirs."""
    g = m.graph
    keep = {v.name for v in list(g.input) + list(g.output)}
    names = {}

    def base36(k):
        s = ""
        while True:
            s, k = "0123456789abcdefghijklmnopqrstuvwxyz"[k % 36] + s, k // 36
            if not k: return s

    def short(n):
        if n == "" or n in keep: return n
        if n not in names: names[n] = "_" + base36(len(names))
        return names[n]

    for i in g.initializer: i.name = short(i.name)
    for k, n in enumerate(g.node):
        for j, x in enumerate(n.input): n.input[j] = short(x)
        for j, x in enumerate(n.output): n.output[j] = short(x)
        n.name, n.doc_string = base36(k), ""  # unique: onnxruntime-web's partitioner reports nodes by name
    del g.value_info[:]
    m.doc_string = ""
    return m


# ------------------------------------------------------------------ calibration

def session(model, fold=False):
    import onnxruntime as ort

    o = ort.SessionOptions()
    o.enable_cpu_mem_arena = False
    o.enable_mem_pattern = False
    o.log_severity_level = 3  # the overridable weights' warnings
    if not fold: o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(model if isinstance(model, bytes) else str(model), o, providers=["CPUExecutionProvider"])


def feeds(g, mix):
    """The graph's inputs from a mixture [2, samples] at 44.1 kHz, as separate.js cuts and transforms it: SCNet one
    11 s segment (no window, 1/sqrt(n)), TIGER one 12 s segment of the first channel (Hann), MRX 10 s of each
    channel at three resolutions (Hann, 1/sqrt(n), magnitudes)."""
    ins = {i.name: [d.dim_value or None for d in i.type.tensor_type.shape.dim] for i in g.input}

    def spec(x, n, hop, win, scale):
        pad = np.pad(x, ((0, 0), (n // 2, n // 2)), mode="reflect")
        T = 1 + (pad.shape[1] - n) // hop
        fr = np.stack([pad[:, t * hop:t * hop + n] for t in range(T)], 1) * win
        return np.fft.rfft(fr, axis=-1).transpose(0, 2, 1) * scale  # [C, F, T]

    hann = lambda n: 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)
    if list(ins) == ["mix_spec"]:
        _, C2, F, T = ins["mix_spec"]
        n = 2 * (F - 1)
        tiger = n == 2048
        x = mix[:C2 // 2, :(T - 1) * (n // 4)]
        X = spec(x, n, n // 4, hann(n) if tiger else 1.0, 1.0 if tiger else n ** -0.5)[..., :T]
        X = np.pad(X, ((0, 0), (0, 0), (0, T - X.shape[2])))
        return {"mix_spec": np.stack([X.real, X.imag], 1).reshape(1, C2, F, T).astype(np.float32)}
    x = mix[:, :441000]
    return {k: np.abs(spec(x, int(k.split("_")[1]), 256, hann(int(k.split("_")[1])), int(k.split("_")[1]) ** -0.5)).astype(np.float32) for k in ins}


def outputs(s, feed):
    """The outputs as the stems see them: a mask times the magnitudes it masks."""
    ys = dict(zip([o.name for o in s.get_outputs()], s.run(None, feed)))
    return {k: (y * feed["mag_" + k[5:]][:, None] if k.startswith("mask_") else y).astype(np.float64) for k, y in ys.items()}


def noise(y, y0):
    """Error power over signal power, the worst output's."""
    return max(((y[k] - y0[k]) ** 2).sum() / max((y0[k] ** 2).sum(), 1e-300) for k in y0)


def costs(m, feed, W):
    """The cost at int8 of all the weights rounded together, and a function giving each one's alone (the others
    float32): noise power over output power. The weights are fed as overridable initializers, one session for all."""
    import onnx
    from onnx import helper, numpy_helper

    inits = {i.name: i for i in m.graph.initializer}
    probe = onnx.ModelProto()
    probe.CopyFrom(m)
    probe.graph.input.extend(helper.make_tensor_value_info(n, inits[n].data_type, list(inits[n].dims)) for n in W)
    s = session(probe.SerializeToString())
    y0 = outputs(s, feed)
    q = {n: rounded(node, numpy_helper.to_array(inits[n])) for n, node in W.items()}
    return noise(outputs(s, {**feed, **q}), y0), lambda n: noise(outputs(s, {**feed, n: q[n]}), y0)


def int8(m, feed, budget):
    """{weight: 'int8' | 'fp16'}: every weight int8 when their rounding together costs budget dB under the output or
    less (no budget: always); else int8 for the weights cheapest per byte saved, each rounded alone, while their summed
    cost stays within the budget (costs of separate weights add), float16 for the rest."""
    W = weights(m.graph)
    if budget is None: return {n: "int8" for n in W}
    whole, alone = costs(m, feed, W)
    print(f"  every weight int8: SNR {-10 * np.log10(whole):.1f} dB against float32")
    if whole <= 10 ** (-budget / 10): return {n: "int8" for n in W}
    size = {i.name: int(np.prod(i.dims)) for i in m.graph.initializer}
    cost = {}
    for k, n in enumerate(W):
        cost[n] = alone(n)
        print(f"\r  cost {k + 1}/{len(W)}", end="", file=sys.stderr)
    print(file=sys.stderr)
    kinds, total = {}, 0.0
    for n in sorted(W, key=lambda n: cost[n] / size[n]):
        if total + cost[n] <= 10 ** (-budget / 10): kinds[n], total = "int8", total + cost[n]
        else: kinds[n] = "fp16"
    for n in sorted((n for n in kinds if kinds[n] == "fp16"), key=lambda n: -cost[n]):
        print(f"  float16: {W[n].name or n}, {size[n]} values, alone at int8 {-10 * np.log10(max(cost[n], 1e-30)):.1f} dB")
    print(f"  int8: {sum(k == 'int8' for k in kinds.values())} of {len(kinds)} weights, "
          f"{sum(size[n] for n in kinds if kinds[n] == 'int8') / sum(size[n] for n in W):.1%} of their values; predicted SNR {-10 * np.log10(max(total, 1e-30)):.1f} dB")
    return kinds


def compact(path, out, kind, calibration=None, budget=40.0):
    import onnx

    m = onnx.load(str(path))
    # the trained weights, chosen before folding (never a folded constant)
    kinds = int8(m, feeds(m.graph, calibration), budget) if kind == "int8" else {n: kind for n in weights(m.graph)}
    m = shorten(store(fold(m), kinds))
    m.metadata_props.append(onnx.StringStringEntryProto(key="compact", value=kind + (f", budget {budget:g} dB" if kind == "int8" and budget else "")))
    onnx.checker.check_model(m)
    onnx.save(m, str(out))
    return out


def verify(path, base, mix):
    import onnx

    a, b = session(base, True), session(path, True)
    feed = feeds(onnx.load(str(base), load_external_data=False).graph, mix)
    y0, y = outputs(a, feed), outputs(b, feed)
    for k in y0:
        d = y[k] - y0[k]
        rel = np.abs(d).max() / np.abs(y0[k]).max()
        print(f"  {path.name} {k}: max|diff|/max|y| = {rel:.2e}, SNR {10 * np.log10((y0[k] ** 2).sum() / max((d ** 2).sum(), 1e-300)):.1f} dB against float32")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="scnet-large | scnet | mrx | tiger (any <dir>/<model>.onnx)")
    ap.add_argument("--as", dest="kind", default="int8", choices=["fp32", "fp16", "int8"])
    ap.add_argument("--budget", type=float, default=40, help="int8: the rounding's cost, dB under the output (40); 0: every weight int8")
    ap.add_argument("--dir", default=None, help="default: $AUDIO_NEURAL_CACHE/<model> or ~/.cache/audiojs/neural/<model>")
    ap.add_argument("--out", default=None, help="default: <dir>/<model>.<as>.onnx")
    ap.add_argument("--calibrate", default=None, help="planar float32 stereo at 44.1 kHz (default <dir>/test.f32)")
    ap.add_argument("--verify", action="store_true", help="the variant against the float32 graph on the calibration mix")
    args = ap.parse_args()

    cache = Path(os.environ.get("AUDIO_NEURAL_CACHE") or Path.home() / ".cache" / "audiojs" / "neural")
    d = Path(args.dir) if args.dir else cache / args.model
    path = d / f"{args.model}.onnx"
    out = Path(args.out) if args.out else d / f"{args.model}.{args.kind}.onnx"
    cal = Path(args.calibrate) if args.calibrate else d / "test.f32"
    mix = np.fromfile(cal, np.float32).reshape(2, -1) if cal.exists() else None
    budget = args.budget or None
    if mix is None and ((args.kind == "int8" and budget) or args.verify): raise SystemExit(f"{cal}: no calibration mix (export with --verify writes test.f32)")
    compact(path, out, args.kind, mix, budget)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB, from {path.stat().st_size / 1e6:.2f})")
    if args.verify: verify(out, path, mix)


if __name__ == "__main__":
    sys.exit(main())
