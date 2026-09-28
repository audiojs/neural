#!/usr/bin/env python3
# wiener.json: open-unmix-pytorch's own wiener() (openunmix/filtering.py, norbert's algorithm)
# on small seeded inputs, float64 as upstream recommends. test.js runs wienerFilter on the same
# inputs and compares.
#
#   pip install torch openunmix && python3 fixtures/make-wiener.py
import json
from pathlib import Path

import torch
from openunmix.filtering import wiener

torch.manual_seed(0)
T, F, C, S = 6, 9, 2, 3
mix = torch.randn(T, F, C, 2, dtype=torch.float64)        # (frames, bins, channels, re/im)
mags = torch.rand(T, F, C, S, dtype=torch.float64) * 2   # (frames, bins, channels, sources)
cases = {}
for name, kw in {
    "em1": dict(iterations=1),
    "em2_softmask_residual": dict(iterations=2, softmask=True, residual=True),
    "raw": dict(iterations=0),
}.items():
    y = wiener(mags.clone(), mix.clone(), **kw)             # (frames, bins, channels, re/im, sources)
    cases[name] = {"opts": kw, "y": y.permute(4, 2, 0, 1, 3).tolist()}  # [source][channel][frame][bin][re/im]
out = {"mix": mix.permute(2, 0, 1, 3).tolist(), "mags": mags.permute(3, 2, 0, 1).tolist(), "cases": cases}
Path(__file__).with_name("wiener.json").write_text(json.dumps(out))
print("wrote wiener.json")
