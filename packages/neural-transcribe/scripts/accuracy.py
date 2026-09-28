"""The README's accuracy table, step 2: note precision, recall and F1 by mir_eval (onset within
50 ms, pitch within 50 cents; "+ offset" also needs the offset within 20% of the note or 50 ms)
for the estimates scripts/accuracy.mjs wrote. vocadito scores are means over its tracks, per
annotator.

    pip install mir_eval
    python scripts/accuracy.py
"""
import csv
import json
import pathlib
import warnings

import mir_eval
import numpy as np

warnings.filterwarnings("ignore")
DATA = pathlib.Path.home() / ".cache/audiojs/data"


def score(ref, notes, offset):
    if not notes:
        return 0.0, 0.0, 0.0
    return mir_eval.transcription.precision_recall_f1_overlap(
        *ref, np.array([n[:2] for n in notes]), np.array([n[2] for n in notes]),
        onset_tolerance=0.05, pitch_tolerance=50.0, offset_ratio=0.2 if offset else None, offset_min_tolerance=0.05,
    )[:3]


def synthetic(name):
    notes = json.loads((DATA / "transcribe" / f"{name}.json").read_text())["notes"]
    return np.array([[n["time"], n["time"] + n["duration"]] for n in notes]), np.array([440 * 2 ** ((n["midi"] - 69) / 12) for n in notes])


def vocadito(name, annotator):
    rows = [list(map(float, r)) for r in csv.reader(open(DATA / "vocadito/Annotations/Notes" / f"{name}_notes{annotator}.csv")) if r]
    return np.array([[r[0], r[0] + r[2]] for r in rows]), np.array([r[1] for r in rows])


def line(label, n, scores):
    (p, r, f), (_, _, fo) = scores
    return f"{label:28s} {n:5d}  {p:.2f} / {r:.2f} / {f:.2f}  {fo:.2f}"


est = json.loads((DATA / "transcribe" / "estimates.json").read_text())
print(f"{'':28s} {'notes':>5s}  P / R / F1            + offset")
for name, e in est.items():
    if name.startswith("vocadito"):
        continue
    ref = synthetic(name)
    for m in ("neural", "mir"):
        print(line(f"{name} {m}", len(ref[0]), [score(ref, e[m], o) for o in (False, True)]))
tracks = [n for n in est if n.startswith("vocadito")]
for annotator in ("A1", "A2") if tracks else ():
    refs = {n: vocadito(n, annotator) for n in tracks}
    for m in ("neural", "mir"):
        s = [np.mean([score(refs[n], est[n][m], o) for n in tracks], axis=0) for o in (False, True)]
        print(line(f"vocadito {annotator} ({len(tracks)}) {m}", sum(len(r[0]) for r in refs.values()), s))
