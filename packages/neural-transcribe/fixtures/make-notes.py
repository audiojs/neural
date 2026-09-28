"""fixtures/notes.json: basic-pitch's note creation (note_creation.model_output_to_notes) on
synthetic posteriors that test.js regenerates bit for bit (posteriors() below, mirrored there):
24-bit LCG noise under 0.25 plus note events that exercise onset peaks, neighbour clearing,
gaps shorter and longer than the energy tolerance, the minimum length, the melodia trick,
inferred onsets, argmax ties, both ends of the keyboard and of the time axis.

    python fixtures/make-notes.py   # needs basic-pitch as scripts/reference.py installs it
"""
import json
import pathlib

import numpy as np
from basic_pitch import note_creation
from basic_pitch.constants import AUDIO_SAMPLE_RATE, FFT_HOP

T = 400
# key, first frame, end frame, note posterior, onset peak (0: none), gap (from, to) at 0.125, bend pattern in bins
EVENTS = [
    (39, 10, 40, 0.75, 0.875, None, [0]),
    (40, 20, 60, 0.625, 0.75, None, [0, 1, 1, 0]),
    (0, 50, 80, 0.75, 0.875, None, [-1, 0, 1, 2]),
    (87, 50, 90, 0.75, 0.875, None, [0, 1, 2, 1, -2]),
    (60, 100, 150, 0.75, 0.875, (115, 121), [0, 0, 1]),
    (62, 100, 170, 0.75, 0.875, (120, 135), [-1, 0]),
    (20, 180, 191, 0.75, 0.875, None, [0]),
    (25, 180, 192, 0.75, 0.875, None, [0]),
    (70, 200, 240, 0.9375, 0, None, [1, 0, -1]),
    (50, 250, 280, 0.625, 0, None, [0]),
    (55, 250, 280, 0.625, 0, None, [0]),
    (30, 330, 400, 0.75, 0.875, None, [0, 3, -3]),
]
VARIANTS = [
    {},
    {"onsetThreshold": 0.3, "frameThreshold": 0.26},
    {"onsetThreshold": 0.7, "frameThreshold": 0.5, "minDuration": 0.05},
    {"minFreq": 80, "maxFreq": 1000},
    {"minFreq": 30, "maxFreq": 4000},
    {"melodiaTrick": False},
    {"inferOnsets": False},
    {"onsetThreshold": 0},
]


def posteriors():
    s = 1

    def u():
        nonlocal s
        s = (s * 1664525 + 1013904223) % 2**32
        return (s >> 8) / 2**24

    note = np.array([[u() * 0.25 for _ in range(88)] for _ in range(T)], dtype=np.float32)
    onset = np.array([[u() * 0.25 for _ in range(88)] for _ in range(T)], dtype=np.float32)
    contour = np.array([[u() * 0.25 for _ in range(264)] for _ in range(T)], dtype=np.float32)
    for key, t0, t1, amp, peak, gap, bend in EVENTS:
        for t in range(t0, t1):
            note[t, key] = 0.125 if gap and gap[0] <= t < gap[1] else amp
            b = 3 * key + bend[(t - t0) % len(bend)]
            for d, v in ((-1, 0.5), (1, 0.5), (0, 0.875)):
                if 0 <= b + d < 264:
                    contour[t, b + d] = v
        if peak:
            onset[t0, key] = peak
    return {"note": note, "onset": onset, "contour": contour}


def main():
    out = []
    for opts in VARIANTS:
        ms = opts.get("minDuration", 0.1277) * 1000
        _, events = note_creation.model_output_to_notes(
            {k: v.copy() for k, v in posteriors().items()},
            onset_thresh=opts.get("onsetThreshold", 0.5),
            frame_thresh=opts.get("frameThreshold", 0.3),
            infer_onsets=opts.get("inferOnsets", True),
            min_note_len=int(np.round(ms / 1000 * (AUDIO_SAMPLE_RATE / FFT_HOP))),
            min_freq=opts.get("minFreq"),
            max_freq=opts.get("maxFreq"),
            melodia_trick=opts.get("melodiaTrick", True),
        )
        notes = [[float(s), float(e), int(p), float(a), [int(b) for b in bends]] for s, e, p, a, bends in events]
        out.append({"opts": opts, "notes": notes})
        print(opts, len(notes), "notes")
    (pathlib.Path(__file__).parent / "notes.json").write_text(json.dumps(out))


if __name__ == "__main__":
    main()
