"""Reference outputs for test.js: Spotify's basic-pitch (Python) on audio files, written to
$AUDIO_NEURAL_CACHE or ~/.cache/audiojs/neural, under basic-pitch/:

    <name>.src.f32      the file as basic-pitch reads it: mono float32 at json.rate
    <name>.22k.f32      librosa.load(sr=22050) of it, the model input before windowing
    <name>.note.f32     unwrapped posteriors, frame-major float32, T x 88
    <name>.onset.f32    T x 88
    <name>.contour.f32  T x 264
    <name>.json         { rate, frames, variants: [{ opts, notes }] }, notes as predict()
                        returns them: [start_s, end_s, midi, amplitude, bends]

The first variant is inference.predict() end to end; the others run
note_creation.model_output_to_notes on copies of the same posteriors, with predict()'s
conversion of the minimum note length to frames. basic-pitch needs no TensorFlow: with
onnxruntime alone it runs saved_models/icassp_2022/nmp.onnx.

    pip install --no-deps basic-pitch  # its TensorFlow pins skip; the ONNX path needs only:
    pip install librosa mir_eval pretty_midi 'resampy<0.4.3' 'setuptools<81' onnxruntime
    python scripts/reference.py vocadito_10.wav mix.wav
"""
import json
import os
import pathlib
import sys

import librosa
import numpy as np
from basic_pitch import ICASSP_2022_MODEL_PATH, inference, note_creation
from basic_pitch.constants import AUDIO_SAMPLE_RATE, FFT_HOP

# JS option names (index.d.ts) -> model_output_to_notes arguments
VARIANTS = [
    {},
    {"onsetThreshold": 0.3, "frameThreshold": 0.2},
    {"onsetThreshold": 0.7, "frameThreshold": 0.5, "minDuration": 0.05},
    {"minFreq": 80, "maxFreq": 1000},
    {"minFreq": 130},
    {"melodiaTrick": False},
    {"inferOnsets": False},
]


def notes(output, opts):
    ms = opts.get("minDuration", inference.DEFAULT_MINIMUM_NOTE_LENGTH_MS / 1000) * 1000
    _, events = note_creation.model_output_to_notes(
        {k: v.copy() for k, v in output.items()},
        onset_thresh=opts.get("onsetThreshold", inference.DEFAULT_ONSET_THRESHOLD),
        frame_thresh=opts.get("frameThreshold", inference.DEFAULT_FRAME_THRESHOLD),
        infer_onsets=opts.get("inferOnsets", True),
        min_note_len=int(np.round(ms / 1000 * (AUDIO_SAMPLE_RATE / FFT_HOP))),
        min_freq=opts.get("minFreq"),
        max_freq=opts.get("maxFreq"),
        melodia_trick=opts.get("melodiaTrick", True),
    )
    return events


def plain(events):
    return [[float(s), float(e), int(p), float(a), [int(b) for b in bends]] for s, e, p, a, bends in events]


def main(paths):
    out = pathlib.Path(os.environ.get("AUDIO_NEURAL_CACHE") or pathlib.Path.home() / ".cache/audiojs/neural") / "basic-pitch"
    out.mkdir(parents=True, exist_ok=True)
    model = inference.Model(ICASSP_2022_MODEL_PATH)
    for path in map(pathlib.Path, paths):
        name = path.stem
        src, rate = librosa.load(str(path), sr=None, mono=True)
        audio, _ = librosa.load(str(path), sr=AUDIO_SAMPLE_RATE, mono=True)
        output, _, events = inference.predict(path, model)
        src.astype(np.float32).tofile(out / f"{name}.src.f32")
        audio.astype(np.float32).tofile(out / f"{name}.22k.f32")
        for k in ("note", "onset", "contour"):
            np.ascontiguousarray(output[k], dtype=np.float32).tofile(out / f"{name}.{k}.f32")
        variants = [{"opts": {}, "notes": plain(events)}]
        assert plain(notes(output, {})) == variants[0]["notes"], "predict() and model_output_to_notes disagree"
        variants += [{"opts": o, "notes": plain(notes(output, o))} for o in VARIANTS[1:]]
        manifest = {"rate": int(rate), "frames": int(output["note"].shape[0]), "variants": variants}
        (out / f"{name}.json").write_text(json.dumps(manifest))
        print(f"{name}: {manifest['frames']} frames, {len(events)} notes -> {out}/{name}.*")


if __name__ == "__main__":
    main(sys.argv[1:])
