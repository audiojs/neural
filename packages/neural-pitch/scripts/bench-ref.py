# Reference methods on the benchmark audio (scripts/bench-prep.py): librosa.pyin, and PESTO (Riou et
# al., ISMIR 2023) for measurement only, from its own environment (LGPL-3.0, not a dependency):
#   python scripts/bench-ref.py librosa [set ...]
#   <pesto-venv>/bin/python scripts/bench-ref.py pesto [set ...]      # pip install pesto-pitch==2.0.1
# Writes <set>/out/<method>/<cond>/<clip>.json: { times, f0 (Hz, 0 unvoiced) } and, for PESTO, its
# confidence; PESTO decides no voicing.
import os, sys, json, time
import numpy as np

OUT = os.environ.get('NP_BENCH', os.path.expanduser('~/.cache/audiojs/data/neural-pitch-bench'))
CONDS = ['clean', 'pink20', 'pink10', 'pink0', 'reverb']


def run_librosa(x, fs):
    import librosa
    n = 2 ** int(np.ceil(np.log2(2 * fs / 50)))      # as @audio/pitch-pyin frames: 2048 at 44.1 kHz
    f0, vf, vp = librosa.pyin(x, fmin=50, fmax=2000, sr=fs, frame_length=n)
    t = librosa.times_like(f0, sr=fs, hop_length=n // 4)
    return {'times': t.tolist(), 'f0': np.nan_to_num(f0).tolist()}


def run_pesto(x, fs):
    import torch, pesto
    # on the GPU (MPS) when there is one: the same output as on the CPU within 0.01 cent, 30 times sooner;
    # a clip in 10 s chunks, since a whole one needs over 4 GB
    dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
    with torch.inference_mode():
        t, f, c, _ = pesto.predict(torch.from_numpy(x).to(dev), fs, step_size=10.0, model_name='mir-1k_g7', convert_to_freq=True,
                                   num_chunks=max(1, int(np.ceil(x.size / fs / 10))))
    return {'times': (t.cpu().numpy() / 1000).tolist(), 'f0': f.cpu().numpy().tolist(), 'confidence': c.cpu().numpy().tolist()}


if __name__ == '__main__':
    method, sets = sys.argv[1], sys.argv[2:] or ['vocadito', 'mdb-stem-synth', 'mir-1k']
    run = {'librosa': run_librosa, 'pesto': run_pesto}[method]
    if method == 'pesto':
        import torch
        torch.set_num_threads(4)
        if torch.backends.mps.is_available(): torch.mps.set_per_process_memory_fraction(min(1.0, 3e9 / torch.mps.recommended_max_memory()))
    for s in sets:
        meta = json.load(open(os.path.join(OUT, s, 'meta.json')))
        for c in CONDS:
            d = os.path.join(OUT, s, 'out', method, c)
            os.makedirs(d, exist_ok=True)
            t0, dur = time.time(), 0
            for k, fs in meta.items():
                f = os.path.join(d, k + '.json')
                if os.path.exists(f): continue
                x = np.fromfile(os.path.join(OUT, s, c, k + '.f32'), dtype=np.float32)
                dur += x.size / fs
                json.dump(run(x, fs), open(f, 'w'))
            print(method, s, c, f'{time.time() - t0:.0f} s for {dur:.0f} s', flush=True)
