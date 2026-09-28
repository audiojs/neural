# Scores of the benchmark outputs (scripts/bench.js, bench-ref.py), mean over clips, with mir_eval 0.8:
#   melody.evaluate: raw pitch and raw chroma accuracy (within 50 cents, chroma folded to one octave),
#   voicing recall and false alarm, overall accuracy; estimates resampled to the reference times.
#   A method with voicing reports f0 0 where unvoiced; `voicing` beside f0 (ours-raw, PESTO's
#   confidence) is passed as est_voicing, so its raw pitch counts every frame (Salamon et al., IEEE SPM
#   2014 define raw pitch accuracy regardless of voicing).
#   transcription (Vocadito notes, both annotators): onset within 50 ms (COn), + pitch within 50 cents
#   (COnP), + offset within 20% of the note or 50 ms (COnPOff), F-measure.
# python scripts/bench-score.py [methods] [sets]
import os, sys, json
import numpy as np
import mir_eval

OUT = os.environ.get('NP_BENCH', os.path.expanduser('~/.cache/audiojs/data/neural-pitch-bench'))
V = os.path.expanduser('~/.cache/audiojs/data/vocadito/Annotations/Notes')
CONDS = ['clean', 'pink20', 'pink10', 'pink0', 'reverb']


def melody(s, m, c):
    meta = json.load(open(os.path.join(OUT, s, 'meta.json')))
    rows = []
    for k in meta:
        f = os.path.join(OUT, s, 'out', m, c, k + '.json')
        if not os.path.exists(f): return None
        e = json.load(open(f))
        r = np.loadtxt(os.path.join(OUT, s, 'ref', k + '.csv'), delimiter=',', ndmin=2)
        et, ef = np.array(e['times']), np.array(e['f0'])
        ev = e.get('voicing')
        if ev is not None: ev = (np.array(ev) >= 0.5).astype(float)
        elif m == 'pesto': ev = np.ones_like(ef)
        sc = mir_eval.melody.evaluate(r[:, 0], r[:, 1], et, np.abs(ef), est_voicing=ev)
        rows.append([sc['Raw Pitch Accuracy'], sc['Raw Chroma Accuracy'], sc['Voicing Recall'], sc['Voicing False Alarm'], sc['Overall Accuracy']])
    return np.nanmean(np.array(rows, float), 0)


def notes(m, c):
    S = {'COn': [], 'COnP': [], 'COnPOff': []}
    for i in range(1, 41):
        f = os.path.join(OUT, 'vocadito', 'out', m, c, f'vocadito_{i}.json')
        if not os.path.exists(f): return None
        est = json.load(open(f))
        ei = np.array([[n['time'], n['time'] + n['duration']] for n in est]).reshape(-1, 2)
        ep = np.array([n['freq'] for n in est])
        for a in ('A1', 'A2'):
            d = np.loadtxt(os.path.join(V, f'vocadito_{i}_notes{a}.csv'), delimiter=',', ndmin=2)
            ri, rp = np.c_[d[:, 0], d[:, 0] + d[:, 2]], d[:, 1]
            S['COn'].append(mir_eval.transcription.onset_precision_recall_f1(ri, ei)[2])
            S['COnP'].append(mir_eval.transcription.precision_recall_f1_overlap(ri, rp, ei, ep, offset_ratio=None)[2])
            S['COnPOff'].append(mir_eval.transcription.precision_recall_f1_overlap(ri, rp, ei, ep)[2])
    return [np.mean(S[k]) for k in ('COn', 'COnP', 'COnPOff')]


if __name__ == '__main__':
    methods = sys.argv[1].split(',') if len(sys.argv) > 1 else ['pyin', 'ours', 'librosa', 'pesto', 'ours-raw']
    sets = sys.argv[2].split(',') if len(sys.argv) > 2 else ['vocadito', 'mdb-stem-synth', 'mir-1k']
    res = {}
    for s in sets:
        print(f'\n### {s}\n\n| condition | method | RPA | RCA | VR | VFA | OA |\n|---|---|---|---|---|---|---|')
        for c in CONDS:
            for m in methods:
                r = melody(s, m, c)
                if r is None: continue
                res[f'{s}/{c}/{m}'] = r.tolist()
                vr = '–' if m == 'pesto' else f'{r[2]:.3f}'
                vfa = '–' if m == 'pesto' else f'{r[3]:.3f}'
                oa = '–' if m == 'pesto' else f'{r[4]:.3f}'
                print(f'| {c} | {m} | {r[0]:.3f} | {r[1]:.3f} | {vr} | {vfa} | {oa} |')
    nm = [m for m in ('pyin-notes', 'ours-notes') if os.path.isdir(os.path.join(OUT, 'vocadito', 'out', m))]
    if nm:
        print('\n### vocadito notes (F)\n\n| condition | method | COn | COnP | COnPOff |\n|---|---|---|---|---|')
        for c in CONDS:
            for m in nm:
                r = notes(m, c)
                if r is None: continue
                res[f'notes/{c}/{m}'] = r
                print(f'| {c} | {m} | {r[0]:.3f} | {r[1]:.3f} | {r[2]:.3f} |')
    json.dump(res, open(os.path.join(OUT, 'scores.json'), 'w'), indent=1)
