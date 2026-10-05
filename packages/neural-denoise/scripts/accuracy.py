# Scores for the Accuracy section, from the outputs scripts/accuracy.mjs writes.
#   python scripts/accuracy.py prepare                     the band-limited VoiceBank+DEMAND inputs and the MUSDB18 previews
#   python scripts/accuracy.py SET SYSTEM [SYSTEM...]      SET: vbdemand | vbtrain, or either @RATE or -lp16k (noisy: the input)
#   python scripts/accuracy.py limits SET SYSTEM LIMITS    SYSTEM's unlimited output with the input mixed back, per limit;
#                                                          each paired against the last of LIMITS, 95% intervals
#   python scripts/accuracy.py clean SET SYSTEM [LIMITS]   SET: vbclean | vbtrain-clean: what speech with nothing to remove loses
#   python scripts/accuracy.py music SYSTEM [SYSTEM...]    level change per band of music, chords and songs
#   python scripts/accuracy.py rooms [SYSTEM...]
#   python scripts/accuracy.py presence SYSTEM             can the model's own removal tell music from noisy speech?
#   python scripts/accuracy.py guard-sets                  the music guard's labelled sets (scripts/accuracy.mjs guard)
# pip install numpy scipy pesq pystoi onnxruntime; DNSMOS: sig_bak_ovr.onnx from microsoft/DNS-Challenge at
# 82f1b17e77 (DNSMOS/DNSMOS/, CC BY 4.0) in $AUDIO_NEURAL_CACHE/dnsmos/ (default ~/.cache/audiojs/neural/dnsmos/).
# prepare needs ffmpeg (the MUSDB18 previews are AAC in MP4).
#
# VoiceBank+DEMAND, against the clean reference, all at 16 kHz (scipy.signal.resample_poly by the reduced ratio):
#   PESQ: ITU-T P.862.2 wideband MOS-LQO (python-pesq 0.0.4, the ITU reference C code)
#   STOI: Taal, Hendriks, Heusdens, Jensen, IEEE TASLP 19(7), 2011 (pystoi 0.4.1, extended=False)
#   SI-SDR: Le Roux, Wisdom, Erdogan, Hershey, ICASSP 2019, eq. 3, zero-mean signals, dB
# Both sets, no reference: DNSMOS P.835 (Reddy, Gopal, Cutler, ICASSP 2022), dnsmos_local.py's method at
# 82f1b17e77 (non-personalized): 9.01 s windows at a 1 s hop, shorter clips tiled, SIG/BAK/OVRL polynomial fits,
# averaged over windows. Rooms also: the noise floor, RMS of the quietest 500 ms at a 50 ms hop (ACX Check's
# measure), and the speech level, mean power of the loudest half of 50 ms frames, against the raw input.
# Clean speech, against the input, 10 ms frames: active within 35 dB of the 99th-percentile frame; word ends the last
# 100 ms before each pause (inactive for 100 ms or more, or the end); quiet syllables the other active frames 20 to
# 35 dB under the active level (the mean power of the active frames, as ITU-T P.56's active speech level); the level
# change of each class over its frames, and SI-SDR of the output against the input.
# Music, against the input: the level change per band (power from the whole-file FFT), SI-SDR (inf: the input itself).
# Presence: per file, what an unlimited model takes out of what stands 10 dB or more over its band's floor (the
# quietest 100 ms of the 20 s around, minimum statistics, Martin 2001), averaged over five bands (0.3, 1, 3, 8 kHz
# apart; 20 ms frames at a 10 ms hop): a guard would pass the input through where the model takes much of what stands over the noise.
import os, sys, glob, csv, json, subprocess, numpy as np, onnxruntime as ort
from math import gcd
from multiprocessing import Pool
from scipy.signal import resample_poly, firwin, fftconvolve
from scipy.io import wavfile
DATA = os.path.expanduser('~/.cache/audiojs/data')
CACHE = os.environ.get('AUDIO_NEURAL_CACHE') or os.path.expanduser('~/.cache/audiojs/neural')
P_SIG = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
P_BAK = np.poly1d([-0.13166888, 1.60915514, -0.39604546])
P_OVR = np.poly1d([-0.06766283, 1.11546468, 0.04602535])
_sess = None

def dnsmos(x16):
    global _sess
    if _sess is None:
        so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
        _sess = ort.InferenceSession(os.path.join(CACHE, 'dnsmos', 'sig_bak_ovr.onnx'), so)
    fs, need = 16000, int(9.01 * 16000)
    a = x16.astype(np.float64)
    while len(a) < need: a = np.append(a, a)
    s = []
    for i in range(int(np.floor(len(a) / fs) - 9.01) + 1):
        seg = a[int(i * fs): int((i + 9.01) * fs)]
        if len(seg) == need: s.append(_sess.run(None, {'input_1': seg.astype('float32')[None]})[0][0])
    s = np.array(s)
    return float(np.mean(P_SIG(s[:, 0]))), float(np.mean(P_BAK(s[:, 1]))), float(np.mean(P_OVR(s[:, 2])))

def sisdr(ref, est):
    if np.array_equal(ref, est): return float('inf')   # bit for bit: music passed untouched
    ref = ref - ref.mean(); est = est - est.mean()
    t = np.dot(est, ref) / np.dot(ref, ref) * ref
    return float(10 * np.log10(np.dot(t, t) / max(np.dot(est - t, est - t), 1e-30)))

def rwav(p): return wavfile.read(p)[1].astype(np.float64) / 32768
def rf32(p): return np.fromfile(p, dtype=np.float32).astype(np.float64)
def to16(x, rate): g = gcd(16000, rate); return resample_poly(x, 16000 // g, rate // g)

def vbset(setname):
    """SET → data dir, clean dir, input (a wav dir, or a float32 dir), output dir, rate"""
    base, _, rate = setname.partition('@'); base, _, lp = base.partition('-lp')
    test = not base.startswith('vbtrain'); d = os.path.join(DATA, 'vbdemand' if test else 'vbdemand-train')
    cdir, ndir = ('clean_testset_wav', 'noisy_testset_wav') if test else ('clean', 'noisy')
    if base in ('vbclean', 'vbtrain-clean'): return d, cdir, (cdir, 'wav'), 'out-clean', 48000
    if rate: return d, cdir, (f'rate{rate}', 'f32'), f'out@{rate}', int(rate)
    if lp: return d, cdir, (f'lp{lp}', 'f32'), f'out-lp{lp}', 48000
    return d, cdir, (ndir, 'wav'), 'out', 48000

def names(setname):
    d, cdir = vbset(setname)[:2]; return sorted(os.path.basename(p)[:-4] for p in glob.glob(f'{d}/{cdir}/*.wav'))

def pair(setname, system, name, limit=None):
    """(clean reference, input, output, rate); `limit` mixes the input back into an unlimited output"""
    d, cdir, (idir, ext), out, rate = vbset(setname)
    ref = rwav(f'{d}/{cdir}/{name}.wav'); inp = rwav(f'{d}/{idir}/{name}.wav') if ext == 'wav' else rf32(f'{d}/{idir}/{name}.f32')
    est = inp if system in ('noisy', 'raw') else rf32(f'{d}/{out}/{system}/{name}.f32')
    assert len(est) == len(inp), (system, name)
    if limit: a = 10 ** (-limit / 20); est = est * (1 - a) + inp * a
    return ref, inp, est, rate

def vb(args):
    from pesq import pesq
    from pystoi import stoi
    setname, system, name, limit = args
    ref, _, est, rate = pair(setname, system, name, limit)
    r, e = resample_poly(ref, 1, 3), to16(est, rate); n = min(len(r), len(e)); r, e = r[:n], e[:n]
    sig, bak, ovr = dnsmos(e)
    return dict(name=name, limit=limit or 0, pesq=pesq(16000, r, e, 'wb'), stoi=stoi(r, e, 16000, extended=False), sisdr=sisdr(r, e), sig=sig, bak=bak, ovrl=ovr)

def classes(x, fs):
    n = fs // 100; m = len(x) // n; lv = 10 * np.log10(np.sum(x[:m * n].reshape(m, n) ** 2, 1) + 1e-20)
    act = lv > np.percentile(lv, 99) - 35; A = 10 * np.log10(np.mean(10 ** (lv[act] / 10)))
    end = np.zeros(m, bool); i = 0
    while i < m:
        if act[i]: i += 1; continue
        j = i
        while j < m and not act[j]: j += 1
        if j - i >= 10 or j == m: end[max(0, i - 10):i] = act[max(0, i - 10):i]
        i = j
    return dict(ends=end, quiet=act & ~end & (lv >= A - 35) & (lv < A - 20), active=act), n, m

def change(x, y, mask, n, m):
    if not mask.any(): return float('nan')
    X, Y = x[:m * n].reshape(m, n)[mask], y[:m * n].reshape(m, n)[mask]
    return float(10 * np.log10((np.sum(Y ** 2) + 1e-20) / (np.sum(X ** 2) + 1e-20)))

def cleanrow(args):
    setname, system, name, limit = args
    _, x, y, fs = pair(setname, system, name, limit)
    c, n, m = classes(x, fs)
    return dict(name=name, limit=limit or 0, sisdr=sisdr(x, y), **{k: change(x, y, v, n, m) for k, v in c.items()})

BANDS = [(0, 250), (250, 1000), (1000, 4000), (4000, 8000), (8000, 16000), (16000, 22050)]
def musicrow(args):
    system, name = args
    d = os.path.join(DATA, 'repair', 'out-neural')
    x, y = rf32(f'{d}/raw/{name}.f32'), rf32(f'{d}/{system}/{name}.f32')
    X, Y = np.abs(np.fft.rfft(x)) ** 2, np.abs(np.fft.rfft(y)) ** 2; f = np.fft.rfftfreq(len(x), 1 / 44100)
    return dict(name=name, sisdr=sisdr(x, y), **{f'{a}-{b}': float(10 * np.log10((Y[(f >= a) & (f < b)].sum() + 1e-20) / (X[(f >= a) & (f < b)].sum() + 1e-20))) for a, b in BANDS})

def taken(x, y, fs, over=10):
    from scipy.signal import stft
    from scipy.ndimage import minimum_filter1d, uniform_filter1d
    f, _, X = stft(x, fs, nperseg=fs // 50, noverlap=fs // 100, boundary=None, padded=False)
    _, _, Y = stft(y, fs, nperseg=fs // 50, noverlap=fs // 100, boundary=None, padded=False)
    r = []
    for a, b in zip([0, 300, 1000, 3000, 8000], [300, 1000, 3000, 8000, fs / 2]):
        px, py = (np.abs(X[(f >= a) & (f < b)]) ** 2).sum(0), (np.abs(Y[(f >= a) & (f < b)]) ** 2).sum(0)
        m = uniform_filter1d(px, 10, mode='nearest'); m[m <= 1e-12 * max(m.max(), 1e-30)] = np.inf
        on = px > minimum_filter1d(m, 2001, mode='nearest') * 10 ** (over / 10)
        if on.any() and px[on].sum() > 0: r.append(10 * np.log10(py[on].sum() / px[on].sum() + 1e-30))
    return float(np.mean(r)) if r else 0.

def presence(args):
    kind, system, name = args
    if kind == 'music':
        d = os.path.join(DATA, 'repair', 'out-neural'); return kind, name, taken(rf32(f'{d}/raw/{name}.f32'), rf32(f'{d}/{system}/{name}.f32'), 44100)
    _, x, y, fs = pair(kind, system, name); return kind, name, taken(x, y, fs)

def floor(x, fs=48000):
    w, h = fs // 2, fs // 20
    return 10 * np.log10(max(min(np.mean(x[i:i + w] ** 2) for i in range(0, len(x) - w, h)), 1e-20))

def speech(x, fs=48000):
    h = fs // 20; p = np.sort([np.mean(x[i:i + h] ** 2) for i in range(0, len(x) - h, h)])
    return 10 * np.log10(np.mean(p[len(p) // 2:]) + 1e-20)

def prepare():
    """noisy inputs at 16 and 44.1 kHz (resample_poly), low-passed at 16 kHz (1023-tap Kaiser-windowed FIR, β 10,
    zero phase); the MUSDB18 test previews' mixtures (stream 0) as 44.1 kHz mono float32"""
    h = firwin(1023, 16000, fs=48000, window=('kaiser', 10))
    for d, ndir in [('vbdemand', 'noisy_testset_wav'), ('vbdemand-train', 'noisy')]:
        d = os.path.join(DATA, d)
        for p in sorted(glob.glob(f'{d}/{ndir}/*.wav')):
            x, n = rwav(p), os.path.basename(p)[:-4]
            for r in [16000, 44100]:
                os.makedirs(f'{d}/rate{r}', exist_ok=True); g = gcd(r, 48000)
                if not os.path.exists(f'{d}/rate{r}/{n}.f32'): resample_poly(x, r // g, 48000 // g).astype(np.float32).tofile(f'{d}/rate{r}/{n}.f32')
            os.makedirs(f'{d}/lp16k', exist_ok=True)
            if not os.path.exists(f'{d}/lp16k/{n}.f32'): fftconvolve(x, h, 'same').astype(np.float32).tofile(f'{d}/lp16k/{n}.f32')
    out = os.path.join(DATA, 'musdb', 'test-mono'); os.makedirs(out, exist_ok=True)
    for i, p in enumerate(sorted(glob.glob(os.path.join(DATA, 'musdb', 'test', '*.stem.mp4')))):
        f = f'{out}/{i:02d}.f32'
        if not os.path.exists(f): subprocess.run(['ffmpeg', '-v', 'error', '-i', p, '-map', '0:0', '-ac', '1', '-ar', '44100', '-f', 'f32le', f], check=True)

def guardsets():
    """The music guard's labelled sets, into DATA/guard/<split>-<kind>/<name>.f32 (mono float32) with a file `rate`;
    `train` sets choose the guard's constants, `test` sets report them. Speech, to be enhanced: VoiceBank+DEMAND clean
    and noisy (training speakers / test set), the noise alone (noisy − clean), the clean utterances over a song's
    accompaniment (MUSDB18 stems: drums + bass + other) 10, 15 and 20 dB under the speech's active level (bed10,
    bed15, bed20; every third utterance), and in white and pink Gaussian noise at 5, 10 and 20 dB SNR (every third),
    Spoken Wikipedia narrations (the spoken-train files / the ten ROOMS), VocalSet's spoken excerpts (vsspoken).
    Music, to pass: the MUSDB18 previews (songs) and their accompaniment (instr), Slakh2100's mixes (slakh: 60 s from
    30 s in; tracks 1–10 / 11–20), the four repair pieces (test). Singing alone, reported: MUSDB18 vocal stems (sung),
    VocalSet sung phrases (vssung) and long tones (vstone); singers 1–5 of each sex / the rest. Programmes: six
    segments each, speech (5 utterances of one kind, or 15 s of a narration) and music (up to 20 s of a song, its
    accompaniment or a Slakh mix) in turn, at -23 and -18 dBFS (the louder half of 50 ms frames), a label per 10 ms
    (1 speech, 0 music) in <name>.lab; 8 per split."""
    out = os.path.join(DATA, 'guard'); rng = np.random.default_rng(1)
    def put(s, n, x, sr):
        d = f'{out}/{s}'; os.makedirs(d, exist_ok=True); np.asarray(x, np.float32).tofile(f'{d}/{n}.f32'); open(f'{d}/rate', 'w').write(str(sr))
    def get(s, n):
        sr = int(open(f'{out}/{s}/rate').read()); x = rf32(f'{out}/{s}/{n}.f32'); g = gcd(48000, sr)
        return x if sr == 48000 else resample_poly(x, 48000 // g, sr // g)
    def ls(s): return sorted(os.path.basename(p)[:-4] for p in glob.glob(f'{out}/{s}/*.f32'))
    def active(x, sr):
        F = sr // 100; p = (x[:len(x) // F * F].reshape(-1, F) ** 2).mean(1); return p[p > np.percentile(p, 99) * 10 ** -3.5].mean()
    def stem(p, i): return np.frombuffer(subprocess.run(['ffmpeg', '-v', 'error', '-i', p, '-map', f'0:{i}', '-ac', '1', '-ar', '44100', '-f', 'f32le', '-'], check=True, capture_output=True).stdout, np.float32).astype(np.float64)
    def pink(n):
        X = np.fft.rfft(rng.standard_normal(n)); X[1:] /= np.sqrt(np.arange(1, len(X))); X[0] = 0; return np.fft.irfft(X, n)
    for split, vb, cdir, ndir in [('train', 'vbdemand-train', 'clean', 'noisy'), ('test', 'vbdemand', 'clean_testset_wav', 'noisy_testset_wav')]:
        acc = {}
        for p in sorted(glob.glob(os.path.join(DATA, 'musdb', split, '*.stem.mp4'))):
            n = os.path.basename(p)[:-9]; S = [stem(p, i) for i in range(5)]; acc[n] = S[1] + S[2] + S[3]
            put(f'{split}-songs', n, S[0], 44100); put(f'{split}-instr', n, acc[n], 44100); put(f'{split}-sung', n, S[4], 44100)
        keys = sorted(acc); utts = sorted(os.path.basename(p)[:-4] for p in glob.glob(f'{DATA}/{vb}/{cdir}/*.wav'))
        for n in utts:
            c, y = rwav(f'{DATA}/{vb}/{cdir}/{n}.wav'), rwav(f'{DATA}/{vb}/{ndir}/{n}.wav')
            put(f'{split}-clean', n, c, 48000); put(f'{split}-noisy', n, y, 48000); put(f'{split}-noise', n, y - c, 48000)
        for db in (10, 15, 20):
            for i, n in enumerate(utts[::3]):
                c = rwav(f'{DATA}/{vb}/{cdir}/{n}.wav'); a = resample_poly(acc[keys[(i * 7 + db) % len(keys)]], 160, 147)
                o = rng.integers(0, max(1, len(a) - len(c))); a = np.resize(a[o:o + len(c)], len(c))
                put(f'{split}-bed{db}', n, c + a * np.sqrt(active(c, 48000) / max(active(a, 48000), 1e-12) * 10 ** (-db / 10)), 48000)
        for kind in ('white', 'pink'):
            for db in (5, 10, 20):
                for n in utts[::3]:
                    c = rwav(f'{DATA}/{vb}/{cdir}/{n}.wav'); w = rng.standard_normal(len(c)) if kind == 'white' else pink(len(c))
                    put(f'{split}-{kind}{db}', n, c + w * np.sqrt(active(c, 48000) / np.mean(w ** 2) * 10 ** (-db / 10)), 48000)
        for p in sorted(glob.glob(os.path.join(DATA, 'spoken-train' if split == 'train' else 'spoken', '*.f32'))): put(f'{split}-narr', os.path.basename(p)[:-4], rf32(p), 48000)
        for p in sorted(glob.glob(os.path.join(DATA, 'vocalset', 'FULL', '*', 'excerpts', '*', '*.wav'))) + sorted(glob.glob(os.path.join(DATA, 'vocalset', 'FULL', '*', 'long_tones', 'straight', '*.wav'))):
            singer = p.split(os.sep)[-4]
            if (int(singer.lstrip('femal')) <= 5) != (split == 'train'): continue
            kind = 'vsspoken' if '/spoken/' in p else 'vstone' if 'long_tones' in p else 'vssung'; put(f'{split}-{kind}', os.path.basename(p)[:-4], rwav(p), 44100)
        for p in sorted(glob.glob(os.path.join(DATA, 'slakh', 'mix44', '*.wav'))):
            if (int(os.path.basename(p)[5:10]) <= 10) != (split == 'train'): continue
            x = rwav(p); put(f'{split}-slakh', os.path.basename(p)[:-4], x[30 * 44100:90 * 44100], 44100)
        if split == 'test':
            for n in ['vibeace', 'brahms', 'nutcracker', 'trumpet']: put('test-pieces', n, rf32(os.path.join(DATA, 'repair', n + '.f32'))[:60 * 44100], 44100)
        def level(x, db):
            p = np.sort((x[:len(x) // 2400 * 2400].reshape(-1, 2400) ** 2).mean(1))[::-1]; return x * np.sqrt(10 ** (db / 10) / max(p[:max(1, len(p) // 2)].mean(), 1e-12))
        def speech():
            k = rng.choice(['noisy', 'clean', 'bed15', 'narr'])
            if k == 'narr': x = get(f'{split}-narr', rng.choice(ls(f'{split}-narr'))); o = rng.integers(0, len(x) - 15 * 48000); return level(x[o:o + 15 * 48000], -23)
            return level(np.concatenate([get(f'{split}-{k}', n) for n in rng.choice(ls(f'{split}-{k}'), 5, replace=False)]), -23)
        def music():
            k = rng.choice(['songs', 'instr', 'slakh']); x = get(f'{split}-{k}', rng.choice(ls(f'{split}-{k}'))); return level(x[:20 * 48000], -18)
        for i in range(8):
            parts = [(speech(), 1) if (i + j) % 2 == 0 else (music(), 0) for j in range(6)]
            x = np.concatenate([p for p, _ in parts]); lab = np.zeros(len(x) // 480, np.uint8); o = 0
            for p, l in parts: lab[o // 480:(o + len(p)) // 480] = l; o += len(p)
            put(f'{split}-prog', f'prog{i}', x, 48000); lab.tofile(f'{out}/{split}-prog/prog{i}.lab')

def table(rows, keys, label):
    for L in sorted(set(r['limit'] for r in rows)):
        R = [r for r in rows if r['limit'] == L]
        print(f'{label:28s} limit {L:4.0f}  n={len(R)}  ' + '  '.join(f'{k} {np.nanmean([r[k] for r in R]):7.3f}' for k in keys), flush=True)

if __name__ == '__main__':
    what, args = (sys.argv[1] if len(sys.argv) > 1 else ''), sys.argv[2:]
    jobs = int(os.environ.get('JOBS', 4))
    if what == 'prepare': prepare()
    elif what == 'guard-sets': guardsets()
    elif what.startswith(('vbdemand', 'vbtrain')) or what == 'limits':
        setname, systems, limits = (args[0], [args[1]], [float(v) for v in args[2].split(',')]) if what == 'limits' else (what, args, [None])
        d = vbset(setname)[0]; os.makedirs(f'{d}/scores', exist_ok=True)
        with Pool(jobs) as pool:
            for system in systems:
                rows = pool.map(vb, [(setname, system, n, L) for n in names(setname) for L in limits], chunksize=8)
                with open(f'{d}/scores/neural-{setname}-{system}{"-limits" if what == "limits" else ""}.csv', 'w', newline='') as fh:
                    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
                table(rows, ['pesq', 'stoi', 'sisdr', 'sig', 'bak', 'ovrl'], f'{setname} {system}')
                if what == 'limits':
                    ref = {r['name']: r for r in rows if r['limit'] == limits[-1]}
                    for L in limits[:-1]:
                        dd = {k: np.array([r[k] - ref[r['name']][k] for r in rows if r['limit'] == L]) for k in ['pesq', 'stoi', 'sisdr', 'sig', 'bak', 'ovrl']}
                        print(f'{"":28s} limit {L:4.0f} against {limits[-1]:.0f}, paired: ' + '  '.join(f'{k} {v.mean():+.4f} ± {1.96 * v.std(ddof=1) / np.sqrt(len(v)):.4f}' for k, v in dd.items()), flush=True)
                if what == 'limits' and os.path.exists(f'{d}/scores/neural-{setname}-noisy.csv'):   # files that lose the voice
                    noisy = {r['name']: float(r['stoi']) for r in csv.DictReader(open(f'{d}/scores/neural-{setname}-noisy.csv'))}
                    for L in limits:
                        loss = np.array([noisy[r['name']] - r['stoi'] for r in rows if r['limit'] == L])
                        print(f'{"":28s} limit {L:4.0f}  STOI down by more than 0.1: {(loss > .1).sum()} files, 0.2: {(loss > .2).sum()}, most {loss.max():.3f}', flush=True)
    elif what == 'clean':
        setname, system = args[0], args[1]; limits = [float(v) for v in args[2].split(',')] if len(args) > 2 else [None]
        with Pool(jobs) as pool: rows = pool.map(cleanrow, [(setname, system, n, L) for n in names(setname) for L in limits], chunksize=8)
        table(rows, ['ends', 'quiet', 'active', 'sisdr'], f'{setname} {system}')
    elif what == 'music':
        d = os.path.join(DATA, 'repair', 'out-neural'); ns = sorted(os.path.basename(p)[:-4] for p in glob.glob(f'{d}/raw/*.f32'))
        groups = [('pieces', lambda n: not n.startswith(('chord', 'musdb'))), ('VocalSet chords', lambda n: n.startswith('chord')), ('MUSDB18 previews', lambda n: n.startswith('musdb'))]
        for system in args:
            rows = [musicrow((system, n)) for n in ns]
            for group, sel in groups:
                R = [r for r in rows if sel(r['name'])]
                if R: print(f'{system:18s} {group:16s} n={len(R):2d}  SI-SDR {np.median([r["sisdr"] for r in R]):5.1f} dB  per band, median (worst) dB: ' + '  '.join(f'{a}-{b} {np.median([r[f"{a}-{b}"] for r in R]):5.1f} ({np.min([r[f"{a}-{b}"] for r in R]):5.1f})' for a, b in BANDS), flush=True)
    elif what == 'presence':
        system = args[0]; md = os.path.join(DATA, 'repair', 'out-neural', system)
        jobs_ = [(k, system, n) for k in ['vbtrain', 'vbtrain-clean'] for n in names(k)] + [('music', system, os.path.basename(p)[:-4]) for p in sorted(glob.glob(f'{md}/*.f32'))]
        with Pool(jobs) as pool: rows = pool.map(presence, jobs_, chunksize=8)
        groups = [('noisy speech (vbtrain)', lambda k, n: k == 'vbtrain'), ('clean speech (vbtrain)', lambda k, n: k == 'vbtrain-clean'),
                  ('music pieces', lambda k, n: k == 'music' and not n.startswith(('chord', 'musdb'))), ('VocalSet chords', lambda k, n: n.startswith('chord')), ('MUSDB18 previews', lambda k, n: n.startswith('musdb'))]
        for g, sel in groups:
            v = np.array([r[2] for r in rows if sel(r[0], r[1])])
            if len(v): print(f'{system:8s} {g:24s} n={len(v):3d}  taken, dB: min {v.min():6.1f}  p5 {np.percentile(v, 5):6.1f}  median {np.median(v):6.1f}  p95 {np.percentile(v, 95):6.1f}  max {v.max():6.1f}  | files under -6 / -9 / -12 dB: {(v < -6).sum()} / {(v < -9).sum()} / {(v < -12).sum()}', flush=True)
    elif what == 'rooms':
        d = os.path.join(DATA, 'spoken'); res = {}
        ns = sorted(os.path.basename(p)[:-4] for p in glob.glob(f'{d}/out/raw/*.f32'))
        for system in args or ['raw', 'omlsa', 'enhance-denoise', 'rnnoise', 'rnnoise-limit16', 'rnnoise-guard', 'dfn3-upstream', 'dfn3', 'dfn3-limit18', 'dfn3-0.2-limit18', 'dfn3-limit12', 'dfn3-guard']:
            if not os.path.isdir(f'{d}/out/{system}'): continue
            rows = []
            for n in ns:
                x = rf32(f'{d}/out/{system}/{n}.f32'); raw = rf32(f'{d}/out/raw/{n}.f32')
                sig, bak, ovr = dnsmos(resample_poly(x, 1, 3))
                rows.append(dict(name=n, sig=sig, bak=bak, ovrl=ovr, floor=floor(x), dspeech=speech(x) - speech(raw)))
            res[system] = rows
            narr = [r for r in rows if r['name'] != 'lena']; fl = [r['floor'] for r in narr if r['floor'] > -190]
            print(f"{system:16s} SIG {np.mean([r['sig'] for r in narr]):.2f} BAK {np.mean([r['bak'] for r in narr]):.2f} OVRL {np.mean([r['ovrl'] for r in narr]):.2f}  floor {min(fl):.0f} to {max(fl):.0f} (median {np.median(fl):.0f}) dBFS  speech {np.mean([r['dspeech'] for r in narr]):+.1f} dB", flush=True)
        json.dump(res, open(f'{d}/scores.json', 'w'), indent=1)
    else:
        print('usage: python scripts/accuracy.py prepare | SET SYSTEM... | limits SET SYSTEM LIMITS | clean SET SYSTEM [LIMITS] | music SYSTEM... | rooms [SYSTEM...]')
