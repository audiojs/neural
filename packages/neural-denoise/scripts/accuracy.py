# Scores for the Accuracy section, from the outputs scripts/accuracy.mjs writes.
#   python scripts/accuracy.py vbdemand SYSTEM [SYSTEM...]   (noisy: the input itself)
#   python scripts/accuracy.py rooms
# pip install numpy scipy pesq pystoi onnxruntime; DNSMOS: sig_bak_ovr.onnx from microsoft/DNS-Challenge at
# 82f1b17e77 (DNSMOS/DNSMOS/, CC BY 4.0) in $AUDIO_NEURAL_CACHE/dnsmos/ (default ~/.cache/audiojs/neural/dnsmos/).
#
# VoiceBank+DEMAND, against the clean reference, all at 16 kHz (scipy.signal.resample_poly(x, 1, 3)):
#   PESQ: ITU-T P.862.2 wideband MOS-LQO (python-pesq 0.0.4, the ITU reference C code)
#   STOI: Taal, Hendriks, Heusdens, Jensen, IEEE TASLP 19(7), 2011 (pystoi 0.4.1, extended=False)
#   SI-SDR: Le Roux, Wisdom, Erdogan, Hershey, ICASSP 2019, eq. 3, zero-mean signals, dB
# Both sets, no reference: DNSMOS P.835 (Reddy, Gopal, Cutler, ICASSP 2022), dnsmos_local.py's method at
# 82f1b17e77 (non-personalized): 9.01 s windows at a 1 s hop, shorter clips tiled, SIG/BAK/OVRL polynomial fits,
# averaged over windows. Rooms also: the noise floor, RMS of the quietest 500 ms at a 50 ms hop (ACX Check's
# measure), and the speech level, mean power of the loudest half of 50 ms frames, against the raw input.
import os, sys, glob, csv, json, numpy as np, onnxruntime as ort
from multiprocessing import Pool
from scipy.signal import resample_poly
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
    ref = ref - ref.mean(); est = est - est.mean()
    t = np.dot(est, ref) / np.dot(ref, ref) * ref
    return float(10 * np.log10(np.dot(t, t) / np.dot(est - t, est - t)))

def vb(args):
    from pesq import pesq
    from pystoi import stoi
    system, name = args
    d = os.path.join(DATA, 'vbdemand')
    ref = wavfile.read(f'{d}/clean_testset_wav/{name}.wav')[1].astype(np.float64) / 32768
    est = wavfile.read(f'{d}/noisy_testset_wav/{name}.wav')[1].astype(np.float64) / 32768 if system == 'noisy' else np.fromfile(f'{d}/out/{system}/{name}.f32', dtype=np.float32).astype(np.float64)
    assert len(est) == len(ref), (system, name)
    r, e = resample_poly(ref, 1, 3), resample_poly(est, 1, 3)
    sig, bak, ovr = dnsmos(e)
    return dict(name=name, pesq=pesq(16000, r, e, 'wb'), stoi=stoi(r, e, 16000, extended=False), sisdr=sisdr(r, e), sig=sig, bak=bak, ovrl=ovr)

def floor(x, fs=48000):
    w, h = fs // 2, fs // 20
    return 10 * np.log10(max(min(np.mean(x[i:i + w] ** 2) for i in range(0, len(x) - w, h)), 1e-20))

def speech(x, fs=48000):
    h = fs // 20; p = np.sort([np.mean(x[i:i + h] ** 2) for i in range(0, len(x) - h, h)])
    return 10 * np.log10(np.mean(p[len(p) // 2:]) + 1e-20)

if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else ''
    if what == 'vbdemand':
        d = os.path.join(DATA, 'vbdemand'); os.makedirs(f'{d}/scores', exist_ok=True)
        names = sorted(os.path.basename(p)[:-4] for p in glob.glob(f'{d}/clean_testset_wav/*.wav'))
        with Pool(4) as pool:
            for system in sys.argv[2:]:
                rows = pool.map(vb, [(system, n) for n in names], chunksize=8)
                with open(f'{d}/scores/{system}.csv', 'w', newline='') as fh:
                    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
                m = {k: np.mean([r[k] for r in rows]) for k in ['pesq', 'stoi', 'sisdr', 'sig', 'bak', 'ovrl']}
                print(f"{system:16s} n={len(rows)} PESQ {m['pesq']:.3f}  STOI {m['stoi']:.4f}  SI-SDR {m['sisdr']:6.2f}  DNSMOS SIG {m['sig']:.3f} BAK {m['bak']:.3f} OVRL {m['ovrl']:.3f}", flush=True)
    elif what == 'rooms':
        d = os.path.join(DATA, 'spoken'); res = {}
        names = sorted(os.path.basename(p)[:-4] for p in glob.glob(f'{d}/out/raw/*.f32'))
        for system in ['raw', 'omlsa', 'enhance-denoise', 'rnnoise', 'dfn3', 'dfn3-limit12']:
            rows = []
            for n in names:
                x = np.fromfile(f'{d}/out/{system}/{n}.f32', dtype=np.float32).astype(np.float64)
                raw = np.fromfile(f'{d}/out/raw/{n}.f32', dtype=np.float32).astype(np.float64)
                sig, bak, ovr = dnsmos(resample_poly(x, 1, 3))
                rows.append(dict(name=n, sig=sig, bak=bak, ovrl=ovr, floor=floor(x), dspeech=speech(x) - speech(raw)))
            res[system] = rows
            narr = [r for r in rows if r['name'] != 'lena']; fl = [r['floor'] for r in narr if r['floor'] > -190]
            print(f"{system:16s} SIG {np.mean([r['sig'] for r in narr]):.2f} BAK {np.mean([r['bak'] for r in narr]):.2f} OVRL {np.mean([r['ovrl'] for r in narr]):.2f}  floor {min(fl):.0f} to {max(fl):.0f} (median {np.median(fl):.0f}) dBFS  speech {np.mean([r['dspeech'] for r in narr]):+.1f} dB", flush=True)
        json.dump(res, open(f'{d}/scores.json', 'w'), indent=1)
    else:
        print('usage: python scripts/accuracy.py vbdemand SYSTEM... | rooms')
