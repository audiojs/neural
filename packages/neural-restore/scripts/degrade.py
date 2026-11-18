"""The take as a bad recording makes it: a clean 48 kHz voice through a random chain, on the fly, per item.

Order is the physical one: the voice in a room, noise in the room, the microphone's colour and preamp, its
compressor and clipping, the channel's band limit and codec, the file's word length. Each stage draws its own
chance and strength from the ranges below; `Chain(rirs, noises)(clean) → degraded`, same length and about the
same level. CPU torch on (N,) tensors, cheap enough for data-loader workers; codecs through ffmpeg.

A room or a noise measured at 16 or 32 kHz stops at 8 or 16 kHz; `extend_rir` and `extend_noise` continue it to
the full band with synthetic content of the same decay (rooms) or colour (noise), at the level the recording has
in the octave under its edge, so a measured room never low-passes the voice by its own rate alone.
"""
import math, os, random, shutil, subprocess, tempfile
import torch
import torch.nn.functional as F
import torchaudio.functional as AF
from dsp import SR, band, lowmask, resample

FFMPEG = shutil.which('ffmpeg')
CODECS = [  # (name, args, container, bitrates kb/s): codecs a voice reaches us through, those this ffmpeg encodes
    ('mp3', ['-c:a', 'libmp3lame'], 'mp3', (24, 32, 48, 64, 96, 128)),
    ('aac', ['-c:a', 'aac'], 'm4a', (24, 32, 48, 64, 96)),
    ('opus', ['-c:a', 'libopus', '-application', 'voip'], 'ogg', (8, 12, 16, 24, 32, 48)),
    ('gsm', ['-c:a', 'libgsm', '-ar', '8000'], 'gsm', (13,)),
]
if FFMPEG:
    _enc = subprocess.run([FFMPEG, '-hide_banner', '-encoders'], capture_output=True, text=True).stdout
    CODECS[:] = [c for c in CODECS if f' {c[1][1]} ' in _enc]

def u(a, b): return random.uniform(a, b)
def lu(a, b): return math.exp(u(math.log(a), math.log(b)))
def chance(p): return random.random() < p

def db(x): return 10 ** (x / 20)
def power(x): return x.pow(2).mean() + 1e-12

def spectral(x, curve_db):
    """x through a zero-phase filter whose log-magnitude is `curve_db` over freqs(len(x)): a linear convolution
    (the buffer doubled, so the filter's ringing never wraps to the end of a room's tail), the first len(x) kept."""
    n = len(x)
    c = torch.nn.functional.interpolate(db(curve_db)[None, None], size=n + 1, mode='linear', align_corners=True)[0, 0]
    return torch.fft.irfft(torch.fft.rfft(x, 2 * n) * c, 2 * n)[:n]

def freqs(n): return torch.fft.rfftfreq(n, 1 / SR)

# ---------------------------------------------------------------- stages

def mic(x):
    """A cheap microphone and its placement: tilt, two or three resonances and dips, a low cut; ±12 dB at most."""
    f = freqs(len(x)); lf = torch.log2(f.clamp(min=20) / 1000)
    c = u(-4, 4) * lf                                   # dB per octave about 1 kHz
    for _ in range(random.randint(1, 3)):
        fc, q, g = lu(150, 12000), lu(0.5, 4), u(-12, 9)
        c = c + g * torch.exp(-((torch.log2(f.clamp(min=1) / fc)) * q) ** 2)
    hp = lu(40, 300)                                    # proximity's absence, a windscreen, a phone
    c = c + 12 * torch.log10(1 / (1 + (hp / f.clamp(min=1)) ** 4))
    c = c - c[(f > 300) & (f < 3000)].mean()            # the voice band's level kept
    return spectral(x, c.clamp(-30, 12))

def synth_rir(t60=None):
    """A room without a measurement: exponentially decaying Gaussian noise with a sparser first 20 ms and a
    direct path `drr` dB over the tail; T60 0.15–2.5 s."""
    t60 = t60 or lu(0.15, 2.5)
    n = int(min(t60 * 1.2, 3.0) * SR)
    t = torch.arange(n) / SR
    h = torch.randn(n) * torch.exp(-6.908 * t / t60)
    early = int(0.02 * SR)
    h[:early] *= (torch.rand(early) < u(0.02, 0.2)).float() * 3
    h[0] = 0
    drr = u(-6, 12)
    h = h * (h[1:].pow(2).sum() + 1e-9).rsqrt() * db(-drr)
    h[0] = 1.0
    if chance(0.5):  # air and walls take the top
        h = spectral(h, (-u(2, 10)) * torch.log2(freqs(n).clamp(min=1000) / 1000))
    return h

def t60(h, sr=SR):
    """Reverberation time of a response by Schroeder's backward integration: the decay from −5 dB to −25 dB (or
    to what the response reaches before its last tenth) extrapolated to −60 dB; 0.05–5 s."""
    e = h.pow(2).flip(0).cumsum(0).flip(0)
    d = 10 * torch.log10(e / (e[0] + 1e-20) + 1e-12)
    lo = (d < -5).nonzero()
    if not len(lo): return 0.1
    i5 = int(lo[0]); hi = (d < -25).nonzero(); n = len(h)
    i25 = int(hi[0]) if len(hi) and int(hi[0]) < 0.9 * n else int(0.9 * n)
    drop = float(d[i5] - d[i25])
    if i25 <= i5 or drop < 1: return 0.1
    return float(min(5.0, max(0.05, (i25 - i5) / sr * 60 / drop)))

def extend(x, bw, synth):
    """x, whose spectrum stops at `bw` Hz, continued to the full band with `synth` (the same length) above it,
    at the level x has in the octave under `bw`; bw ≥ 20 kHz leaves x as it is."""
    if bw >= 20000: return x
    n = len(x); f = freqs(n)
    X, S = torch.fft.rfft(x), torch.fft.rfft(synth)
    sel = (f > bw / 2) & (f < bw)
    g = (X[sel].abs().pow(2).mean() / (S[sel].abs().pow(2).mean() + 1e-20)).sqrt()
    m = lowmask(n, torch.tensor([float(bw)]))[0]
    return torch.fft.irfft(X * m + S * g * (1 - m), n)

def extend_rir(h, bw):
    """A measured room that stops at `bw` Hz, continued above it by a synthetic room of the same T60, its direct
    path at the same sample as the measured one."""
    if bw >= 20000: return h
    s = F.pad(synth_rir(t60(h)), (int(h.abs().argmax()), 0))
    n = max(len(h), len(s))
    return extend(F.pad(h, (0, n - len(h))), bw, F.pad(s, (0, n - len(s))))

def extend_noise(z, bw):
    """A noise recording that stops at `bw` Hz, continued above it by coloured noise at its level under the edge."""
    return z if bw >= 20000 else extend(z, bw, synth_noise(len(z), hum=False))

def room(x, h):
    """x in the room h, the direct path's level kept: y scaled so its energy equals x's."""
    n = len(x) + len(h) - 1
    y = torch.fft.irfft(torch.fft.rfft(x, n) * torch.fft.rfft(h, n), n)[: len(x)]
    return y * (power(x) / power(y)).sqrt()

def synth_noise(n, hum=True):
    """Coloured noise at a random slope, sometimes with mains hum and its harmonics."""
    f = freqs(n)
    y = spectral(torch.randn(n), -u(0, 12) * torch.log2(f.clamp(min=20) / 20))
    if hum and chance(0.3):
        t = torch.arange(n) / SR; f0 = random.choice((50, 60))
        for k in range(1, random.randint(2, 12)): y = y + torch.sin(2 * math.pi * f0 * k * t + u(0, 6.3)) * db(-u(0, 30)) / k
    return y

def noise(x, z, snr):
    z = z * (power(x) / power(z) / db(snr) ** 2).sqrt()
    return x + z

def compress(x, ratio, thresh_db, attack=0.005, release=0.1):
    """A preamp's or a phone's compressor: gain from a one-pole-smoothed envelope, over `thresh_db` reduced by ratio."""
    env = x.abs()
    a, r = math.exp(-1 / (attack * SR)), math.exp(-1 / (release * SR))
    one = lambda c: AF.lfilter(env[None], torch.tensor([1.0, -c]), torch.tensor([1 - c, 0.0]), clamp=False)[0]
    e = torch.maximum(one(r), one(a))
    edb = 20 * torch.log10(e + 1e-6)
    over = (edb - thresh_db).clamp(min=0)
    g = db(-over * (1 - 1 / ratio))
    y = x * g
    return y * (power(x) / power(y)).sqrt()

def clip(x):
    peak = x.abs().max() + 1e-9
    th = peak * u(0.15, 0.7)
    if chance(0.5): return x.clamp(-th, th)
    return torch.tanh(x / th) * th

def bandlimit(x, hz):
    return band(x[None], torch.tensor([hz]))[0]

def codec(x):
    """x through a real codec at a low bitrate: the file written, encoded, decoded back to 48 kHz by ffmpeg."""
    name, args, ext, rates = random.choice(CODECS)
    kb = random.choice(rates)
    with tempfile.TemporaryDirectory() as d:
        src, enc = os.path.join(d, 'in.wav'), os.path.join(d, 'enc.' + ext)
        import soundfile as sf
        sf.write(src, x.numpy(), SR, subtype='FLOAT')
        q = ['-loglevel', 'error', '-y']
        if subprocess.run([FFMPEG, *q, '-i', src, *args, '-b:a', f'{kb}k', enc]).returncode: return x
        out = subprocess.run([FFMPEG, *q, '-i', enc, '-f', 'f32le', '-ac', '1', '-ar', str(SR), '-'], capture_output=True)
        if out.returncode: return x
        y = torch.frombuffer(bytearray(out.stdout), dtype=torch.float32)
    # the codec's delay: the lag, under half a second, where y best meets x (cross-correlation by FFT)
    k = min(len(x), len(y), SR * 2)
    c = torch.fft.irfft(torch.fft.rfft(y[:k], 2 * k) * torch.fft.rfft(x[:k], 2 * k).conj(), 2 * k)[: SR // 2]
    lag = int(c.argmax())
    y = y[lag: lag + len(x)]
    return F.pad(y, (0, len(x) - len(y))) if len(y) < len(x) else y

def quantize(x):
    bits = random.choice((8, 12, 16))
    q = 2 ** (bits - 1)
    return torch.round(x * q + (torch.rand_like(x) - 0.5)) / q

# ---------------------------------------------------------------- the chain

class Chain:
    """P: the chance of each stage. `rirs`: (N,) tensors at 48 kHz, unit energy (data.Rirs.items), or none for
    synthetic rooms only; `noises`: anything with `crop(n) → (n,) tensor or None` (data.Noises), or None for
    synthetic noise only."""
    P = dict(mic=0.7, room=0.6, noise=0.8, compress=0.3, clip=0.2, bandlimit=0.6, codec=0.3 if FFMPEG else 0, quantize=0.3)

    def __init__(self, rirs=(), noises=None, p=None):
        self.rirs, self.noises = list(rirs), noises
        self.p = {**self.P, **(p or {})}

    def __call__(self, x):
        p = self.p
        y = x
        if chance(p['room']):
            h = random.choice(self.rirs) if self.rirs and chance(0.7) else synth_rir()
            y = room(y, h)
        if chance(p['noise']):
            z = self.noises.crop(len(y)) if self.noises is not None and chance(0.85) else None
            if z is None: z = synth_noise(len(y))
            y = noise(y, z, u(-5, 35))
        if chance(p['mic']): y = mic(y)
        if chance(p['compress']): y = compress(y, lu(2, 10), u(-35, -15))
        if chance(p['clip']): y = clip(y)
        if chance(p['bandlimit']): y = bandlimit(y, lu(2000, 20000))
        if chance(p['codec']): y = codec(y)
        if chance(p['quantize']): y = quantize(y)
        return y
