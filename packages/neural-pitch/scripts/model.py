# The network, shared by train.py and export.py. Every layer commutes with a shift along frequency
# (transposition-equivariant, as PESTO: Riou et al., ISMIR 2023), so a transposed input moves the
# pitch posterior by as many bins:
#   input  (B, 288) variable-Q dB spectrum in 0…1 (features.js), plus its level
#   stem   conv 1 → C, kernel 9                                    local spectral shape
#   blocks depthwise conv kernel 5, dilation d, then pointwise C → C, residual, LeakyReLU
#   proj   pointwise C → T                                          harmonic evidence maps
#   comb   one kernel per map over offsets −D1…+D2 bins, summed:    y_j = b + Σ_c Σ_d w_cd h_c(j+d)
#          a Toeplitz layer (PESTO's final layer), banded to 2 octaves below and 4 above: harmonics
#          up to the 16th vote for their f0, subharmonics against it
#   → logits over the 288 input bins (bin k ↔ MIDI 23 + k/3), softmax = pitch posterior
#   voicing: per map max and mean over frequency, the logits' max and log-sum-exp, the level → MLP → sigmoid
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

BINS, LOW, PER = 288, 23, 3
SLOPE = 0.1                     # LeakyReLU negative slope


class Net(nn.Module):
    def __init__(self, C=16, T=4, blocks=(1, 2, 4), D1=72, D2=144, H=16):
        super().__init__()
        self.cfg = dict(C=C, T=T, blocks=list(blocks), D1=D1, D2=D2, H=H)
        self.stem = nn.Conv1d(1, C, 9, padding=4)
        self.dw = nn.ModuleList(nn.Conv1d(C, C, 5, padding=2 * d, dilation=d, groups=C) for d in blocks)
        self.pw = nn.ModuleList(nn.Conv1d(C, C, 1) for _ in blocks)
        self.proj = nn.Conv1d(C, T, 1)
        self.comb = nn.Parameter(torch.randn(T, D1 + D2 + 1) * 0.02)
        self.bias = nn.Parameter(torch.zeros(1))
        # the comb as a banded Toeplitz matrix (T·K × K): one matmul, much faster than a long conv on MPS
        t = torch.arange(BINS)[:, None] - torch.arange(BINS)[None] + D1
        self.register_buffer('tap', t.clamp(0, D1 + D2), persistent=False)
        self.register_buffer('band', ((t >= 0) & (t <= D1 + D2)).float(), persistent=False)
        self.head1 = nn.Linear(2 * T + 3, H)
        self.head2 = nn.Linear(H, 1)

    def forward(self, x, level):
        a = lambda v: F.leaky_relu(v, SLOPE)
        h = a(self.stem(x.unsqueeze(1)))
        for dw, pw in zip(self.dw, self.pw):
            h = a(h + pw(dw(h)))
        m = a(self.proj(h))                                             # (B, T, K)
        W = (self.comb[:, self.tap] * self.band).reshape(-1, BINS)     # W[c·K + i, j] = comb[c, i − j + D1]
        y = m.reshape(m.shape[0], -1) @ W + self.bias
        s = torch.cat([m.amax(2), m.mean(2), y.amax(1, keepdim=True), torch.logsumexp(y, 1, keepdim=True) - math.log(BINS), level.unsqueeze(1) / 100], 1)
        v = self.head2(a(self.head1(s))).squeeze(1)
        return y, v                                                     # pitch logits, voicing logit

    def macs(self):
        """Multiply-adds a frame of the convolutions; the comb runs as an FFT convolution in model.js."""
        C, T = self.cfg['C'], self.cfg['T']
        return BINS * (9 * C + len(self.dw) * (5 * C + C * C) + C * T)


def hz(midi):
    return 440 * 2 ** ((midi - 69) / 12)


def decode(p, w=2):
    """Posterior (B, K) → fractional bin: argmax, then the mean over ±w bins weighted by p."""
    j = p.argmax(1, keepdim=True)
    idx = (j + torch.arange(-w, w + 1, device=p.device)).clamp(0, p.shape[1] - 1)
    q = p.gather(1, idx)
    return (q * idx).sum(1) / q.sum(1).clamp_min(1e-12)
