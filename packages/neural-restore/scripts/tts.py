"""Clean voices from text: synthetic targets for a language with no open studio corpus (Russian, French), by
Supertonic 3 (Supertone; 31 languages, 44.1 kHz, OpenRAIL-M weights, ONNX on the CPU) or Pocket TTS (Kyutai;
English, French, 24 kHz, CC BY 4.0). A dry, clean, full-band (Supertonic) voice reading a sentence is what the
restorer must give back; the degradation chain makes the recording of it. Runs in a venv of its own
(supertonic, pocket-tts, soundfile; run.sh's `tts`), not the trainer's.

    python tts.py texts DIR --lang ru          DIR/text/ru/*.txt (prose: Project Gutenberg's plain text, split into
                                               sentences; pre-1918 Russian orthography modernised) and *.lines (one
                                               utterance a line) → DIR/text/ru.txt: utterances of one to three
                                               sentences, 50–240 characters, shuffled
    python tts.py synth DIR --lang ru --hours 8 [--engine supertonic|pocket] [--voices M1,F2,..] [--shard 0/3] [--threads 4]
                                               DIR/tts/ru/<engine>-<voice>-<k>.flac (16-bit, the engine's rate) and
                                               its text in <...>.txt; a file that exists is skipped, so shards and
                                               restarts add up; stops at --hours of audio in the directory
"""
import argparse, glob, os, random, re, sys, time
import numpy as np, soundfile as sf

SENT = re.compile(r'(?<=[.!?…])["»”’)]?\s+(?=["«“(]?[A-ZÀ-ÝА-ЯЁ])')
BAD = re.compile(r'[\[\]_*#@/\\|<>{}=+~^]|https?:|www\.|\d{3,}|[A-ZÀ-ÝА-ЯЁ]{4,}')
LETTERS = {'ru': re.compile(r'[а-яё]', re.I), 'en': re.compile(r'[a-z]', re.I), 'fr': re.compile(r'[a-zàâçéèêëîïôûùüÿœæ]', re.I)}
OLD = str.maketrans({'ѣ': 'е', 'Ѣ': 'Е', 'і': 'и', 'І': 'И', 'ѳ': 'ф', 'Ѳ': 'Ф', 'ѵ': 'и', 'Ѵ': 'И'})

def modern(t): return re.sub(r'ъ\b', '', t.translate(OLD))

def prose(path):
    """a Gutenberg text's sentences, the header and footer cut, paragraphs joined"""
    s = open(path, encoding='utf-8', errors='ignore').read()
    a, b = s.find('*** START OF'), s.find('*** END OF')
    if a >= 0: s = s[s.find('\n', a) + 1:]
    if b >= 0: s = s[:b]
    out = []
    for para in re.split(r'\n\s*\n', s):
        para = ' '.join(para.split())
        if len(para) >= 40: out.extend(SENT.split(para))
    return out

def ok(t, lang):
    if not 20 <= len(t) <= 200 or BAD.search(t): return False
    letters = sum(c.isalpha() for c in t)
    return letters >= 0.7 * len(t.replace(' ', '')) and len(LETTERS[lang].findall(t)) >= 0.9 * letters

def texts(d, lang):
    src = os.path.join(d, 'text', lang); books = []
    for p in sorted(glob.glob(os.path.join(src, '*.txt'))): books.append([s.strip() for s in prose(p)])
    for p in sorted(glob.glob(os.path.join(src, '*.lines'))): books.append([l.strip() for l in open(p, encoding='utf-8') if l.strip()])
    if lang == 'ru': books = [[modern(s) for s in b] for b in books]
    random.seed(0); utts = []
    for book in books:
        cur, k = '', 0
        for s in book:
            if not ok(s, lang): cur, k = '', 0; continue
            if cur and len(cur) + 1 + len(s) > 240: utts.append(cur); cur, k = '', 0
            cur, k = (cur + ' ' + s).strip(), k + 1
            if len(cur) >= 60 and (k >= 3 or random.random() < 0.5): utts.append(cur); cur, k = '', 0
        if len(cur) >= 50: utts.append(cur)
    utts = sorted(set(utts)); random.shuffle(utts)
    out = os.path.join(d, 'text', f'{lang}.txt')
    open(out, 'w', encoding='utf-8').write('\n'.join(utts) + '\n')
    print(f'{len(utts)} utterances, {sum(map(len, utts)) / 1e6:.2f} M characters, {len(books)} sources → {out}')

def trim(x, sr, floor_db=-45.0, margin=0.05):
    """leading and trailing silence cut, `margin` seconds kept each side"""
    n = int(0.01 * sr); m = len(x) // n
    if m < 2: return x
    e = 20 * np.log10(np.sqrt((x[: m * n].reshape(m, n) ** 2).mean(1)) + 1e-9)
    on = np.nonzero(e > e.max() + floor_db)[0]
    if not len(on): return x
    a, b = max(0, on[0] * n - int(margin * sr)), min(len(x), (on[-1] + 1) * n + int(margin * sr))
    return x[a:b]

def synth(d, lang, hours, engine, voices, shard, threads):
    lines = [l.rstrip('\n') for l in open(os.path.join(d, 'text', f'{lang}.txt'), encoding='utf-8')]
    out = os.path.join(d, 'tts', lang); os.makedirs(out, exist_ok=True)
    i, n = (int(v) for v in shard.split('/'))
    if engine == 'supertonic':
        from supertonic import TTS
        tts = TTS(intra_op_num_threads=threads); sr = 44100
        voices = voices or ['M1', 'M2', 'M3', 'M4', 'M5', 'F1', 'F2', 'F3', 'F4', 'F5']
        styles = {v: tts.get_voice_style(v) for v in voices}
        def speak(text, v):
            wav, _ = tts.synthesize(text, voice_style=styles[v], lang=lang)
            return np.asarray(wav, dtype=np.float32).reshape(-1)
    else:
        import torch
        from pocket_tts import TTSModel
        from pocket_tts.utils.utils import _ORIGINS_OF_PREDEFINED_VOICES as PRESETS
        torch.set_num_threads(threads)
        m = TTSModel.load_model(language={'en': 'english', 'fr': 'french'}[lang]); sr = m.sample_rate
        voices = voices or list(PRESETS)
        states = {v: m.get_state_for_audio_prompt(v) for v in voices}
        def speak(text, v): return m.generate_audio(states[v], text).reshape(-1).numpy()
    have = sum(sf.info(p).duration for p in glob.glob(os.path.join(out, f'{engine}-*.flac')))
    print(f'{engine} {lang}: {len(lines)} utterances, {len(voices)} voices, {have / 3600:.2f} h there, shard {shard}', flush=True)
    t0 = time.time(); made = 0.0
    for k in range(i, len(lines), n):
        if have + made >= hours * 3600: break
        v = voices[k % len(voices)]
        p = os.path.join(out, f'{engine}-{v}-{k:06d}.flac')
        if os.path.exists(p): continue
        try: wav = trim(speak(lines[k], v), sr)
        except Exception as e: print('fail', k, repr(e)[:100], file=sys.stderr); continue
        if len(wav) < sr * 0.5 or not np.isfinite(wav).all(): continue
        sf.write(p, np.clip(wav, -1, 1), sr, subtype='PCM_16'); open(p[:-5] + '.txt', 'w', encoding='utf-8').write(lines[k] + '\n')
        made += len(wav) / sr
        if (k // n) % 50 == 0: print(f'{k}: {(have + made) / 3600:.2f} h, RTF {(time.time() - t0) / max(made, 1e-9):.2f}', flush=True)
    print(f'done: {(have + made) / 3600:.2f} h in {out}')

if __name__ == '__main__':
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('cmd', choices=['texts', 'synth']); a.add_argument('dir'); a.add_argument('--lang', required=True, choices=['en', 'fr', 'ru'])
    a.add_argument('--hours', type=float, default=8); a.add_argument('--engine', default='supertonic', choices=['supertonic', 'pocket'])
    a.add_argument('--voices'); a.add_argument('--shard', default='0/1'); a.add_argument('--threads', type=int, default=4)
    o = a.parse_args()
    if o.cmd == 'texts': texts(o.dir, o.lang)
    else: synth(o.dir, o.lang, o.hours, o.engine, o.voices.split(',') if o.voices else None, o.shard, o.threads)
