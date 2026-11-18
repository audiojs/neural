#!/usr/bin/env bash
# The whole run, on this laptop (Apple Silicon, MPS) or a CUDA box: environments, data, synthetic voices, manifests,
# the two training stages, export. Each step is restartable; what is there is kept.
#
#   bash run.sh venv                   the trainer's venv (torch, torchaudio, soundfile, scipy, onnx, onnxruntime, pesq, pystoi)
#                                      and the TTS venv (supertonic, pocket-tts); Python 3.12
#   bash run.sh fetch   [DATA]         SIWIS (fr, CC BY 4.0), VCTK 0.92 mic1 (en, CC BY 4.0), LibriTTS-R train-clean-100 (en, CC BY 4.0),
#                                      Dialogs (ru, OpenRAIL), MUSAN noise (CC BY 4.0), WavLM Base+ (MIT); archives unpacked and deleted as they land
#   bash run.sh texts   [DATA]         public-domain text (Project Gutenberg; Dialogs' transcripts) → utterance lists, en fr ru
#   bash run.sh tts     [DATA] [HOURS] synthetic voices in the background: Supertonic 3 (44.1 kHz) ru, fr, en; Pocket TTS (24 kHz) fr, en
#   bash run.sh scan    [DATA]         manifests with each file's bandwidth; VoiceBank's test speakers held out of VCTK
#   bash run.sh listen  [DATA]         20 training pairs in DATA/listen: the half hour that decides more than any hyperparameter
#   bash run.sh a       [DATA] [RUN]   stage A: dim 384, 6 blocks, batch 8, 60k steps (the recipe on real data). On MPS a step costs the
#                                      same at dim 512 (the discriminators and WavLM are the cost), so the laptop goes straight to b
#   bash run.sh b       [DATA] [RUN]   stage B: the full model, dim 512, 8 blocks; resumes if RUN exists
#   bash run.sh watch   [DATA] [RUN]   in the background: as corpora land and the synthesis ends, rescan, then restart stage B on
#                                      every manifest right after a checkpoint (watch.sh)
#   bash run.sh export  RUN            ONNX fp32 and fp16 into the neural cache, verified
#   INIT=RUN/net-N.pt bash run.sh b    start over from a saved generator when last.pt went bad (fresh discriminators, alone for 2000 steps)
#
# DATA defaults to ~/.cache/audiojs/data/restore, RUN to runs/<stage> beside this script. Rooms: measured responses under
# DATA/rirs (the even-numbered MIT IR Survey rooms are linked there on the laptop: the odd ones are the bench's), 30 % of
# rooms stay synthetic by design, and a measured room's band is continued to 24 kHz (degrade.extend_rir).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
CMD=${1:-help}; DATA=${2:-$HOME/.cache/audiojs/data/restore}; RUN=${3:-$HERE/runs/$CMD}
VENV=$HOME/.cache/audiojs/venv; TTSVENV=$HOME/.cache/audiojs/venv-tts
PY=$VENV/bin/python; TPY=$TTSVENV/bin/python

venv() {
  command -v uv >/dev/null || { echo "install uv (pip install uv) first"; exit 1; }
  [ -x "$PY" ] || uv venv "$VENV" --python 3.12
  uv pip install --python "$PY" -q torch torchaudio soundfile scipy numpy pystoi onnx onnxruntime onnxconverter-common pyloudnorm pesq || true
  [ -x "$TPY" ] || uv venv "$TTSVENV" --python 3.12
  uv pip install --python "$TPY" -q supertonic pocket-tts soundfile numpy onnxruntime huggingface-hub
  "$PY" "$HERE/test.py" --steps 4
}

fetch() {
  mkdir -p "$DATA/dl" "$DATA/rirs" "$DATA/noise"
  [ -d "$DATA/musan-noise/noise" ] || "$TPY" -c "from huggingface_hub import snapshot_download as s; s('bilguun/musan-noise', repo_type='dataset', local_dir='$DATA/musan-noise')"
  ln -sfn "$DATA/musan-noise/noise" "$DATA/noise/musan"
  [ -d "$DATA/dialogs-ru/wavs" ] || "$TPY" -c "from huggingface_hub import snapshot_download as s; s('langswap/dialogs-ru-emotional-conversations', repo_type='dataset', local_dir='$DATA/dialogs-ru', allow_patterns=['*.csv', '*.md', 'wavs/*'])"
  "$PY" -c "import torchaudio; torchaudio.pipelines.WAVLM_BASE_PLUS.get_model(); print('wavlm ok')"
  cd "$DATA/dl"
  get() { local url=$1 f=$2; [ -e "$f.done" ] && return 0; curl -fL --retry 10 --retry-all-errors -C - -sS -o "$f" "$url" && touch "$f.done"; }
  if [ ! -e siwis.unpacked ]; then get "https://datashare.ed.ac.uk/bitstreams/64de089d-0ad5-45cb-a1c0-b6b39ecf6225/download" siwis.zip && mkdir -p ../siwis && unzip -qo siwis.zip -d ../siwis && touch siwis.unpacked && rm -f siwis.zip; fi
  if [ ! -e vctk.unpacked ]; then  # DataShare's bundle holds VCTK-Corpus-0.92.zip stored whole: both zips read as streams, only mic1 lands (5.5 GB, not 11.8 + 5.5)
    curl -fL --retry 5 -sS "https://huggingface.co/datasets/duanyu027/vctk_0.92/resolve/main/DS_10283_3443.zip" | tar -xOf - VCTK-Corpus-0.92.zip | tar -xf - -C ../vctk --include='*_mic1.flac' --include='*speaker-info.txt' --include='*README*' --include='*txt/*'
    [ "$(find ../vctk -name '*_mic1.flac' | wc -l)" -ge 44000 ] && touch vctk.unpacked
  fi
  if [ ! -e libritts.unpacked ]; then get https://www.openslr.org/resources/141/train_clean_100.tar.gz libritts_r_train_clean_100.tar.gz && mkdir -p ../libritts-r && tar xzf libritts_r_train_clean_100.tar.gz -C ../libritts-r && touch libritts.unpacked && rm -f libritts_r_train_clean_100.tar.gz; fi
}

texts() {
  mkdir -p "$DATA/text/gutenberg" "$DATA/text/en" "$DATA/text/fr" "$DATA/text/ru"; cd "$DATA/text/gutenberg"
  for id in 1342 158 98 2701 1661 1399; do [ -s "$id.txt" ] || curl -sfL -o "$id.txt" "https://www.gutenberg.org/cache/epub/$id/pg$id.txt"; cp "$id.txt" ../en/; done
  for id in 17489 17493 14155 17989 2650; do [ -s "$id.txt" ] || curl -sfL -o "$id.txt" "https://www.gutenberg.org/cache/epub/$id/pg$id.txt"; cp "$id.txt" ../fr/; done
  for id in 14741 16527 30774 37196; do [ -s "../ru/$id.txt" ] || { curl -sfL -o "ru-$id.raw" "https://www.gutenberg.org/files/$id/$id.txt" && "$PY" -c "
import sys, re; raw = open('ru-$id.raw', 'rb').read()
best = max((len(re.findall(r'[а-яё]', s, re.I)), s) for s in (raw.decode(e, 'ignore') for e in ('utf-8', 'koi8-r', 'cp1251', 'iso-8859-5')))
open('../ru/$id.txt', 'w', encoding='utf-8').write(best[1])"; }; done
  [ -s ../ru/dialogs.lines ] || "$PY" -c "
import csv; rows = csv.DictReader(open('$DATA/dialogs-ru/metadata.csv', encoding='utf-8'), delimiter='|')
open('../ru/dialogs.lines', 'w', encoding='utf-8').write('\n'.join(sorted({r['text'].strip() for r in rows if len(r['text'].strip()) >= 20})) + '\n')"
  cd "$HERE"; for l in en fr ru; do "$TPY" tts.py texts "$DATA" --lang $l; done
}

tts() {  # Supertonic: ru in three shards, fr and en in one; Pocket: fr and en; all in the background, logs in DATA/tts/logs
  local hours=${3:-8}; mkdir -p "$DATA/tts/logs"; cd "$HERE"
  for i in 0 1 2; do nohup "$TPY" tts.py synth "$DATA" --lang ru --hours "$hours" --engine supertonic --shard $i/3 --threads 3 > "$DATA/tts/logs/ru-supertonic-$i.log" 2>&1 & done
  nohup "$TPY" tts.py synth "$DATA" --lang fr --hours "$hours" --engine supertonic --threads 3 > "$DATA/tts/logs/fr-supertonic-0.log" 2>&1 &
  nohup "$TPY" tts.py synth "$DATA" --lang en --hours "$hours" --engine supertonic --threads 2 > "$DATA/tts/logs/en-supertonic-0.log" 2>&1 &
  nohup "$TPY" tts.py synth "$DATA" --lang fr --hours "$hours" --engine pocket --threads 2 > "$DATA/tts/logs/fr-pocket-0.log" 2>&1 &
  nohup "$TPY" tts.py synth "$DATA" --lang en --hours "$hours" --engine pocket --threads 2 > "$DATA/tts/logs/en-pocket-0.log" 2>&1 &
  echo "synthesis running; tail $DATA/tts/logs/*.log"
}

scan() {
  cd "$HERE"
  [ -d "$DATA/vctk/wav48_silence_trimmed" ] && "$PY" data.py scan "$DATA/en-vctk.json" "$DATA/vctk/wav48_silence_trimmed" --lang en --skip 'p232|p257'
  [ -d "$DATA/libritts-r/LibriTTS_R" ] && "$PY" data.py scan "$DATA/en-libritts.json" "$DATA/libritts-r/LibriTTS_R" --lang en
  [ -d "$DATA/siwis" ] && "$PY" data.py scan "$DATA/fr-siwis.json" "$DATA/siwis" --lang fr
  [ -d "$DATA/dialogs-ru/wavs" ] && "$PY" data.py scan "$DATA/ru-dialogs.json" "$DATA/dialogs-ru/wavs" --lang ru
  for l in ru fr en; do [ -d "$DATA/tts/$l" ] && "$PY" data.py scan "$DATA/tts-$l.json" "$DATA/tts/$l" --lang $l; done
  [ -d "$HOME/.cache/audiojs/data/vocalset/FULL" ] && "$PY" data.py scan "$DATA/sing-vocalset.json" "$HOME/.cache/audiojs/data/vocalset/FULL" --kind sing
  [ -d "$HOME/.cache/audiojs/data/vbdemand/clean_testset_wav" ] && "$PY" data.py scan "$DATA/val-vb.json" "$HOME/.cache/audiojs/data/vbdemand/clean_testset_wav" --lang en
  ls "$DATA"/*.json
}

manifests() {  # every training manifest, Russian and French synthetic voices weighted ×2, singing ×2 (sing is 2 h against hundreds of speech)
  local m=""
  for f in en-vctk en-libritts fr-siwis tts-en; do [ -e "$DATA/$f.json" ] && m="$m,$DATA/$f.json"; done
  for f in ru-dialogs tts-ru tts-fr sing-vocalset; do [ -e "$DATA/$f.json" ] && m="$m,$DATA/$f.json:2"; done
  echo "${m#,}"
}

train() {
  cd "$HERE"; mkdir -p "$RUN"; local resume=""; [ -e "$RUN/last.pt" ] && resume=--resume
  "$PY" train.py --out "$RUN" --clean "$(manifests)" --val "$DATA/val-vb.json" --rirs "$DATA/rirs" --noises "$DATA/noise" $resume ${INIT:+--init "$INIT"} "$@" 2>&1 | tee -a "$RUN.log"
}

case "$CMD" in
  venv) venv;;
  fetch) fetch;;
  texts) texts;;
  tts) tts "$@";;
  scan) scan;;
  listen) cd "$HERE"; "$PY" data.py pairs "$DATA/listen" 20 --clean "$(manifests)" --rirs "$DATA/rirs" --noises "$DATA/noise";;
  a) train --dim 384 --blocks 6 --attn 2,4 --batch 8 --dseg 0.5 --steps 60000 --warmup 20000 --every 2000 --workers 4;;
  b) train --dim 512 --blocks 8 --attn 3,6 --batch 8 --dseg 0.5 --steps 300000 --warmup 20000 --every 2000 --workers 4;;
  watch) nohup bash "$HERE/watch.sh" "$DATA" "$RUN" > "$RUN-watch.log" 2>&1 & echo "watching: $RUN-watch.log";;
  export) cd "$HERE"; "$PY" export.py "$(ls "$RUN"/net-*.pt | tail -1)" --fp16 --verify;;
  *) sed -n 2,22p "$0";;
esac
