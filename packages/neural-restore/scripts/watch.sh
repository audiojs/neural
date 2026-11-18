#!/bin/bash
# Beside a stage B run: every ten minutes, scan any corpus that has just been unpacked (VCTK, LibriTTS-R, SIWIS) and
# the synthetic voices once their synthesis has ended; when something new is there, wait for the next checkpoint and
# restart the run on every manifest (`run.sh b` resumes from last.pt). bash watch.sh DATA RUN
DATA=$1; RUN=$2; HERE=$(cd "$(dirname "$0")" && pwd); PY=$HOME/.cache/audiojs/venv/bin/python
scan() { cd "$HERE"; "$PY" data.py scan "$@" 2>&1 | tail -1; }
while :; do
  changed=0
  [ -e "$DATA/dl/vctk.unpacked" ] && [ ! -e "$DATA/en-vctk.json" ] && { scan "$DATA/en-vctk.json" "$DATA/vctk/wav48_silence_trimmed" --lang en --skip 'p232|p257'; changed=1; }
  [ -e "$DATA/dl/libritts.unpacked" ] && [ ! -e "$DATA/en-libritts.json" ] && { scan "$DATA/en-libritts.json" "$DATA/libritts-r/LibriTTS_R" --lang en; changed=1; }
  [ -e "$DATA/dl/siwis.unpacked" ] && [ ! -e "$DATA/fr-siwis.json" ] && { scan "$DATA/fr-siwis.json" "$DATA/siwis" --lang fr; changed=1; }
  if ! pgrep -f "tts.py synth" > /dev/null && [ ! -e "$DATA/tts/scanned" ]; then
    for l in ru fr en; do [ -d "$DATA/tts/$l" ] && scan "$DATA/tts-$l.json" "$DATA/tts/$l" --lang $l; done; touch "$DATA/tts/scanned"; changed=1
  fi
  if [ "$changed" = 1 ]; then
    echo "$(date +%T) new data: $(ls "$DATA"/*.json | xargs -n1 basename | tr '\n' ' ')"
    if pgrep -f "train.py --out $RUN" > /dev/null; then
      while [ $(( $(date +%s) - $(stat -f %m "$RUN/last.pt" 2>/dev/null || echo 0) )) -gt 120 ]; do sleep 30; done   # right after a checkpoint
      pkill -f "train.py --out $RUN"; sleep 15
    fi
    cd "$HERE"; nohup bash run.sh b "$DATA" "$RUN" > /dev/null 2>&1 &
    echo "$(date +%T) restarted $RUN"
  fi
  # the trainer gone before its last step (killed for memory, or by hand): resumed from last.pt
  if ! pgrep -f "train.py --out $RUN" > /dev/null && [ -e "$RUN/last.pt" ] && ! tail -3 "$RUN.log" | grep -q '"step": 300000'; then
    sleep 60; pgrep -f "train.py --out $RUN" > /dev/null || { cd "$HERE"; nohup bash run.sh b "$DATA" "$RUN" > /dev/null 2>&1 & echo "$(date +%T) trainer was gone: resumed"; }
  fi
  sleep 600
done
