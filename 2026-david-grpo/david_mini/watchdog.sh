#!/bin/bash
# 落ちた学習を、空きメモリがあるときにチェックポイントから再開する(物理 GPU メモリの不足で落ちることがあるため)。
cd /content/d/david_mini
export PYTORCH_ALLOC_CONF=expandable_segments:True
declare -A ARGS=(
 [baseline]="--warm_steps 0 --lam 0 --expansion 0 --group 6"
 [full]="--warm_steps 15 --lam 0.5 --expansion 1 --group 5"
 [no_warm]="--warm_steps 0 --lam 0.5 --expansion 1 --group 5"
 [no_reward]="--warm_steps 15 --lam 0 --expansion 1 --group 5"
 [no_expansion]="--warm_steps 15 --lam 0.5 --expansion 0 --group 6")
while true; do
  alldone=1
  for n in "${!ARGS[@]}"; do
    if [ ! -f work/runs/$n/DONE ]; then
      alldone=0
      used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
      if ! pgrep -f "name $n " >/dev/null && [ "$used" -lt 28000 ]; then
        echo "$(date) restart $n (used=$used)" >> /content/watchdog.log
        nohup python train.py --name $n ${ARGS[$n]} --batch 24 --main_steps 50 >> /content/$n.log 2>&1 &
        sleep 90
      fi
    fi
  done
  [ $alldone = 1 ] && break
  sleep 60
done
