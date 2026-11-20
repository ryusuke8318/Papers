#!/bin/bash
# 5 つの学習を 1 つの VM で並列に走らせ(各プロセスは 15GB の上限つき)、終わると評価まで行う。
cd /content/d/david_mini
C="--batch 24 --main_steps 50"
nohup python train.py --name baseline     --warm_steps 0  --lam 0   --expansion 0 --group 6 $C > /content/baseline.log 2>&1 &
nohup python train.py --name full         --warm_steps 15 --lam 0.5 --expansion 1 --group 5 $C > /content/full.log 2>&1 &
nohup python train.py --name no_warm      --warm_steps 0  --lam 0.5 --expansion 1 --group 5 $C > /content/no_warm.log 2>&1 &
nohup python train.py --name no_reward    --warm_steps 15 --lam 0   --expansion 1 --group 5 $C > /content/no_reward.log 2>&1 &
nohup python train.py --name no_expansion --warm_steps 15 --lam 0.5 --expansion 0 --group 6 $C > /content/no_expansion.log 2>&1 &
nohup python evaluate.py --name untrained > /content/untrained.log 2>&1 &
sleep 2; echo launched
