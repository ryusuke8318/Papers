#!/bin/bash
# VM から結果(評価の JSON、学習ログ、ピークメモリ)を取り出す。使い方: fetch.sh [run ...]
P=$(cd "$(dirname "$0")" && pwd)
runs="${@:-untrained baseline full no_warm no_reward no_expansion}"
colab ssh -s david -- "cd /content/d/david_mini && for n in $runs; do mkdir -p out/\$n; cp results/\$n/* out/\$n/ 2>/dev/null; cp work/runs/\$n/log.jsonl work/runs/\$n/peak_mem.json work/runs/\$n/args.json out/\$n/ 2>/dev/null; done; tar czf /content/out.tgz out" > /dev/null
colab download -s david /content/out.tgz $P/work/out.tgz
mkdir -p $P/results && tar xzf $P/work/out.tgz -C $P/work && cp -r $P/work/out/* $P/results/
