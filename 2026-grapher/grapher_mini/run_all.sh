#!/bin/sh
# 全データの実行。work/ に埋め込みなどの中間ファイルができる。
cd "$(dirname "$0")"
PY=../.venv/bin/python
$PY run.py --dataset hotpot --tag tune
for d in hotpot 2wiki musique; do $PY run.py --dataset $d --tag eval; done
