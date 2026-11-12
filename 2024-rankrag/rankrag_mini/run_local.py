"""ローカル実行(Modal なし)。試運転向け。本番規模の学習・評価は Modal を推奨する。

  python rankrag_mini/run_local.py prepare --smoke       # 小さなデータを作る(初回はデータセットを取得)
  python rankrag_mini/run_local.py job smoke_rr --smoke  # 小さな学習 + 評価(GPU が必要)
  python rankrag_mini/run_local.py prepare               # 本番規模のデータ(CPU でも可。時間がかかる)
  python rankrag_mini/run_local.py job 05b_chatqa        # 本番規模の 1 条件(VRAM 12GB 以上を目安)

データは rankrag_mini/data/、LoRA の重みは rankrag_mini/work/、結果は rankrag_mini/results/ に保存される。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from jobs import JOBS  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--smoke", action="store_true")
    j = sub.add_parser("job")
    j.add_argument("name", choices=sorted(JOBS))
    j.add_argument("--smoke", action="store_true", help="小さなデータと 16 問の評価")
    a = ap.parse_args()

    data_dir = HERE / ("data_smoke" if a.smoke else "data")
    if a.cmd == "prepare":
        import data

        data.prepare(str(data_dir), a.smoke)
        return
    import rr_model

    out = rr_model.run_job(JOBS[a.name], str(data_dir), str(HERE / "work"), n_eval=16 if a.smoke else 500)
    (HERE / "results").mkdir(exist_ok=True)
    path = HERE / "results" / f"{a.name}{'_local' if a.smoke else ''}.json"
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print("保存:", path)


if __name__ == "__main__":
    main()
