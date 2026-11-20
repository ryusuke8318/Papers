"""README の本文の数値を、results/ の生データから独立に計算し直して照合する。"""
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from david_stats import paired_diff_ci

R = Path(__file__).resolve().parent / "results"
README = (Path(__file__).resolve().parents[1] / "README.md").read_text()
DS = ["hotpot", "2wiki", "musique"]
ld = lambda r, d: json.loads((R / r / f"eval_{d}.json").read_text())
em = lambda r: np.mean([x["em"] for d in DS for x in ld(r, d)]) * 100
ok = True


def check(label, value, text):
    global ok
    hit = text in README
    ok &= hit
    print("OK " if hit else "NG ", label, value, "->", text)


for r, t in [("untrained", "11.9"), ("baseline", "18.8"), ("full", "15.6"), ("baseline_s1", "14.4"), ("full_s1", "19.6")]:
    check(f"3-data EM {r}", f"{em(r):.1f}", t)
    assert f"{em(r):.1f}" == t, (r, em(r))
check("mean baseline", f"{(em('baseline') + em('baseline_s1')) / 2:.1f}", "baseline 16.6")
check("mean full", f"{(em('full') + em('full_s1')) / 2:.1f}", "full 17.6")
assert f"{(em('baseline') + em('baseline_s1')) / 2:.1f}" == "16.6" and f"{(em('full') + em('full_s1')) / 2:.1f}" == "17.6"


def diff(a, b):
    A = [x["em"] for d in DS for x in ld(a, d)]
    B = [x["em"] for d in DS for x in ld(b, d)]
    m, lo, hi = paired_diff_ci(A, B)
    return f"{100 * m:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]"


for a, b in [("full", "baseline"), ("full_s1", "baseline_s1"), ("baseline", "untrained"), ("no_warm", "full"), ("no_reward", "full"), ("no_expansion", "full")]:
    print(a, "-", b, diff(a, b))
check("seed0 diff", diff("full", "baseline"), "−3.2 [−6.3, −0.1]")
check("seed1 diff", diff("full_s1", "baseline_s1"), "+5.1 [+2.4, +7.9]")
check("untrained->baseline", diff("baseline", "untrained"), "+6.9 [+4.4, +9.4]")
check("no_reward - full", diff("no_reward", "full"), "+2.8 [+0.1, +5.6]")
check("no_warm - full", diff("no_warm", "full"), "+2.8 [−0.2, +5.7]")
# README は full − no_expansion を書いている(符号を反転した値)
check("full - no_expansion", "-2.6 [-5.6, +0.4] == -(+2.6 [-0.4, +5.6])" if diff("no_expansion", "full") == "+2.6 [-0.4, +5.6]" else "MISMATCH", "−2.6 [−5.6, +0.4]")
for r in ["baseline", "full", "no_expansion", "full_s1", "no_reward"]:
    print(r, "searches", [round(np.mean([x["n_search"] for x in ld(r, d)]), 2) for d in DS], "answered", [round(100 * np.mean([x["answered"] for x in ld(r, d)]), 1) for d in DS],
          "recall", [round(100 * np.mean([x["recall"] for x in ld(r, d)]), 1) for d in DS], "fullrecall", [round(100 * np.mean([x["recall"] >= 1 - 1e-9 for x in ld(r, d)]), 1) for d in DS])


def log(r):
    L = {}
    for l in (R / r / "log.jsonl").read_text().splitlines():
        x = json.loads(l)
        L[x["step"]] = x
    return [L[k] for k in sorted(L)]


m = [x for x in log("full") if x["phase"] == "main"]
print("full plan rate first/last10", sum(x["n_plan"] for x in m[:10]) / 240, sum(x["n_plan"] for x in m[-10:]) / 240,
      "answered first/last", np.mean([x["answered"] for x in m[:10]]), np.mean([x["answered"] for x in m[-10:]]),
      "extra/step", np.mean([x["n_extra_rollouts"] for x in m]), "plan/accept total", sum(x["n_plan"] for x in log("full")), sum(x["n_accept"] for x in log("full")))
for t in ["73%", "30%", "35%", "87%", "61.8", "618", "122", "19.7%"]:
    check("literal", "", t)
print("ALL OK" if ok else "MISMATCH")
