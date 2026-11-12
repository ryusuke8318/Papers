"""事例のプロンプト、採点、文脈の組み立て(torch に依存しない。GPU の評価と API の評価で共用する)。

条件(論文 Tab. 5 の構成に対応)
  plain      学習なし・文脈なし
  rag        学習なし + 検索した chunk を挿入(検索結果は sources/rag_contexts.json に固定)
  mocked     学習なし + 正解の unit を直接挿入(完全検索の上限)
  long       学習なし + 条文・通達の全文を挿入(長文脈)
  ft / rag_ft / mocked_ft   LoRA 学習したモデルでの plain / rag / mocked
出力は「根拠 / 理由 / 結論」の 3 行(論文と同じく理由を結論の前に書かせる)。
採点: 結論一致、根拠一致(正解の引用が出力の根拠に含まれる)、両方一致(strict)。
"""
from __future__ import annotations

import re

SPLITS = ("test_seen", "test_exc_seen", "test_exc_unseen", "real")
INSTR = ("次の取引について、消費税の課税対象となる国内取引か国外取引かを判定してください。\n"
         "次の形式で答えてください。\n根拠: <根拠とする条文・通達>\n理由: <判断の理由を3文以内で簡潔に>\n結論: <国内取引 または 国外取引>")

def prompt(case_text: str, ctx: list[str] | None = None) -> str:
    head = ""
    if ctx:
        head = "以下は消費税法・消費税法施行令・消費税法基本通達の抜粋です。必要なものだけを使って判断してください。\n\n【参考資料】\n" + \
               "\n".join(f"[{i + 1}] {c}" for i, c in enumerate(ctx)) + "\n\n"
    return f"{head}{INSTR}\n\n【事例】\n{case_text}"


def target(case: dict) -> str:
    cites = "、".join(case["gold_cites"])
    return f"根拠: {cites}\n理由: {case['reason']}\n結論: {'国内取引' if case['label'] else '国外取引'}"


def norm_cite(s: str) -> str:
    s = s.replace("消費税法施行令", "令").replace("施行令", "令").replace("消費税法基本通達", "基本通達").replace("消費税法", "法")
    s = s.replace("-", "－").replace("ー", "－").replace("−", "－")
    return re.sub(r"[\s「」『』]", "", s)


def score(out: str, case: dict) -> dict:
    m = re.findall(r"結論[:：]\s*(国内|国外)", out)
    pred = None if not m else (m[-1] == "国内")
    basis = re.search(r"根拠[:：](.*?)(?:\n|$)", out)
    b = norm_cite(basis.group(1)) if basis else norm_cite(out)
    cites = [norm_cite(c) for c in case["gold_cites"]]
    # 「令第6条第1項第7号」を「令第6条第1項」と書いても部分的に正しいが、ここでは先頭の正解引用が含まれることを求める
    cite_ok = bool(cites) and cites[0] in b
    return {"pred": pred, "correct": pred is not None and pred == case["label"], "cite": cite_ok,
            "strict": pred is not None and pred == case["label"] and cite_ok}


def agg(rows: list[dict]) -> dict:
    n = len(rows)
    return {"n": n, "acc": sum(r["correct"] for r in rows) / n, "cite": sum(r["cite"] for r in rows) / n,
            "strict": sum(r["strict"] for r in rows) / n, "unparsed": sum(r["pred"] is None for r in rows) / n}




def make_contexts(cond: str, case: dict, units_by_id: dict, units: list[dict], rag: dict) -> list[str] | None:
    if cond in ("plain", "ft"):
        return None
    if cond in ("mocked", "mocked_ft"):
        return [f"【{units_by_id[g]['cite']}】{units_by_id[g]['text']}" for g in case["gold"]]
    if cond == "long":
        return ["\n".join(f"【{u['cite']}】{u['text']}" for u in units)]
    for suffix in ("s2b", "hg", "rrg"):  # 検索の変種(make_s2b_contexts.py、make_ctx_variants.py)
        if cond in (f"rag_{suffix}", f"rag_ft_{suffix}"):
            return rag[f"{suffix}::" + case["id"]]
    return rag[case["id"]]  # rag, rag_ft
