"""検索を LLM で補う 2 つの処理(Anthropic API。Modal の Secret `anthropic-secret` から読む。キーは出力しない)。

  rewrite  事例を、条文を探すための短い文に書き換える(クエリ書き換え)
  rerank   検索で取った候補 unit を、LLM が関連の高い順に並べ替える(リストワイズ)
"""
from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from jp_api import call

REWRITE = """次の取引について、消費税の内外判定(消費税法第4条第3項、同法施行令第6条、消費税法基本通達5-7)の条文を検索するための検索文を書いてください。

- 取引の種類(資産の譲渡、貸付け、役務の提供、など)と、対象の資産や役務の種類(船舶、特許権、有価証券、運送、保険、通信など)を、条文で使われる言葉で書く
- 場所に関する条件(所在地、事務所等の所在地、登録した機関の所在地など)も、条文の言葉で書く
- 固有名詞や金額は書かない。1〜2文。検索文だけを出力する

取引:
{text}
"""

RERANK = """次の取引の内外判定に必要な条文・通達を、候補から選んでください。

取引:
{text}

候補:
{cands}

判定に必要な候補の番号を、重要な順にカンマ区切りで最大 8 個まで出力してください。番号だけを出力します(例: 3,12,5)。"""


def _client():
    import anthropic

    key = next((k for k in ("ANTHROPIC_API_KEY", "ANTHROPIC_KEY", "ANTHROPIC_SECRET") if os.environ.get(k)), None)
    if key is None:
        raise RuntimeError("Secret に API キーの環境変数が見つかりません")
    return anthropic.Anthropic(api_key=os.environ[key], max_retries=2)


def _map(model: str, prompts: list[str], workers: int = 8) -> list[str]:
    client = _client()
    use_temp = [True]

    def one(p):
        try:
            return call(client, model, p, use_temp[0])[0]
        except Exception as e:  # noqa: BLE001
            if use_temp[0] and "temperature" in str(e).lower():
                use_temp[0] = False
                return call(client, model, p, False)[0]
            raise

    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(one, prompts))


def rewrite(model: str, root: str) -> dict:
    data = json.loads((Path(root) / "cases.json").read_text())
    cs = [c for sp in ("dev", "test_seen", "test_exc_seen", "test_exc_unseen", "real") for c in data[sp]]
    outs = _map(model, [REWRITE.format(text=c["text"]) for c in cs])
    return {c["id"]: o.strip().splitlines()[0] if o.strip() else c["text"] for c, o in zip(cs, outs)}


def rerank(model: str, root: str, cands: dict) -> dict:
    """cands: {事例 id: [unit id ...]}(順位順)。返り値: {事例 id: [unit id ...](LLM の選んだ順)}。選ばれなかった候補は後ろに元の順で付ける。"""
    root = Path(root)
    data = json.loads((root / "cases.json").read_text())
    units = {u["id"]: u for u in json.loads((root / "sources" / "corpus.json").read_text())}
    cs = {c["id"]: c for sp in ("dev", "test_seen", "test_exc_seen", "test_exc_unseen", "real") for c in data[sp]}
    ids = list(cands)
    prompts = []
    for i in ids:
        lines = [f"[{n + 1}] 【{units[u]['cite']}】{units[u]['text'][:300]}" for n, u in enumerate(cands[i])]
        prompts.append(RERANK.format(text=cs[i]["text"], cands="\n".join(lines)))
    outs = _map(model, prompts)
    res = {}
    for i, o in zip(ids, outs):
        pick = []
        for n in re.findall(r"\d+", o):
            n = int(n) - 1
            if 0 <= n < len(cands[i]) and cands[i][n] not in pick:
                pick.append(cands[i][n])
        res[i] = pick + [u for u in cands[i] if u not in pick]
    return res
