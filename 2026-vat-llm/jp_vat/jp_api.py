"""Anthropic API で内外判定を評価する(論文が GPT-4o の API を使ったのに対応する商用モデルの比較対象)。

API キーは Modal の Secret `anthropic-secret` から環境変数で読む。キーの値はログにも結果にも出さない。
条件は plain / rag / mocked / long(fine-tuning は API で提供されないので、学習ありの条件は GPU 側の小型モデルで行う)。
"""
from __future__ import annotations

import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from jp_common import SPLITS, agg, make_contexts, prompt, score

MAX_TOKENS = 1500
STATE = {"thinking": {"type": "disabled"}, "max_tokens": MAX_TOKENS}


def call(client, model: str, p: str, use_temperature: bool) -> tuple[str, dict]:
    kw = dict(model=model, max_tokens=STATE["max_tokens"], messages=[{"role": "user", "content": p}])
    if use_temperature:
        kw["temperature"] = 0
    if STATE["thinking"]:
        kw["thinking"] = STATE["thinking"]  # 思考ブロックに出力の枠を使い切られないようにする(固定の短い応答で比べる)
    for attempt in range(8):
        try:
            r = client.messages.create(**kw)
            return "".join(b.text for b in r.content if getattr(b, "type", "") == "text"), \
                {"in": r.usage.input_tokens, "out": r.usage.output_tokens}
        except Exception as e:  # noqa: BLE001
            name = type(e).__name__
            if name == "BadRequestError" and "thinking" in str(e).lower() and "thinking" in kw:
                if "between_tools" in str(e) and kw["thinking"].get("type") != "between_tools":
                    STATE["thinking"] = kw["thinking"] = {"type": "between_tools"}  # このモデルは思考しない設定の指定が異なる
                else:
                    STATE["thinking"] = None
                    kw.pop("thinking")
                    kw["max_tokens"] = STATE["max_tokens"] = 4000
                continue
            if name in ("RateLimitError", "APIConnectionError", "InternalServerError", "APIStatusError", "OverloadedError") and attempt < 7:
                time.sleep(min(60, 2 ** attempt + random.random()))
                continue
            raise
    raise RuntimeError("unreachable")


def run(model: str, conds: list[str], root: str, limit: int | None = None, workers: int = 8, log=print) -> dict:
    import anthropic

    key_name = next((k for k in ("ANTHROPIC_API_KEY", "ANTHROPIC_KEY", "ANTHROPIC_SECRET") if os.environ.get(k)), None)
    if key_name is None:
        raise RuntimeError("Secret に API キーの環境変数が見つかりません: " + ", ".join(sorted(k for k in os.environ if "ANTHROPIC" in k.upper())))
    client = anthropic.Anthropic(api_key=os.environ[key_name], max_retries=2)
    root = Path(root)
    data = json.loads((root / "cases.json").read_text())
    units = json.loads((root / "sources" / "corpus.json").read_text())
    rag = json.loads((root / "sources" / "rag_contexts.json").read_text())
    by_id = {u["id"]: u for u in units}
    use_temp = True
    res: dict = {}
    tok = {"in": 0, "out": 0}
    for cond in conds:
        res[cond] = {}
        for sp in SPLITS:
            cs = data[sp][:limit] if limit else data[sp]
            ps = [prompt(c["text"], make_contexts(cond, c, by_id, units, rag)) for c in cs]

            def one(p):
                nonlocal use_temp
                try:
                    return call(client, model, p, use_temp)
                except Exception as e:  # noqa: BLE001
                    if use_temp and "temperature" in str(e).lower():
                        use_temp = False
                        return call(client, model, p, False)
                    raise

            t0 = time.time()
            with ThreadPoolExecutor(workers) as ex:
                outs = list(ex.map(one, ps))
            for _, u in outs:
                tok["in"] += u["in"]
                tok["out"] += u["out"]
            texts = [o for o, _ in outs]
            rows = [score(o, c) for o, c in zip(texts, cs)]
            m = agg(rows)
            m["sec_per_case"] = (time.time() - t0) / len(cs)
            m["by_family"] = {}
            for fam in sorted({c["family"] for c in cs}):
                sub = [r for r, c in zip(rows, cs) if c["family"] == fam]
                m["by_family"][fam] = {"n": len(sub), "acc": sum(r["correct"] for r in sub) / len(sub)}
            if sp == "real":
                m["items"] = [{"id": c["id"], "label": c["label"], "pred": r["pred"], "cite": r["cite"], "out": o[:300]}
                              for c, r, o in zip(cs, rows, texts)]
            else:
                m["examples"] = [{"text": c["text"][:200], "out": o[:300], "label": c["label"]} for c, o in list(zip(cs, texts))[:3]]
            res[cond][sp] = m
            log(f"  [{model}/{cond}/{sp}] acc {m['acc']:.3f} cite {m['cite']:.3f} strict {m['strict']:.3f} unparsed {m['unparsed']:.2f}")
    return {"cfg": {"name": f"api_{model}", "model": model, "conds": conds, "temperature_used": use_temp, "thinking": STATE["thinking"], "max_tokens": STATE["max_tokens"]}, "eval": res, "tokens": tok}
