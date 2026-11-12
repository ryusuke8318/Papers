"""内外判定に関係する日本の法令・通達を取得し、条文の単位(unit)に分けて sources/corpus.json に保存する。

  python jp_vat/fetch_sources.py

取得元(いずれも公開資料。法令・通達は著作権法 13 条により権利の対象外)
  - 消費税法 第 4 条、消費税法施行令 第 6 条: e-Gov 法令 API v2
  - 消費税法基本通達 第 5 章第 7 節(国内取引の判定): 国税庁サイト
各 unit は {id, cite, text}。cite は引用に使う表記(例: 令第6条第1項第1号)。
"""
from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "sources"
LAW = "https://laws.e-gov.go.jp/api/2/law_data/{}?response_format=json"
TSUTATSU = "https://www.nta.go.jp/law/tsutatsu/kihon/shohi/05/07.htm"
KANJI = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}


def get(url: str) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "paper-reproduction/1.0"}), timeout=60) as r:
        return r.read()


def text_of(n) -> str:
    if isinstance(n, str):
        return n
    if isinstance(n, dict):
        return "".join(text_of(c) for c in n.get("children", []))
    if isinstance(n, list):
        return "".join(text_of(c) for c in n)
    return ""


def find(n, tag, num):
    if isinstance(n, dict):
        if n.get("tag") == tag and n.get("attr", {}).get("Num") == num:
            return n
        for c in n.get("children", []):
            r = find(c, tag, num)
            if r:
                return r
    return None


def kids(n, tag):
    return [c for c in n.get("children", []) if isinstance(c, dict) and c.get("tag") == tag]


def own_text(n) -> str:
    """子の Item / Subitem を含まない、その項・号自身の文。"""
    out = []
    for c in n.get("children", []):
        if isinstance(c, dict) and c.get("tag") in ("ParagraphSentence", "ItemSentence", "Subitem1Sentence", "Column", "Sentence"):
            out.append(text_of(c))
    return "".join(out)


def law_units(law_id: str, article: str, prefix: str, short: str) -> list[dict]:
    d = json.loads(get(LAW.format(law_id)))
    art = find(d["law_full_text"], "Article", article)
    units = []
    for p in kids(art, "Paragraph"):
        pn = p["attr"].get("Num", "1")
        base = f"{short}第{article}条" + (f"第{pn}項" if len(kids(art, "Paragraph")) > 1 else "")
        head = own_text(p)
        units.append({"id": f"{prefix}-{article}-{pn}", "cite": base, "text": head})
        for it in kids(p, "Item"):
            inum = it["attr"].get("Num")
            units.append({"id": f"{prefix}-{article}-{pn}-{inum}", "cite": f"{base}第{inum}号",
                          "text": text_of(it)})
    return units


def tsutatsu_units() -> list[dict]:
    t = get(TSUTATSU).decode("cp932")
    t = re.sub(r"<script.*?</script>|<style.*?</style>", "", t, flags=re.S)
    t = re.sub(r"<[^>]+>", "\n", t)
    t = re.sub(r"[ \t　]+\n", "\n", t)
    t = re.sub(r"\n\s*\n+", "\n", t)
    parts = re.split(r"\n(5[－-]7[－-]\d+(?:の\d+)?)\s*\n", t)
    units = []
    for i in range(1, len(parts) - 1, 2):
        no = parts[i].replace("-", "－")
        body = parts[i + 1]
        body = body.split("\nページの先頭へ")[0].split("\nこのページの先頭")[0]
        lines = [l.strip() for l in body.split("\n") if l.strip()]
        # 末尾に次の通達の見出し「（…）」が付く場合は落とす
        while lines and re.fullmatch(r"（[^）]*）", lines[-1]):
            lines.pop()
        body = "\n".join(lines)
        if "により削除" in body[:40]:
            continue
        units.append({"id": "通達-" + no.replace("－", "-"), "cite": f"基本通達{no}", "text": body})
    return units


def main():
    OUT.mkdir(exist_ok=True)
    units = law_units("363AC0000000108", "4", "法", "消費税法") + law_units("363CO0000000360", "6", "令", "令")
    units += tsutatsu_units()
    for u in units:
        u["text"] = re.sub(r"\s+", " ", u["text"]).strip()
    (OUT / "corpus.json").write_text(json.dumps(units, ensure_ascii=False, indent=1))
    print(len(units), "units;", sum(len(u["text"]) for u in units), "文字")
    for u in units:
        print(f'{u["id"]:14s} {u["cite"]:22s} {len(u["text"]):5d} {u["text"][:50]}')


if __name__ == "__main__":
    main()
