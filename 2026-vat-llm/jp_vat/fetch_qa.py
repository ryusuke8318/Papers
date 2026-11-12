"""国税庁の質疑応答事例(消費税 04: 国内取引・国外取引)の本文を取得し sources/qa_raw.json に保存する。

  python jp_vat/fetch_qa.py

質疑応答事例は国の公表資料(照会要旨・回答要旨)。実務に近い事例として、結論(国内取引か国外取引か)を人手で付けて評価に使う。
"""
from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path

OUT = Path(__file__).parent / "sources"
URL = "https://www.nta.go.jp/law/shitsugi/shohi/04/{:02d}.htm"


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "paper-reproduction/1.0"})
    b = urllib.request.urlopen(req, timeout=60).read()
    for enc in ("utf-8", "cp932"):
        try:
            return b.decode(enc)
        except UnicodeDecodeError:
            pass
    raise RuntimeError("decode")


def clean(h):
    h = re.sub(r"<script.*?</script>|<style.*?</style>", "", h, flags=re.S)
    h = re.sub(r"<br\s*/?>|</p>|</div>|</li>|</tr>", "\n", h)
    h = re.sub(r"<[^>]+>", "", h)
    h = re.sub(r"&nbsp;", " ", h)
    return re.sub(r"\n\s*\n+", "\n", h).strip()


def main():
    rows = []
    for n in range(1, 16):
        t = clean(get(URL.format(n)))
        title = re.search(r"^(.*?)｜国税庁", t, flags=re.M)
        a = t.find("【照会要旨】")
        b = t.find("【回答要旨】")
        c = t.find("【関係法令通達】")
        if a < 0 or b < 0:
            print("skip", n)
            continue
        rows.append({"no": n, "title": title.group(1) if title else "", "inquiry": t[a + 6:b].strip(),
                     "answer": t[b + 6:c if c > b else None].strip()})
    (OUT / "qa_raw.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    for r in rows:
        print(f'--- {r["no"]:02d} {r["title"]}\n照会: {r["inquiry"][:700]}\n回答: {r["answer"][:900]}\n')


if __name__ == "__main__":
    main()
