"""EfficientRAG の純 Python 部分(モデル・GPU に依存しない)。

- 語ごとの 2 値ラベルと、LLM が抜き出した語列との対応づけ(`align_labels`)
- Filter の出力から次の問いを組み立てる処理(`assemble_query`)
- 反復検索のループ(`run_loop`): <Continue> のチャンクだけを候補プールに足し、全て <Terminate> か最大反復数で止まる
- 合成データの負例の選び方(`pick_negatives`)と、学習例の組み立て(`labeler_examples` / `filter_examples`)
- 指標: EM / F1、Recall@K と K、ブートストラップの区間
- Self-Ask / Iter-RetGen の出力の解析
"""
from __future__ import annotations

import random
import re
import string
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Sequence

# ------------------------------------------------------------------ 語とラベル
_PUNCT = string.punctuation + "“”‘’«»"


def split_words(text: str) -> list[str]:
    """空白で語に分ける(論文は spaCy。ここでは追加の依存を避け、空白区切りにする)。"""
    return text.split()


def norm_word(w: str) -> str:
    return w.strip(_PUNCT).lower()


def align_labels(words: Sequence[str], extracted: Sequence[str]) -> tuple[list[int], float]:
    """LLM が抜き出した語列 `extracted` を、元の語列 `words` に、順序を保って対応づけ、語ごとの 2 値ラベルにする。

    貪欲な部分列照合。`extracted` の語が、現在位置より後ろに見つからなければ、その語は無視する(LLM が語を足した場合)。
    戻り値: (ラベル, 照合できた語の割合)。割合が低いときは、LLM の出力が元の語列から外れているので、呼び出し側で捨てる。
    """
    labels = [0] * len(words)
    nw = [norm_word(w) for w in words]
    pos, hit, total = 0, 0, 0
    for e in extracted:
        ne = norm_word(e)
        if not ne:
            continue
        total += 1
        for j in range(pos, len(words)):
            if nw[j] == ne:
                labels[j] = 1
                pos = j + 1
                hit += 1
                break
    return labels, (hit / total if total else 0.0)


def align_two(q_words: Sequence[str], info_words: Sequence[str], extracted: Sequence[str]) -> tuple[list[int], list[int], float]:
    """Filter 用の対応づけ。抜き出された語を、まず質問の語列に順序を保って対応づけ、照合できなかった語を Info の語列に対応づける。

    論文のプロンプトは「質問から語を選び、続いて Info から語を選ぶ」ので、出力は 質問側の語、Info 側の語 の順に並ぶ。
    両方に出る語は質問側が優先される。戻り値: (質問側のラベル, Info 側のラベル, 照合できた語の割合)
    """
    ql = [0] * len(q_words)
    nq = [norm_word(w) for w in q_words]
    pos, rest, total = 0, [], 0
    for e in extracted:
        ne = norm_word(e)
        if not ne:
            continue
        total += 1
        for j in range(pos, len(q_words)):
            if nq[j] == ne:
                ql[j] = 1
                pos = j + 1
                break
        else:
            rest.append(e)
    il, _ = align_labels(info_words, rest)
    hit = sum(ql) + sum(il)
    return ql, il, (min(hit, total) / total if total else 0.0)


def select_words(words: Sequence[str], labels: Sequence[int]) -> list[str]:
    return [w for w, l in zip(words, labels) if l]


def assemble_query(q_words: Sequence[str], q_labels: Sequence[int], info_words: Sequence[str], info_labels: Sequence[int],
                   dedupe: bool = True) -> str:
    """Filter が選んだ語から次の問いを作る。質問側の語、続いて Info 側の語を、元の順序のまま空白でつなぐ。

    質問側で「既知の部分」と判定された語は落ち、Info 側の語がその位置(質問の末尾側)に入るので、
    「未知の部分の語 + 既知の情報の語」の形になる。`dedupe` が真なら、正規化した語が既出のものは Info 側から落とす。
    """
    out = select_words(q_words, q_labels)
    seen = {norm_word(w) for w in out}
    for w, l in zip(info_words, info_labels):
        if not l:
            continue
        n = norm_word(w)
        if dedupe and n in seen:
            continue
        seen.add(n)
        out.append(w)
    return re.sub(r"\s+", " ", " ".join(out)).strip()


# ------------------------------------------------------------------ 反復検索のループ
CONTINUE, TERMINATE = "Continue", "Terminate"


@dataclass
class LoopResult:
    pool: list[str] = field(default_factory=list)            # 候補プールのチャンク ID(発見順)
    info: dict[str, list[str]] = field(default_factory=dict)  # チャンク ID -> ラベルされた語
    iterations: int = 0                                         # 実際に回した反復数
    n_label_calls: int = 0                                      # Labeler に渡したチャンクの数
    n_filter_calls: int = 0
    queries: list[str] = field(default_factory=list)           # 発行した検索の問い


def run_loop(question: str,
             retrieve: Callable[[str, int], list[str]],
             label: Callable[[str, list[str]], list[tuple[float, list[str]]]],
             filt: Callable[[str, list[str]], str],
             k: int = 5, max_iter: int = 4, threshold: float = 0.5) -> LoopResult:
    """EfficientRAG の推論ループ。

    retrieve(query, k)            -> 上位 k 件のチャンク ID
    label(query, chunk_ids)       -> チャンクごとの (P(Continue), ラベルされた語)
    filt(question, info_words)    -> 次の問い(空文字列なら、その枝は止める)

    枝ごとに (問い, それまでに集めた Info の語) を持つ。P(Continue) >= threshold のチャンクだけをプールに足し、
    そのチャンクのラベル語を Info に足して、Filter で次の問いを作る。既にプールにあるチャンクの枝は延ばさない。
    全ての枝が <Terminate> になるか、`max_iter` 回の反復で止まる。
    """
    res = LoopResult()
    frontier: list[tuple[str, list[str]]] = [(question, [])]
    seen_queries = {question}
    for _ in range(max_iter):
        if not frontier:
            break
        res.iterations += 1
        nxt: list[tuple[str, list[str]]] = []
        for query, info in frontier:
            res.queries.append(query)
            ids = retrieve(query, k)
            if not ids:
                continue
            outs = label(query, ids)
            res.n_label_calls += len(ids)
            for cid, (p, words) in zip(ids, outs):
                if p < threshold or cid in res.info:     # <Terminate>、または既にプールにある
                    continue
                res.pool.append(cid)
                res.info[cid] = list(words)
                new_info = info + list(words)
                res.n_filter_calls += 1
                q2 = filt(question, new_info).strip()
                if q2 and q2 not in seen_queries:
                    seen_queries.add(q2)
                    nxt.append((q2, new_info))
        frontier = nxt
    return res


# ------------------------------------------------------------------ 合成データ
def pick_negatives(ranked_ids: Sequence[str], gold_ids: set[str], n: int = 1) -> list[str]:
    """検索結果(類似度の降順)から、質問の正解チャンクではない上位 n 件を、難しい負例として返す。"""
    out: list[str] = []
    for cid in ranked_ids:
        if cid in gold_ids:
            continue
        out.append(cid)
        if len(out) == n:
            break
    return out


def labeler_example(query: str, chunk_words: list[str], labels: list[int] | None, tag: int) -> dict:
    return {"a": split_words(query), "b": list(chunk_words), "la": None,
            "lb": list(labels) if labels is not None else [0] * len(chunk_words), "tag": tag}


def filter_example(question: str, info_words: list[str], q_labels: list[int], info_labels: list[int]) -> dict:
    return {"a": split_words(question), "b": list(info_words), "la": list(q_labels), "lb": list(info_labels), "tag": None}


def info_chain(subs: dict, info_of: dict[str, list[str]], sid: str, _seen: set | None = None) -> list[str]:
    """sid の祖先(推移的な依存先)のラベル語を、依存の順に連結する。"""
    seen = _seen if _seen is not None else {sid}
    out: list[str] = []
    for d in subs[sid]["deps"]:
        if d in seen or d not in subs:
            continue
        seen.add(d)
        out += info_chain(subs, info_of, d, seen)
        out += info_of.get(d, [])
    return out


def build_examples(record: dict, chunk_words: dict[str, list[str]], negatives,
                   min_match: float = 0.6) -> tuple[list[dict], list[dict], dict]:
    """1 問分の合成結果から、Labeler と Filter の学習例を作る。

    record: {"question", "subs": {sid: {"doc": chunk_id, "deps": [sid...], "labeled": [words...] | None (Labeler 用の語),
             "filtered": [words...] | None (Filter 用の語)}}}
      - 依存のない副問い: Labeler の入力の問いは原問い。
      - 依存のある副問い: Filter が選んだ語(原問い + 依存先の Info)から次の問いを作り、それを Labeler の入力の問いにする。
    negatives: 副問いの ID -> 難しい負例のチャンク ID の辞書、または (副問い ID, 入力の問い) -> チャンク ID の列を返す関数。
               負例は <Terminate>、語ラベルはすべて 0。
    戻り値: (labeler 例, filter 例, 統計)
    """
    q = record["question"]
    q_words = split_words(q)
    lab, fil = [], []
    st = Counter()
    subs = record["subs"]
    info_of: dict[str, list[str]] = {}

    # 依存の浅い順に処理する
    order = sorted(subs, key=lambda s: _depth(subs, s))
    for sid in order:
        s = subs[sid]
        words = chunk_words[s["doc"]]
        l_labels, l_rate = align_labels(words, s.get("labeled") or [])
        if l_rate < min_match or sum(l_labels) == 0:
            st["labeler_dropped"] += 1
            continue
        info_of[sid] = select_words(words, l_labels)
        if not s["deps"]:
            query = q
        else:
            info = info_chain(subs, info_of, sid)
            if not info or not s.get("filtered"):
                st["filter_dropped"] += 1
                continue
            ql, il, f_rate = align_two(q_words, info, s["filtered"])
            if f_rate < min_match or sum(ql) + sum(il) == 0:
                st["filter_dropped"] += 1
                continue
            query = assemble_query(q_words, ql, info, il)
            if not query:
                st["filter_dropped"] += 1
                continue
            fil.append(filter_example(q, info, ql, il))
            st["filter"] += 1
        lab.append(labeler_example(query, words, l_labels, 1))
        st["labeler_pos"] += 1
        for nid in (negatives(sid, query) if callable(negatives) else negatives.get(sid, [])):
            lab.append(labeler_example(query, chunk_words[nid], None, 0))
            st["labeler_neg"] += 1
    return lab, fil, dict(st)


def _depth(subs: dict, sid: str, _seen: tuple = ()) -> int:
    """依存の深さ(依存なし = 0)。循環があっても止まるよう、辿った経路を持つ。"""
    if sid in _seen:
        return 0
    deps = [d for d in subs[sid]["deps"] if d in subs]
    return 1 + max(_depth(subs, d, _seen + (sid,)) for d in deps) if deps else 0


def valid_decomposition(subs: dict, gold_docs: Sequence[str]) -> bool:
    """分解の検査: 各正解チャンクがちょうど 1 つの副問いに対応し、依存先が存在し、循環しない。"""
    docs = [s["doc"] for s in subs.values()]
    if sorted(docs) != sorted(gold_docs):
        return False
    for sid, s in subs.items():
        for d in s["deps"]:
            if d not in subs or d == sid:
                return False

    def acyclic(sid: str, path: tuple) -> bool:
        if sid in path:
            return False
        return all(acyclic(d, path + (sid,)) for d in subs[sid]["deps"])
    return all(acyclic(s, ()) for s in subs)


# ------------------------------------------------------------------ 指標
def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))


def f1_score(pred: str, gold: str) -> float:
    p, g = normalize_answer(pred).split(), normalize_answer(gold).split()
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    n = sum(common.values())
    if n == 0:
        return 0.0
    prec, rec = n / len(p), n / len(g)
    return 2 * prec * rec / (prec + rec)


def recall_at_pool(pool: Sequence[str], gold: Sequence[str]) -> float:
    """候補プールに含まれる正解チャンクの割合(Recall@K の K は、プールの大きさ len(set(pool)))。"""
    if not gold:
        return 0.0
    ps = set(pool)
    return sum(g in ps for g in gold) / len(gold)


def pool_size(pool: Sequence[str]) -> int:
    return len(set(pool))


def bootstrap_ci(values: Sequence[float], n_boot: int = 10000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float, float]:
    """平均と、質問単位のブートストラップの区間(パーセンタイル法)。"""
    import numpy as np
    vals = np.asarray(list(values), dtype=float)
    n = len(vals)
    rng = np.random.default_rng(seed)
    means = np.sort(vals[rng.integers(0, n, (n_boot, n))].mean(axis=1))
    return float(vals.mean()), float(means[int(alpha / 2 * n_boot)]), float(means[int((1 - alpha / 2) * n_boot) - 1])


def paired_bootstrap(a: Sequence[float], b: Sequence[float], n_boot: int = 10000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float, float]:
    """同じ質問での差 mean(a - b) と、その 95% 区間。"""
    assert len(a) == len(b)
    return bootstrap_ci([x - y for x, y in zip(a, b)], n_boot, seed, alpha)


# ------------------------------------------------------------------ 基準手法の出力の解析
_FOLLOW = re.compile(r"follow[- ]?up\s*:\s*(.+)", re.I)
_FINAL = re.compile(r"so the final answer is\s*:?\s*(.+)", re.I)


def parse_selfask(text: str) -> tuple[str, str]:
    """Self-Ask の生成を解析する。("final", 答え) / ("follow_up", 副問い) / ("none", "")。最初に現れた方を採る。"""
    mf, ma = _FOLLOW.search(text), _FINAL.search(text)
    if mf and (not ma or mf.start() < ma.start()):
        return "follow_up", mf.group(1).strip().split("\n")[0].strip()
    if ma:
        return "final", ma.group(1).strip().split("\n")[0].strip().rstrip(".")
    return "none", ""


def extract_json_value(text: str, key: str) -> str | None:
    """JSON 風の出力から、文字列の値を取り出す(壊れた JSON でも、`"key": "..."` の形なら拾う)。"""
    import json
    s, e = text.find("{"), text.rfind("}")
    if s != -1 and e > s:
        try:
            v = json.loads(text[s:e + 1]).get(key)
            if v is not None:
                return str(v)
        except Exception:  # noqa: BLE001
            pass
    m = re.search(r'"%s"\s*:\s*"((?:[^"\\]|\\.)*)"' % re.escape(key), text)
    return m.group(1) if m else None
