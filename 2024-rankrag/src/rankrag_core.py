"""RankRAG 縮小再現の共通処理(標準ライブラリ + numpy + scipy のみ。GPU・torch は使わない)。

- プロンプト: 論文 Table 1 の (x, c, y) 形式 4 種(context-rich QA / retrieval-augmented QA / context ranking /
  retrieval-augmented ranking)
- BM25: 学習データの hard negative と、評価時の retriever に使う(scipy の疎行列)
- 指標: EM / F1 / 部分一致(SQuAD 公式の正規化)、ranking の recall@k と MRR
"""
from __future__ import annotations

import math
import re
import string
from collections import Counter

import numpy as np
from scipy import sparse

QA_INSTR = "Answer the following question from context."
MAX_WORDS = 160  # 1 つの passage を切り詰める単語数(仮定: 論文に記載なし)


# ---------------------------------------------------------------- プロンプト(Table 1)
def truncate_words(text: str, n: int = MAX_WORDS) -> str:
    w = text.split()
    return " ".join(w[:n])


def qa_prompt(question: str, passages: list[str]) -> str:
    """Context-rich QA(passage 1 件)/ Retrieval-augmented QA(passage 5 件)。"""
    if len(passages) == 1:
        body = f"Passage: {passages[0]}"
    else:
        body = "\n".join(f"Passage {i + 1}: {p}" for i, p in enumerate(passages))
    return f"{QA_INSTR} {question}\n\n{body}"


def rank_prompt(question: str, passage: str) -> str:
    """Context ranking。論文の文面(access whether ...)のまま。出力は True / False。"""
    return f"For the question {question}, access whether the passage is relevant to the question.\n\nPassage: {passage}"


def rar_prompt(question: str, passages: list[str]) -> str:
    """Retrieval-augmented ranking。出力は関連する passage の番号。"""
    body = "\n".join(f"Passage {i + 1}: {p}" for i, p in enumerate(passages))
    return f"For the question {question}, find all passages from the context that are relevant to the question.\n\n{body}"


def parse_indexes(text: str, n: int) -> list[int]:
    """RAR の出力("2, 4" など)から 0 始まりの番号を取り出す。"""
    out = []
    for m in re.findall(r"\d+", text):
        i = int(m) - 1
        if 0 <= i < n and i not in out:
            out.append(i)
    return out


# ---------------------------------------------------------------- 答えの評価(SQuAD 公式の正規化)
def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(pred: str, golds: list[str]) -> float:
    p = normalize_answer(pred)
    return float(any(p == normalize_answer(g) for g in golds))


def _f1(pred: str, gold: str) -> float:
    pt, gt = normalize_answer(pred).split(), normalize_answer(gold).split()
    common = Counter(pt) & Counter(gt)
    ns = sum(common.values())
    if ns == 0:
        return 0.0
    prec, rec = ns / len(pt), ns / len(gt)
    return 2 * prec * rec / (prec + rec)


def f1_score(pred: str, golds: list[str]) -> float:
    return max(_f1(pred, g) for g in golds)


def contains_answer(pred: str, golds: list[str]) -> float:
    """出力に正解文字列が含まれるか(冗長な出力にも寛容な指標。未調整の instruct モデル向け)。"""
    p = normalize_answer(pred)
    return float(any(normalize_answer(g) and normalize_answer(g) in p for g in golds))


def answer_in_text(answers: list[str], text: str) -> bool:
    t = text.lower()
    return any(a.lower() in t for a in answers if a)


# ---------------------------------------------------------------- ranking の指標
def recall_at_k(ranked: list, gold: set, k: int) -> float:
    """gold のうち上位 k 件に入った割合(HotpotQA は gold が 2 件)。"""
    if not gold:
        return 0.0
    return len(set(ranked[:k]) & gold) / len(gold)


def mrr(ranked: list, gold: set) -> float:
    for i, r in enumerate(ranked):
        if r in gold:
            return 1.0 / (i + 1)
    return 0.0


# ---------------------------------------------------------------- BM25
_STOP = set("a an the of in on at to for and or is are was were be been by with as from that this it its which who whom "
            "what when where how why do does did has have had not no but if then than so such into about over after "
            "before between during".split())


def tokenize(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP]


class BM25:
    """疎行列で実装した BM25(k1=1.5, b=0.75)。文書数 数万件・クエリ数 数千件で十分に速い。"""

    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.vocab: dict[str, int] = {}
        rows, cols, vals = [], [], []
        lens = []
        for i, d in enumerate(docs):
            tf = Counter(tokenize(d))
            lens.append(sum(tf.values()))
            for w, c in tf.items():
                j = self.vocab.setdefault(w, len(self.vocab))
                rows.append(i)
                cols.append(j)
                vals.append(c)
        n = len(docs)
        tf = sparse.csr_matrix((vals, (rows, cols)), shape=(n, len(self.vocab)), dtype=np.float32)
        df = np.asarray((tf > 0).sum(0)).ravel()
        self.idf = np.log(1 + (n - df + 0.5) / (df + 0.5)).astype(np.float32)
        lens = np.asarray(lens, dtype=np.float32)
        avg = max(float(lens.mean()), 1.0)
        tf = tf.tocoo()
        denom = tf.data + k1 * (1 - b + b * lens[tf.row] / avg)
        w = tf.data * (k1 + 1) / denom * self.idf[tf.col]
        self.mat = sparse.csc_matrix((w, (tf.row, tf.col)), shape=tf.shape)  # 文書 × 語彙(列単位で引く)
        self.n = n

    def scores(self, query: str) -> np.ndarray:
        s = np.zeros(self.n, dtype=np.float32)
        for w in set(tokenize(query)):
            j = self.vocab.get(w)
            if j is not None:
                col = self.mat[:, j]
                s[col.indices] += col.data
        return s

    def topk(self, query: str, k: int) -> list[int]:
        s = self.scores(query)
        k = min(k, self.n)
        idx = np.argpartition(-s, k - 1)[:k]
        return [int(i) for i in idx[np.argsort(-s[idx], kind="stable")]]


def seeded_shuffle(items: list, seed: int) -> list:
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(items))
    return [items[i] for i in order]


def safe_mean(xs: list[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else math.nan
