"""RASER(Recoverability-Aware Selective Escalation Router)の、LLM に依存しない部分。

特徴量 6 つ、PRUNE の検証器の規則、ラベル、費用つき選択、質問単位の交差検証、指標、ブートストラップ。
純 Python + numpy + scikit-learn だけで動く。LLM の呼び出しと検索は raser_pipeline.py 側に分けてある。
"""
from __future__ import annotations

import json
import re
import string
from collections import Counter

import numpy as np

ROUTES = ("one_shot", "prune", "ircot")
FEATURES = ("confidence", "ans_len", "bridge_cues", "score_gap", "score_top1", "qtype")
QTYPES = {"entity": 0, "date": 1, "yes_no": 2, "count": 3, "other": 4}

# ------------------------------------------------------------------ 答えの F1(SQuAD の正規化)


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def f1_score(pred: str, gold: str) -> float:
    p, g = normalize_answer(pred).split(), normalize_answer(gold).split()
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    n = sum(common.values())
    if n == 0:
        return 0.0
    pr, rc = n / len(p), n / len(g)
    return 2 * pr * rc / (pr + rc)


def f1_max(pred: str, golds: list[str]) -> float:
    return max((f1_score(pred, g) for g in golds), default=0.0)


# ------------------------------------------------------------------ 段落の分割(文で区切り、128 トークン程度、16 トークン重なり)


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])", text.strip())
    return [p for p in parts if p]


def chunk_passage(text: str, count, max_tokens: int = 128, overlap: int = 16) -> list[str]:
    """文を貪欲に詰めて max_tokens を超えない塊にする。次の塊の先頭に、直前の塊の末尾の文を overlap トークン以内で重ねる。

    count は、文字列 -> トークン数の関数。1 文が max_tokens を超えるときは、その文だけで 1 塊にする。
    """
    sents = split_sentences(text)
    chunks: list[list[str]] = []
    cur: list[str] = []
    cur_n = 0
    for s in sents:
        n = count(s)
        if cur and cur_n + n > max_tokens:
            chunks.append(cur)
            tail: list[str] = []
            tn = 0
            for t in reversed(cur):
                c = count(t)
                if tn + c > overlap:
                    break
                tail.insert(0, t)
                tn += c
            cur, cur_n = tail, tn
        cur.append(s)
        cur_n += n
    if cur:
        chunks.append(cur)
    return [" ".join(c) for c in chunks]


# ------------------------------------------------------------------ 特徴量(論文 Table 2)
_YESNO = ("is", "are", "was", "were", "do", "does", "did", "can", "could", "has", "have", "had", "will", "would", "should")
_ABSTAIN = re.compile(r"(don't|do not|dont|cannot|can't) (know|find|determine)|\bunknown\b|not (mentioned|provided|specified|found)|no information", re.I)


def is_abstain(answer: str) -> bool:
    return bool(_ABSTAIN.search(answer))


def question_type(q: str) -> str:
    """規則で問いの形を決める。yes-no > count > date > entity > other の順に判定する。"""
    ql = q.lower().strip()
    w = re.findall(r"[a-z']+", ql)
    first = w[0] if w else ""
    if first in _YESNO:
        return "yes_no"
    if re.search(r"\bhow (many|much)\b|\bnumber of\b", ql):
        return "count"
    if re.search(r"\bwhen\b|\bwhat (year|date|day|month)\b|\bwhich (year|date|day|month)\b|\bdate of\b|\bborn (first|earlier|later)\b|\b(older|younger|earlier|later)\b", ql):
        return "date"
    if first in ("who", "whom", "whose", "where", "which", "what") or re.search(r"\b(who|whom|where)\b", ql):
        return "entity"
    return "other"


_BRIDGE_PATTERNS = [
    re.compile(r"\b\w+(\s+\w+){0,3}\s+of\s+(the\s+)?[^,?]*\bof\b", re.I),                       # X of the Y of Z
    re.compile(r"\bwho\b.*\b(was born|born|lived|died|located|founded|directed|wrote)\b.*\b(where|which|what)\b", re.I),
    re.compile(r"\bthe (country|city|state|place|town|year|county|province|region) (where|in which|that)\b", re.I),
    re.compile(r"\b(the|a) (director|author|founder|performer|composer|producer|mother|father|spouse|husband|wife|capital|birthplace) of\b", re.I),
    re.compile(r"'s\b.*'s\b"),                                                                    # 所有格が 2 回
]


def bridge_cues(q: str) -> int:
    """多段の言い回しに 1 つでも一致すれば 1。"""
    return int(any(p.search(q) for p in _BRIDGE_PATTERNS))


_PRIOR = {"entity": 0.6, "date": 0.8, "yes_no": 0.9, "count": 0.7, "other": 0.5}


def confidence(answer: str, qtype: str, chunk_texts: list[str]) -> float:
    """規則ベースの信頼度。論文は定義の式を示していないので、次のように置いた。
    答えが「分からない」なら 0。そうでなければ問いの形ごとの事前値(entity 0.6、date 0.8、yes-no 0.9、count 0.7、other 0.5)に、
    答えの文字列が検索された塊に含まれるときだけ 0.2 を足す(上限 1.0)。yes-no は含まれ判定の対象外。
    """
    if is_abstain(answer):
        return 0.0
    c = _PRIOR[qtype]
    a = normalize_answer(answer)
    if qtype != "yes_no" and a and any(a in normalize_answer(t) for t in chunk_texts):
        c += 0.2
    return min(c, 1.0)


def extract_features(question: str, answer: str, scores: list[float], chunk_texts: list[str]) -> dict:
    """1 回目の答え・検索スコア(降順の上位 10)・質問文から、6 つの特徴量を作る。追加の LLM 呼び出しは要らない。"""
    qt = question_type(question)
    s = sorted(scores, reverse=True)
    top5 = s[4] if len(s) >= 5 else s[-1]
    return {"confidence": confidence(answer, qt, chunk_texts), "ans_len": len(answer.split()), "bridge_cues": bridge_cues(question),
            "score_gap": float(s[0] - top5), "score_top1": float(s[0]), "qtype": QTYPES[qt]}


def feature_matrix(rows: list[dict]) -> np.ndarray:
    return np.array([[r["features"][f] for f in FEATURES] for r in rows], dtype=float)


# ------------------------------------------------------------------ PRUNE の部品
def parse_bridges(text: str, max_n: int = 2) -> list[str]:
    """提案者の出力(JSON オブジェクト)から、bridge_entity を最大 max_n 個取り出す。

    1 行 1 JSON でも、複数行の JSON やコードフェンスつきでも読めるよう、"bridge_entity": "..." の値を正規表現で拾う。
    壊れた出力は 0 個になる。大文字小文字を無視した重複は除く。
    """
    out: list[str] = []
    for m in re.finditer(r'"bridge_entity"\s*:\s*"((?:[^"\\]|\\.)*)"', text):
        e = m.group(1).replace('\\"', '"').strip()
        if e and e.lower() not in {x.lower() for x in out}:
            out.append(e)
        if len(out) >= max_n:
            break
    return out


def jaccard(a, b) -> float:
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a | b) else 0.0


def verify_branch(entity: str, branch_ids: list[int], one_shot_ids: list[int], chunk_texts: dict[int, str],
                  min_novelty: float = 0.05, min_support: float = 0.05) -> dict:
    """規則ベースの検証器(LLM は呼ばない)。
    novelty : 枝で検索した塊のうち、1 回目の上位に無かった塊の割合
    support : その新しい塊のうち、橋渡しの実体の名前を含むものの割合
    info_gain: 1 - Jaccard(枝の塊, 1 回目の塊)
    novelty か support が下限未満なら、その枝を捨てる。
    """
    if not branch_ids:
        return {"novelty": 0.0, "support": 0.0, "info_gain": 0.0, "keep": False}
    one = set(one_shot_ids)
    new = [i for i in branch_ids if i not in one]
    novelty = len(new) / len(branch_ids)
    ent = entity.lower()
    support = (sum(ent in chunk_texts[i].lower() for i in new) / len(new)) if new else 0.0
    return {"novelty": novelty, "support": support, "info_gain": 1.0 - jaccard(branch_ids, one_shot_ids),
            "keep": novelty >= min_novelty and support >= min_support}


# ------------------------------------------------------------------ ラベルと費用つき選択
def bridgeable_label(f1_prune, f1_one, tau: float = 0.1):
    """y = 1[F1_PRUNE - F1_one-shot > tau](論文の式 2)。等しい場合は 0。"""
    return (np.asarray(f1_prune, float) - np.asarray(f1_one, float) > tau).astype(int)


def cost_aware_argmax(pred_f1: np.ndarray, costs: np.ndarray, lam) -> np.ndarray:
    """r* = argmax_r [ f_r(s) - lam * c_r ](論文の式 1)。同点は、先頭(費用が最小の経路)を選ぶ。
    pred_f1: (n, 3)、costs: (3,) または (n, 3)、lam: スカラーまたは (n,)。戻り値: 経路の番号(n,)
    """
    pred_f1 = np.asarray(pred_f1, float)
    lam = np.asarray(lam, float)
    if lam.ndim == 1:
        lam = lam[:, None]
    return np.argmax(pred_f1 - lam * np.asarray(costs, float), axis=1)


def choose_lambda(pred_f1: np.ndarray, tokens: np.ndarray, cbar: np.ndarray, budget: float, grid: np.ndarray) -> float:
    """訓練分割で、平均トークンが「常に IRCOT* の平均トークン × budget」以下になる最小の lam を選ぶ。
    (論文は「最大の lam」と書くが、上限を守る範囲で最も多く使う設定、つまり最小の lam と解釈した。)
    """
    limit = budget * tokens[:, 2].mean()
    for lam in grid:
        r = cost_aware_argmax(pred_f1, cbar, lam)
        if tokens[np.arange(len(r)), r].mean() <= limit:
            return float(lam)
    return float(grid[-1])


def escalate(prob, theta: float = 0.20):
    """RASER-2 の判定。p(BRIDGEABLE) >= theta なら 1(PRUNE に上げる)、そうでなければ 0(1 回目の答えを返す)。"""
    return (np.asarray(prob, float) >= theta).astype(int)


def match_tokens_r2(p2: np.ndarray, T: np.ndarray, target: float) -> np.ndarray:
    """平均トークンが target になるよう、p2 の高い順に PRUNE へ上げる(経路は 0 か 1)。費用は、経路ごとの平均で見積もる。"""
    n = len(p2)
    t0, t1 = T[:, 0].mean(), T[:, 1].mean()
    m = int(round(n * min(max((target - t0) / (t1 - t0), 0.0), 1.0)))
    r = np.zeros(n, int)
    r[np.argsort(-p2, kind="stable")[:m]] = 1
    return r


def match_tokens_r3(pred3: np.ndarray, cbar: np.ndarray, T: np.ndarray, target: float, grid: np.ndarray | None = None) -> np.ndarray:
    """平均トークンが target に最も近くなる lambda(全質問で共通)で、費用つき argmax を行う。"""
    best, best_d = None, 1e18
    for lam in (default_lam_grid() if grid is None else grid):
        r = cost_aware_argmax(pred3, cbar, lam)
        d = abs(T[np.arange(len(r)), r].mean() - target)
        if d < best_d:
            best, best_d = r, d
    return best


# ------------------------------------------------------------------ 質問単位の交差検証
def group_folds(groups, k: int = 5, seed: int = 0) -> np.ndarray:
    """同じ group(質問)の行は、同じ分割に入る。行ごとに分割の番号を返す。"""
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    rng = np.random.RandomState(seed)
    rng.shuffle(uniq)
    fold_of = {g: i % k for i, g in enumerate(uniq)}
    return np.array([fold_of[g] for g in groups])


def make_gbm_clf(seed=0):
    from sklearn.ensemble import GradientBoostingClassifier
    return GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, subsample=0.8, random_state=seed)


def make_gbm_reg(seed=0):
    from sklearn.ensemble import GradientBoostingRegressor
    return GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, subsample=0.8, random_state=seed)


def _proba(clf, X, y):
    if len(set(y)) < 2:       # 訓練分割に片方のクラスしか無いとき
        return np.full(len(X), float(y[0]))
    return clf.predict_proba(X)[:, 1]


def cv_router(X, F1, T, groups, tau=0.1, k=5, seed=0, budgets=(), lam_grid=None, make_clf=make_gbm_clf, make_reg=make_gbm_reg, cols=None):
    """質問単位の k 分割交差検証で、RASER-2 の確率と RASER-3 の予測 F1 を、評価側の行について作る。

    X: (n, 6) 特徴量、F1: (n, 3) 経路ごとの F1、T: (n, 3) 経路ごとのトークン数(累積)、cols: 使う特徴量の列(アブレーション用)
    戻り値: p2 (n,)、pred3 (n, 3)、cbar (n, 3)=訓練分割の経路ごとの平均トークン、lam[b] (n,)=予算 b のときの訓練分割で選んだ lam、fold (n,)
    """
    X = np.asarray(X, float)
    if cols is not None:
        X = X[:, cols]
    F1, T = np.asarray(F1, float), np.asarray(T, float)
    n = len(X)
    fold = group_folds(groups, k, seed)
    y = bridgeable_label(F1[:, 1], F1[:, 0], tau)
    p2, pred3, cbar = np.zeros(n), np.zeros((n, 3)), np.zeros((n, 3))
    lam = {b: np.zeros(n) for b in budgets}
    grid = lam_grid if lam_grid is not None else default_lam_grid()
    for f in range(k):
        tr, te = fold != f, fold == f
        clf = make_clf(seed).fit(X[tr], y[tr]) if len(set(y[tr])) > 1 else None
        p2[te] = _proba(clf, X[te], y[tr])
        regs = [make_reg(seed).fit(X[tr], F1[tr, r]) for r in range(3)]
        pred3[te] = np.column_stack([m.predict(X[te]) for m in regs])
        cbar[te] = T[tr].mean(axis=0)
        if budgets:
            ptr = np.column_stack([m.predict(X[tr]) for m in regs])
            for b in budgets:
                lam[b][te] = choose_lambda(ptr, T[tr], T[tr].mean(axis=0), b, grid)
    return {"p2": p2, "pred3": pred3, "cbar": cbar, "lam": lam, "fold": fold, "y": y}


def default_lam_grid() -> np.ndarray:
    return np.concatenate([[0.0], np.geomspace(1e-6, 1e-3, 60)])


# ------------------------------------------------------------------ 指標
def route_stats(route: np.ndarray, F1: np.ndarray, T: np.ndarray, idx=None) -> dict:
    """経路の選択から、平均 F1・平均トークン・経路の比率を出す。idx はブートストラップの再標本。"""
    route = np.asarray(route)
    if idx is not None:
        route, F1, T = route[idx], F1[idx], T[idx]
    r = np.arange(len(route))
    return {"f1": float(F1[r, route].mean()), "tokens": float(T[r, route].mean()), "mix": [float((route == j).mean()) for j in range(F1.shape[1])]}


def oracle_route(F1: np.ndarray, T: np.ndarray) -> np.ndarray:
    """F1 が最大の経路。同点は、トークンが少ない方(つまり先の経路)。"""
    return np.argmax(np.round(F1, 9), axis=1)


def recoverability(F1: np.ndarray, gain_thr: float = 0.1, correct_thr: float = 0.5) -> dict:
    """論文 Table 1 の分類。F1 は (n, 3)=[one_shot, prune, ircot]。

    bridge helps : max(PRUNE, IRCOT*) - one_shot >= gain_thr
    one-shot correct: 上に当てはまらず、one_shot >= correct_thr
    no method helps: 上に当てはまらず、どの経路も one_shot を上回らない
    small gain   : 残り(わずかに上回るが 0.1 未満)
    """
    one = F1[:, 0]
    gain = F1[:, 1:].max(axis=1) - one
    helps = gain >= gain_thr - 1e-12
    correct = ~helps & (one >= correct_thr)
    nohelp = ~helps & ~correct & (gain <= 1e-12)
    small = ~(helps | correct | nohelp)
    return {"one_shot_correct": correct.mean(), "no_method_helps": nohelp.mean(), "small_gain": small.mean(), "bridge_helps": helps.mean()}


def bootstrap(fn, n: int, B: int = 2000, seed: int = 0, level: float = 0.95):
    """質問の添字を復元抽出して fn(idx) を B 回評価し、(点推定, 下限, 上限) を返す。fn がスカラーでも配列でもよい。"""
    rng = np.random.RandomState(seed)
    point = np.asarray(fn(np.arange(n)), float)
    vals = np.array([np.asarray(fn(rng.randint(0, n, n)), float) for _ in range(B)])
    a = (1 - level) / 2
    return point, np.quantile(vals, a, axis=0), np.quantile(vals, 1 - a, axis=0)
