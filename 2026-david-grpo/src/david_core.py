"""DAVID-GRPO の中核(依存は標準ライブラリのみ)。

軌跡の解析、報酬(正解一致、証拠の再現率)、グループ内の相対的な利点、
grounded expansion の判定、warm-start の混合グループの利点を扱う。
モデルや GPU には触れない。
"""
from __future__ import annotations

import re
import string
from dataclasses import dataclass, field

MAX_SEARCHES = 3

SYSTEM_PROMPT = "You are a careful research assistant. You answer multi-hop questions with a search tool."

PROMPT_TEMPLATE = (
    "Answer the question using the search tool.\n"
    "Rules:\n"
    "- Think briefly inside <think></think>.\n"
    "- To search, write <search>your query</search>. The results will be returned inside <information></information>.\n"
    "- You may search at most {max_searches} times. Questions often need several searches (for example, first find a bridge entity, then search for it).\n"
    "- When you know the answer, write it inside <answer></answer> as a short phrase (an entity, a date, a number, or yes/no), with no explanation.\n\n"
    "Question: {question}"
)

_SEARCH = re.compile(r"<search>(.*?)</search>", re.S)
_ANSWER = re.compile(r"<answer>(.*?)</answer>", re.S)


def build_prompt(question: str, max_searches: int = MAX_SEARCHES) -> str:
    return PROMPT_TEMPLATE.format(question=question.strip(), max_searches=max_searches)


# ---------------------------------------------------------------- 正規化と正解一致

def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(pred: str | None, golds) -> float:
    if pred is None:
        return 0.0
    if isinstance(golds, str):
        golds = [golds]
    p = normalize_answer(pred)
    return float(any(p == normalize_answer(g) for g in golds))


# ---------------------------------------------------------------- 1 回分の生成の解析

@dataclass(frozen=True)
class Action:
    kind: str  # "search" | "answer" | "invalid"
    text: str = ""


def parse_segment(seg: str) -> Action:
    """モデルが 1 回に生成した文字列から、最初の有効な行動を取り出す。

    <information> を自分で書いた場合(検索結果の捏造)は invalid。
    """
    if "<information>" in seg:
        return Action("invalid")
    ms, ma = _SEARCH.search(seg), _ANSWER.search(seg)
    cands = [(m.start(), k, m) for k, m in (("search", ms), ("answer", ma)) if m]
    if not cands:
        return Action("invalid")
    _, kind, m = min(cands, key=lambda c: c[0])
    text = m.group(1).strip()
    if not text:
        return Action("invalid")
    return Action(kind, text)


def format_information(docs: list[tuple[str, str]], max_words: int = 50) -> str:
    """検索結果 [(title, text)] を <information> ブロックにする。"""
    lines = []
    for i, (title, text) in enumerate(docs, 1):
        words = text.split()
        body = " ".join(words[:max_words]) + (" ..." if len(words) > max_words else "")
        lines.append(f"Doc {i} ({title}): {body}")
    return "\n<information>\n" + "\n".join(lines) + "\n</information>\n"


# ---------------------------------------------------------------- 軌跡

@dataclass
class Step:
    gen: str                       # モデルが生成した文字列(<think>... <search>q</search> など)
    action: Action
    retrieved: list[str] = field(default_factory=list)  # 検索したときの結果の title(順位順)


@dataclass
class Trajectory:
    steps: list[Step] = field(default_factory=list)

    @property
    def answer(self) -> str | None:
        for s in self.steps:
            if s.action.kind == "answer":
                return s.action.text
        return None

    @property
    def n_searches(self) -> int:
        return sum(1 for s in self.steps if s.action.kind == "search")

    def retrieved_union(self, upto: int | None = None) -> set[str]:
        """最初の upto 回(None なら全部)の検索で得た title の和集合(式 3)。"""
        out: set[str] = set()
        k = 0
        for s in self.steps:
            if s.action.kind != "search":
                continue
            if upto is not None and k >= upto:
                break
            out.update(s.retrieved)
            k += 1
        return out


# ---------------------------------------------------------------- 報酬

def outcome_reward(traj: Trajectory, golds) -> float:
    return exact_match(traj.answer, golds)


def grounded_reward(traj: Trajectory, gold_titles, upto: int | None = None) -> float:
    """式 (4): 検索で得た文書の和集合に含まれる、正解の根拠文書の割合。"""
    gold = set(gold_titles)
    if not gold:
        return 0.0
    return len(traj.retrieved_union(upto) & gold) / len(gold)


def total_reward(r_o: float, r_g: float, lam: float) -> float:
    """式 (5)。lam = 0 なら outcome のみ。"""
    return lam * r_g + (1.0 - lam) * r_o


def trajectory_reward(traj: Trajectory, golds, gold_titles, lam: float) -> float:
    return total_reward(outcome_reward(traj, golds), grounded_reward(traj, gold_titles), lam)


# ---------------------------------------------------------------- 利点

def group_advantages(rewards: list[float], eps: float = 1e-6) -> list[float]:
    """グループ内で標準化する(GRPO)。分散が 0 なら全部 0。"""
    n = len(rewards)
    if n == 0:
        return []
    mean = sum(rewards) / n
    var = sum((r - mean) ** 2 for r in rewards) / n
    std = var ** 0.5
    if std < eps:
        return [0.0] * n
    return [(r - mean) / (std + eps) for r in rewards]


def mixed_group_advantages(expert_reward: float, onpolicy_rewards: list[float], imitation_floor: float = 1.0):
    """warm-start: 専門家の軌跡 1 本と on-policy の G-1 本で 1 グループを作る(式 2)。

    専門家の利点は、グループ内の標準化の値と imitation_floor の大きい方にする(模倣を促すため)。
    返り値は (専門家の利点, [on-policy の利点])。
    """
    advs = group_advantages([expert_reward] + list(onpolicy_rewards))
    return max(advs[0], imitation_floor), advs[1:]


def clipped_surrogate(ratio: float, adv: float, eps: float = 0.2) -> float:
    """f_clip(ρ, A) = min(ρA, clip(ρ, 1-ε, 1+ε) A)。"""
    clipped = min(max(ratio, 1 - eps), 1 + eps)
    return min(ratio * adv, clipped * adv)


# ---------------------------------------------------------------- grounded expansion

def best_worst(rewards: list[float]) -> tuple[int, int]:
    """(最良の添字, 最悪の添字)。同点は先に出たものを最良、後に出たものを最悪とする。"""
    best = max(range(len(rewards)), key=lambda i: (rewards[i], -i))
    worst = min(range(len(rewards)), key=lambda i: (rewards[i], i))
    return best, worst


def truncation_point(traj: Trajectory, gold_titles) -> int | None:
    """式 (6): 最終の再現率に初めて達する検索の回数 t'(1 始まり)。

    証拠が 1 つも得られていなければ None(切り詰める点がない)。
    """
    final = grounded_reward(traj, gold_titles)
    if final <= 0:
        return None
    for t in range(1, traj.n_searches + 1):
        if grounded_reward(traj, gold_titles, upto=t) >= final - 1e-12:
            return t
    return None


def plan_expansion(rewards: list[float], trajs: list[Trajectory], gold_titles) -> tuple[int, int, int] | None:
    """grounded expansion の対象になるか判定する。

    最良の軌跡の報酬が 1 未満で、切り詰める点があり、最良と最悪が別の軌跡のとき
    (最良の添字, 最悪の添字, t') を返す。そうでなければ None。
    """
    if len(rewards) < 2:
        return None
    m_best, m_worst = best_worst(rewards)
    if rewards[m_best] >= 1.0 or m_best == m_worst or rewards[m_best] == rewards[m_worst]:
        return None
    t = truncation_point(trajs[m_best], gold_titles)
    if t is None:
        return None
    return m_best, m_worst, t


def accept_expansion(old_best_reward: float, candidate_rewards: list[float]) -> int | None:
    """l 本の継続の中で最高の報酬の添字を返す。元の最良を上回らなければ None(式 7 と置換の条件)。"""
    if not candidate_rewards:
        return None
    j = max(range(len(candidate_rewards)), key=lambda i: (candidate_rewards[i], -i))
    return j if candidate_rewards[j] > old_best_reward else None


# ---------------------------------------------------------------- 専門家の軌跡(warm-start)

def render_expert(steps: list[tuple[str, str]], final_think: str, answer: str, retrieve) -> tuple[list[str], Trajectory]:
    """人手で書いた (思考, 検索クエリ) の列と最終の思考・答えから、専門家の軌跡を作る。

    retrieve(query) は [(title, text)] を返す。返り値は、1 回ずつの生成文字列
    (information は含まない)のリストと、報酬計算用の Trajectory。
    """
    gens, traj = [], Trajectory()
    for think, query in steps:
        g = f"<think>{think}</think>\n<search>{query}</search>"
        docs = retrieve(query)
        gens.append(g)
        traj.steps.append(Step(g, Action("search", query), [t for t, _ in docs]))
    g = f"<think>{final_think}</think>\n<answer>{answer}</answer>"
    gens.append(g)
    traj.steps.append(Step(g, Action("answer", answer)))
    return gens, traj
