"""metrics.py の動作確認: 共通の潜在変数から作った 2 つの表現は高く、無関係なら偶然の水準になる。"""
import numpy as np

from metrics import evaluate_alignment, mutual_knn


def make(rng, n, shared):
    z = rng.normal(size=(n, 20))
    A = rng.normal(size=(20, 32))
    B = rng.normal(size=(20, 48))
    z2 = z if shared else rng.normal(size=(n, 20))
    return z @ A + 0.1 * rng.normal(size=(n, 32)), z2 @ B + 0.1 * rng.normal(size=(n, 48))


def split(rng, shared):
    t, i = make(rng, 3000 + 400 + 400, shared)
    s = [slice(0, 3000), slice(3000, 3400), slice(3400, 3800)]
    return [x for sl in s for x in (t[sl], i[sl])]


rng = np.random.default_rng(0)
good = evaluate_alignment(*split(rng, True))
bad = evaluate_alignment(*split(rng, False))
print("共通の潜在変数:", {k: round(v, 3) for k, v in good.items()})
print("無関係        :", {k: round(v, 3) for k, v in bad.items()})
assert good["R@1"] > 0.5 and good["mutual_knn"] > 0.3
assert bad["R@1"] < 0.02 and bad["mutual_knn"] < 0.05
X = rng.normal(size=(500, 16))
assert abs(mutual_knn(X, X) - 1.0) < 1e-9  # 同一表現なら 1
print("OK")
