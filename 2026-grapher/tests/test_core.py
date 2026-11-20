import numpy as np
import pytest

import grapher_core as gc


def fs(*x):
    return frozenset(x)


def test_adjacency_formula_asymmetric():
    # E_0={a,b}, E_1={a,b,c,d}: A_01 = |共有|/|E_1| = 2/4、A_10 = 2/|E_0| = 1
    A = gc.adjacency_from_entities([fs("a", "b"), fs("a", "b", "c", "d")])
    assert A[0, 1] == pytest.approx(0.5)
    assert A[1, 0] == pytest.approx(1.0)


def test_adjacency_diagonal_zero_and_no_share():
    A = gc.adjacency_from_entities([fs("a"), fs("b"), fs("a", "c")])
    assert np.all(np.diag(A) == 0)
    assert A[0, 1] == 0 and A[1, 0] == 0
    assert A[0, 2] == pytest.approx(0.5) and A[2, 0] == pytest.approx(1.0)


def test_adjacency_empty_entity_sets():
    A = gc.adjacency_from_entities([fs(), fs(), fs("a")])
    assert A.sum() == 0
    A2 = gc.adjacency_from_entities([fs(), fs()])
    assert A2.shape == (2, 2) and A2.sum() == 0


def test_adjacency_nonnegative_and_at_most_one():
    rng = np.random.default_rng(0)
    ents = [frozenset(rng.integers(0, 10, size=rng.integers(0, 6)).tolist()) for _ in range(30)]
    A = gc.adjacency_from_entities(ents)
    assert (A >= 0).all() and (A <= 1 + 1e-12).all()


def test_adjacency_from_pairs_symmetric():
    A = gc.adjacency_from_pairs(4, [(0, 1), (1, 2), (2, 2)])
    assert (A == A.T).all() and A[2, 2] == 0 and A[0, 1] == 1 and A[0, 2] == 0


def test_row_normalize_rows_sum_to_one_or_zero():
    A = np.array([[0, 2, 2], [1, 0, 0], [0, 0, 0.0]])
    W = gc.row_normalize(A)
    assert W[0].sum() == pytest.approx(1) and W[1].sum() == pytest.approx(1)
    assert W[2].sum() == 0
    assert W[0, 1] == pytest.approx(0.5)


def test_gcs_equals_seed_without_edges():
    s = np.array([0.9, 0.5, 0.1])
    assert np.allclose(gc.gcs(np.zeros((3, 3)), s, 0.5), s)


def test_gcs_never_below_seed_random_graphs():
    rng = np.random.default_rng(1)
    for _ in range(50):
        n = 20
        A = (rng.random((n, n)) < 0.2) * rng.random((n, n))
        s = rng.random(n)
        for a in (0.1, 0.5, 0.9):
            assert (gc.gcs(A, s, a) >= s - 1e-12).all()


def test_gcs_alpha_one_is_seed():
    rng = np.random.default_rng(2)
    A = rng.random((8, 8)); s = rng.random(8)
    assert np.allclose(gc.gcs(A, s, 1.0), s)


def test_gcs_raises_low_neighbour_of_high_node():
    # 0 と 1 は隣接。0 の点数が高いので 1 が持ち上がり、0 は変わらない
    A = np.array([[0, 1.0], [1.0, 0]])
    out = gc.gcs(A, np.array([0.9, 0.1]), 0.5)
    assert out[0] == pytest.approx(0.9)
    assert out[1] > 0.1


def test_gcs_fixed_point():
    rng = np.random.default_rng(3)
    A = (rng.random((10, 10)) < 0.4) * 1.0
    s = rng.random(10)
    out = gc.gcs(A, s, 0.3, tol=1e-12)
    W = gc.row_normalize(A)
    # max 前の p は p = alpha s + (1-alpha) W p を満たす(線形方程式の解と比べる)
    p_star = np.linalg.solve(np.eye(10) - 0.7 * W, 0.3 * s)
    assert np.allclose(np.maximum(p_star, s), out, atol=1e-8)


def test_gcs_scale_equivariance():
    rng = np.random.default_rng(4)
    A = rng.random((6, 6)); s = rng.random(6)
    assert np.allclose(gc.gcs(A, 3 * s, 0.4), 3 * gc.gcs(A, s, 0.4))


def test_gcs_ignores_edge_scale():
    rng = np.random.default_rng(5)
    A = rng.random((6, 6)); s = rng.random(6)
    assert np.allclose(gc.gcs(A, s, 0.4), gc.gcs(5 * A, s, 0.4))


def test_ppr_is_distribution():
    rng = np.random.default_rng(6)
    A = (rng.random((12, 12)) < 0.3) * 1.0
    p = gc.ppr(A, rng.random(12), 0.2)
    assert p.sum() == pytest.approx(1.0, abs=1e-8) and (p >= 0).all()


def test_ppr_no_edges_returns_seed_distribution():
    s = np.array([2.0, 1.0, 1.0])
    assert np.allclose(gc.ppr(np.zeros((3, 3)), s, 0.3), s / s.sum())


def star(n_leaf):
    A = np.zeros((n_leaf + 1, n_leaf + 1))
    A[0, 1:] = A[1:, 0] = 1
    return A


def test_ppr_hub_bias_on_star():
    # ハブ(0)の seed はリーフと同じ低さ。PPR ではハブが 1 位になり、GCS(max(p, s))ではならない
    A = star(8)
    s = np.full(9, 0.1)
    s[1] = 0.12  # 本当に関連があるのは 1 だけ
    p = gc.ppr(A, s, 0.15)
    assert int(np.argmax(p)) == 0
    g = gc.gcs(A, s, 0.15)
    assert int(np.argmax(g)) != 0 or g[0] <= g[1] + 0.05


def test_gcs_hub_not_inflated_beyond_neighbour_mean():
    A = star(8)
    s = np.full(9, 0.1)
    g = gc.gcs(A, s, 0.15)
    assert g[0] == pytest.approx(0.1)   # 近傍も同じ点数なので持ち上がらない


def test_ppr_vs_gcs_rank_of_relevant_leaf():
    A = star(10)
    s = np.full(11, 0.05); s[3] = 0.5
    p = gc.ppr(A, s, 0.15)
    assert p[0] > p[3]                  # ハブが本当に関連のあるリーフを上回る
    g = gc.gcs(A, s, 0.15)
    assert g[3] > g[0]


def test_hybrid_scores_weights_and_zero_bm25():
    cos = np.array([1.0, 0.0]); bm = np.array([2.0, 4.0])
    assert np.allclose(gc.hybrid_scores(cos, bm), [0.7 + 0.15, 0.3])
    assert np.allclose(gc.hybrid_scores(cos, np.zeros(2)), [0.7, 0.0])


def test_top_candidates_order_and_cap():
    sc = np.array([0.1, 0.9, 0.5, 0.7])
    assert gc.top_candidates(sc, 3).tolist() == [1, 3, 2]
    assert len(gc.top_candidates(sc, 200)) == 4


def test_rank_from_scores_tie_keeps_seed_order():
    cand = np.array([10, 20, 30])
    assert gc.rank_from_scores(cand, np.array([0.5, 0.5, 0.9])) == [30, 10, 20]


def test_bm25_ranks_matching_doc_first():
    b = gc.BM25(["the cat sat on a mat", "dogs bark loudly", "cat and dog play"])
    s = b.scores("cat mat")
    assert int(np.argmax(s)) == 0
    assert gc.BM25(["x y"]).scores("zzz").sum() == 0


def test_degrees():
    assert gc.degrees(star(4)).tolist() == [4, 1, 1, 1, 1]
