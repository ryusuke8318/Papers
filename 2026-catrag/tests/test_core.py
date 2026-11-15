import math

import numpy as np
import scipy.sparse as sp

import catrag_core as cc


def small_graph():
    # 0-1-2-3 の鎖(重み 1, 2, 1)と 1-4
    edges = [(0, 1, 1.0), (1, 2, 2.0), (2, 3, 1.0), (1, 4, 1.0)]
    r, c, v = [], [], []
    for a, b, w in edges:
        r += [a, b]; c += [b, a]; v += [w, w]
    return sp.csr_matrix((v, (r, c)), shape=(5, 5))


def test_ppr_matches_analytic_solution():
    W = small_graph()
    seed = np.array([1.0, 0, 0, 0, 0])
    for d in (0.3, 0.5, 0.85):
        np.testing.assert_allclose(cc.ppr(W, seed, d=d), cc.ppr_exact(W, seed, d=d), atol=1e-8)


def test_ppr_is_distribution_and_seed_has_most_mass_with_small_d():
    W = small_graph()
    v = cc.ppr(W, np.array([0, 1.0, 0, 0, 0]), d=0.5)
    assert abs(v.sum() - 1) < 1e-8
    assert v.argmax() == 1


def test_ppr_closed_form_two_nodes():
    W = sp.csr_matrix(np.array([[0, 1.0], [1.0, 0]]))
    d = 0.5
    v = cc.ppr(W, np.array([1.0, 0]), d=d)
    # v0 = (1-d) + d v1, v1 = d v0  ->  v0 = (1-d)/(1-d^2)
    assert math.isclose(v[0], (1 - d) / (1 - d * d), rel_tol=1e-8)


def test_extract_entities():
    ents = cc.extract_entities("The Lord of the Rings was written by J. R. R. Tolkien in 1954. He was born in Bloemfontein.")
    assert "lord of the rings" in ents
    assert "bloemfontein" in ents
    assert "1954" in ents
    assert "he" not in ents and "the" not in ents


def test_normalize_key_drops_parenthesis():
    assert cc.normalize_key("Ed Wood (film)") == "ed wood"
    assert cc.normalize_key("Curie's") == "curie"


def _mini_corpus():
    return [
        {"title": "Marie Curie", "sents": ["Marie Curie studied under Gabriel Lippmann.", "She won the Nobel Prize."]},
        {"title": "Gabriel Lippmann", "sents": ["Gabriel Lippmann attended the Ecole Normale Superieure.", "He won the Nobel Prize."]},
        {"title": "Nobel Prize", "sents": ["The Nobel Prize is awarded each year.", "Nobel Prize winners include Marie Curie and Gabriel Lippmann."]},
        {"title": "Radioactivity", "sents": ["Radioactivity was studied by Marie Curie."]},
    ]


def test_graph_structure():
    idx = cc.build_graph(_mini_corpus())
    assert idx.n_p == 4
    # 無向の対称な重み
    assert abs(idx.W - idx.W.T).sum() == 0
    mc = idx.ent_id["marie curie"]
    gl = idx.ent_id["gabriel lippmann"]
    # 2 つの記事名は、文内で共起している
    assert (min(mc, gl), max(mc, gl)) in idx.pair_sents
    # 語句と、それを含む段落の間に文脈辺がある
    assert idx.W[mc, idx.n_e + 0] == 1.0
    assert idx.W[mc, idx.n_e + 3] == 1.0
    assert idx.ent_npass[mc] == 3


def _sims(idx, hot_sents):
    s = np.full(len(idx.sent_pass), 0.2)
    for i, v in hot_sents.items():
        s[i] = v
    return s


def test_static_config_returns_base_matrix():
    idx = cc.build_graph(_mini_corpus())
    sim = _sims(idx, {0: 0.9})
    cfg = cc.CatConfig()
    ent_w = cc.seed_entities(idx, sim, cfg)
    assert cc.modify_graph(idx, ent_w, cc.ranks_from_sim(sim), cfg) is idx.W


def test_edge_weighting_prunes_and_amplifies_only_seed_rows():
    idx = cc.build_graph(_mini_corpus())
    sim = _sims(idx, {0: 0.9})           # 文 0: "Marie Curie studied under Gabriel Lippmann."
    rank = cc.ranks_from_sim(sim)
    cfg = cc.CatConfig(use_edge=True, k_fact=1, n_seed=1, tier_ranks=(1, 1, 1), k_edge=100)
    ent_w = cc.seed_entities(idx, sim, cfg)
    Wq = cc.modify_graph(idx, ent_w, rank, cfg)
    mc, gl, nb = idx.ent_id["marie curie"], idx.ent_id["gabriel lippmann"], idx.ent_id["nobel prize"]
    # 起点の語句 (marie curie か gabriel lippmann) から出る、質問に近い文で共起する辺は 5 倍(Direct)
    seeds = [e for e in idx.sent_ents[0]]
    u = max(seeds, key=lambda e: ent_w[e])
    other = gl if u == mc else mc
    assert Wq[u, other] == idx.W[u, other] * 5.0
    # 根拠の文が文 0 でない辺(順位が 1 を超える)は 0 に刈られる
    assert 0 not in idx.pair_sents[(min(u, nb), max(u, nb))]
    assert Wq[u, nb] == 0.0
    # 起点でない語句の行は変わらない
    unrelated = idx.ent_id["radioactivity"]
    np.testing.assert_array_equal(Wq[unrelated].toarray(), idx.W[unrelated].toarray())
    # 逆向きの辺(非対称)は変わらない
    assert Wq[other, u] == idx.W[other, u]


def test_stage_one_prunes_to_k_edge():
    idx = cc.build_graph(_mini_corpus())
    sim = _sims(idx, {0: 0.9, 1: 0.8, 2: 0.7})
    rank = cc.ranks_from_sim(sim)
    base = cc.CatConfig(use_edge=True, k_fact=1, n_seed=1, tier_ranks=(1000, 2000, 3000))
    ent_w = cc.seed_entities(idx, sim, base)
    u = int(np.argmax(ent_w))
    W_all = cc.modify_graph(idx, ent_w, rank, base)
    W_k1 = cc.modify_graph(idx, ent_w, rank, cc.CatConfig(**{**base.__dict__, "k_edge": 1}))
    rel_cols = [c for c in idx.W[u].indices if c < idx.n_e]
    scaled_all = sum(W_all[u, c] > 0.5 * idx.W[u, c] * base.weak_min + 1e-12 and abs(W_all[u, c] / idx.W[u, c] - base.weak_min) > 1e-9 for c in rel_cols)
    scaled_k1 = sum(abs(W_k1[u, c] / idx.W[u, c] - base.weak_min) > 1e-9 for c in rel_cols)
    assert scaled_k1 <= 1 < len(rel_cols)
    assert scaled_all >= scaled_k1


def test_key_fact_boost_multiplies_supported_edges_only():
    idx = cc.build_graph(_mini_corpus())
    sim = _sims(idx, {2: 0.9})           # 文 2: "Gabriel Lippmann attended the Ecole Normale Superieure."(段落 1)
    rank = cc.ranks_from_sim(sim)
    cfg = cc.CatConfig(use_key=True, k_fact=1, n_seed=3, beta=2.5, r_key=1)
    ent_w = cc.seed_entities(idx, sim, cfg)
    Wq = cc.modify_graph(idx, ent_w, rank, cfg)
    gl = idx.ent_id["gabriel lippmann"]
    assert math.isclose(Wq[gl, idx.n_e + 1], idx.W[gl, idx.n_e + 1] * 3.5)   # 支持される段落
    assert Wq[gl, idx.n_e + 0] == idx.W[gl, idx.n_e + 0]                     # 支持されない段落
    # 語句どうしの辺には効かない
    assert Wq[gl, idx.ent_id["nobel prize"]] == idx.W[gl, idx.ent_id["nobel prize"]]


def test_tier_weight_is_monotone_and_matches_table():
    ranks = (10, 100, 1000)
    assert cc.tier_weight(1, ranks) == 5.0
    assert cc.tier_weight(10, ranks) == 5.0
    assert math.isclose(cc.tier_weight(11, ranks), 3.0, abs_tol=0.05)
    assert math.isclose(cc.tier_weight(100, ranks), 2.0)
    assert math.isclose(cc.tier_weight(101, ranks), 0.3, abs_tol=0.01)
    assert math.isclose(cc.tier_weight(1000, ranks), 0.2)
    assert cc.tier_weight(1001, ranks) == 0.0
    vals = [cc.tier_weight(r, ranks) for r in range(1, 1200)]
    assert all(a >= b - 1e-12 for a, b in zip(vals, vals[1:]))


def test_symbolic_anchor_adds_weak_mass_to_anchor_only():
    idx = cc.build_graph(_mini_corpus())
    sim = _sims(idx, {0: 0.9})
    dense = np.array([0.5, 0.4, 0.3, 0.2])
    anchor = idx.ent_id["radioactivity"]
    s0, _ = cc.build_seed(idx, sim, dense, [anchor], cc.CatConfig(use_anchor=False, k_fact=1))
    s1, _ = cc.build_seed(idx, sim, dense, [anchor], cc.CatConfig(use_anchor=True, eps=0.2, k_fact=1))
    assert s0[anchor] == 0.0
    assert math.isclose(s1[anchor], 0.2, rel_tol=1e-9)
    others = np.arange(idx.n_nodes) != anchor
    np.testing.assert_allclose(s1[others], s0[others])


def test_retrieve_runs_and_prefers_seed_passage():
    idx = cc.build_graph(_mini_corpus())
    sim = _sims(idx, {0: 0.9})
    dense = np.array([0.9, 0.5, 0.3, 0.2])
    cfg = cc.CatConfig(use_anchor=True, use_edge=True, use_key=True)
    order, v = cc.retrieve(idx, sim, dense, idx.match_query_entities("Marie Curie"), cfg)
    assert order[0] == 0
    assert abs(v.sum() - 1) < 1e-6


# ---- 活性化の拡散

def chain_adj(w):
    return {0: [(1, w)], 1: [(2, w)], 2: [(3, w)]}


def test_rescale_formula():
    assert math.isclose(cc.rescale_weight(0.7, 0.4), 0.5)
    assert cc.rescale_weight(0.4, 0.4) == 0.0
    assert cc.rescale_weight(0.3, 0.4) == 0.0
    assert cc.rescale_weight(1.0, 0.4) == 1.0


def test_spreading_activation_chain_values():
    act = cc.spreading_activation(chain_adj(0.5), {0: 1.0}, c=0.0, max_hops=3)
    assert math.isclose(act[1], 0.5) and math.isclose(act[2], 0.25) and math.isclose(act[3], 0.125)
    act = cc.spreading_activation(chain_adj(0.7), {0: 1.0}, c=0.4, max_hops=2)
    assert math.isclose(act[1], 0.5) and math.isclose(act[2], 0.25) and 3 not in act


def test_rescale_suppresses_over_spreading():
    # 全結合に近い 30 節点。重み 0.5〜0.9 の辺が多いとき、c が大きいほど閾値を超える節点が減る
    rng = np.random.default_rng(0)
    n = 30
    adj = {i: [(j, float(rng.uniform(0.5, 0.9))) for j in range(n) if j != i and rng.random() < 0.3] for i in range(n)}
    counts = []
    for c in (0.0, 0.3, 0.5, 0.7):
        act = cc.spreading_activation(adj, {0: 1.0}, c=c, max_hops=3)
        counts.append(sum(1 for k, a in act.items() if k != 0 and a > 0.2))
    assert counts == sorted(counts, reverse=True)
    assert counts[0] > counts[-1]


def test_activation_is_capped():
    adj = {0: [(2, 1.0)], 1: [(2, 1.0)]}
    act = cc.spreading_activation(adj, {0: 1.0, 1: 1.0}, c=0.0, max_hops=1)
    assert act[2] == 1.0


def test_hub_stats_monotone_with_mass():
    idx = cc.build_graph(_mini_corpus())
    deg = idx.weighted_degree()[: idx.n_e]
    hub = int(np.argmax(deg))
    low = int(np.argmin(deg))
    v_hub = np.zeros(idx.n_nodes); v_hub[hub] = 1.0; v_hub[low] = 0.1
    v_low = np.zeros(idx.n_nodes); v_low[hub] = 0.1; v_low[low] = 1.0
    a, b = cc.hub_stats(idx, v_hub, top_n=2), cc.hub_stats(idx, v_low, top_n=2)
    assert a["s_ppr"] > b["s_ppr"]
    assert a["hub_mass_all"] > b["hub_mass_all"]


def test_bm25_ranks_matching_doc_first():
    bm = cc.BM25(["the cat sat", "dogs bark loudly", "a cat and a dog"])
    s = bm.scores("cat")
    assert s[0] > s[1] and s[2] > s[1]
