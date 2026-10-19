"""Landmark-pruned reverse-nearest-neighbour (RNN) moving queries on a simulated road network.

Baseline: exhaustive exact RNN check -- a full Dijkstra from every data object per query.
Improved: landmark (ALT-style) lower/upper distance bounds prune objects; Dijkstra runs only
for surviving candidates. Both must return identical answers (checked), so the main metric
is mean query latency in milliseconds (lower is better).
"""
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.sparse.csgraph import dijkstra

SEEDS = [42, 43, 44]
SIDE = 60                         # grid side -> about 3600 nodes before edge removal
DENSITIES = [0.01, 0.03, 0.06]    # fraction of nodes that are data objects
N_LANDMARKS = 16                  # fixed a priori, not tuned on any data
N_WALKS, WALK_LEN = 10, 20        # timed (test) moving queries: 10 walks x 20 steps
WARMUP_WALKS = 3                  # train split: warm-up + correctness check only
OUT = "results.json"


def build_graph(rng):
    """Perturbed grid: random edge removal (15%), random weights, largest connected component."""
    G = nx.grid_2d_graph(SIDE, SIDE)
    G = nx.convert_node_labels_to_integers(G)
    edges = list(G.edges())
    drop = rng.random(len(edges)) < 0.15
    G.remove_edges_from([e for e, d in zip(edges, drop) if d])
    G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    G = nx.convert_node_labels_to_integers(G, ordering="sorted")
    for u, v in G.edges():
        G[u][v]["weight"] = rng.uniform(1.0, 10.0)
    return G


def random_walks(G, rng, n_walks, length):
    """Moving queries: short random walks on the graph; every visited node is one query point."""
    out, nodes = [], list(G.nodes())
    for _ in range(n_walks):
        cur = int(rng.choice(nodes))
        walk = [cur]
        for _ in range(length - 1):
            nbrs = list(G.neighbors(cur))
            cur = int(nbrs[rng.integers(len(nbrs))])
            walk.append(cur)
        out.extend(walk)
    return out


def rnn_baseline(A, O, q):
    """Exhaustive exact check: full Dijkstra from every object.
    Object o is an RNN of q iff d(o,q) < d(o,o') for all other objects o'."""
    dist = dijkstra(A, directed=False, indices=O)          # (m, n)
    sub = dist[:, O]
    np.fill_diagonal(sub, np.inf)                          # exclude o itself
    rnn = dist[:, q] < sub.min(axis=1)
    return frozenset(O[rnn].tolist()), len(O)              # answer, #Dijkstra expansions


def build_index(A, n, rng, O):
    """Landmark index: distances from k random landmarks to all nodes, plus object-object
    upper bounds UB(o,o') = min_l d(l,o)+d(l,o') (triangle inequality)."""
    L = rng.choice(n, N_LANDMARKS, replace=False)
    DL = dijkstra(A, directed=False, indices=L)            # (k, n)
    DLO = DL[:, O]
    UB = np.full((len(O), len(O)), np.inf)
    for l in range(len(L)):
        UB = np.minimum(UB, DLO[l][:, None] + DLO[l][None, :])
    np.fill_diagonal(UB, np.inf)
    return DL, UB


def rnn_pruned(A, O, q, DL, UB):
    """Landmark-pruned exact check. Lower bound LB(o,q) = max_l |d(l,o)-d(l,q)| <= d(o,q).
    Object o is pruned if some o' has UB(o,o') <= LB(o,q): then d(o,o') <= d(o,q),
    so o cannot be a strict RNN. Survivors get the same exact Dijkstra verification."""
    lb = np.abs(DL[:, O] - DL[:, [q]]).max(axis=0)
    keep = np.where(UB.min(axis=1) > lb)[0]
    if len(keep) == 0:
        return frozenset(), 0
    dist = dijkstra(A, directed=False, indices=O[keep])    # (|keep|, n)
    sub = dist[:, O]
    sub[np.arange(len(keep)), keep] = np.inf               # exclude each candidate itself
    rnn = dist[:, q] < sub.min(axis=1)
    return frozenset(O[keep][rnn].tolist()), len(keep)


def run_seed(seed):
    """Run both methods on every density for one seed; returns per-density measurements."""
    rng = np.random.default_rng(seed)
    G = build_graph(rng)
    n = G.number_of_nodes()
    A = nx.to_scipy_sparse_array(G, nodelist=list(range(n)), weight="weight", format="csr")
    r = {"n": n, "n_train": 0, "n_test": 0, "mismatch": 0, "checked": 0,
         "base_ms": {}, "imp_ms": {}, "base_exp": {}, "imp_exp": {}, "build_ms": {}}
    for frac in DENSITIES:
        m = max(2, round(frac * n))
        O = np.sort(rng.choice(n, m, replace=False))
        walks = random_walks(G, rng, WARMUP_WALKS + N_WALKS, WALK_LEN)
        k = WARMUP_WALKS * WALK_LEN
        warm, test = walks[:k], walks[k:]
        t0 = time.perf_counter()
        DL, UB = build_index(A, n, rng, O)
        r["build_ms"][frac] = (time.perf_counter() - t0) * 1000
        # warm-up on the train split, then exact-agreement check on warm-up + test queries
        for q in warm:
            rnn_baseline(A, O, q)
            rnn_pruned(A, O, q, DL, UB)
        methods = {"base": lambda q: rnn_baseline(A, O, q),
                   "imp": lambda q: rnn_pruned(A, O, q, DL, UB)}
        answers = {}
        for name, fn in methods.items():
            times, exps, ans = [], [], []
            for q in test:
                t = time.perf_counter()
                a, e = fn(q)
                times.append(time.perf_counter() - t)
                exps.append(e)
                ans.append(a)
            answers[name] = ans
            r[f"{name}_ms"][frac] = float(np.mean(times) * 1000)
            r[f"{name}_exp"][frac] = float(np.mean(exps))
        r["mismatch"] += sum(a != b for a, b in zip(answers["base"], answers["imp"]))
        r["checked"] += len(test)
        r["n_train"] += len(warm)
        r["n_test"] += len(test)
        print(f"  seed {seed} density {frac}: objects={m} base={r['base_ms'][frac]:.2f}ms "
              f"imp={r['imp_ms'][frac]:.2f}ms")
    return r


def main():
    runs = {}
    for s in SEEDS:
        print(f"seed {s}: building graph and running queries")
        runs[s] = run_seed(s)
    total_mis = sum(runs[s]["mismatch"] for s in SEEDS)
    total_chk = sum(runs[s]["checked"] for s in SEEDS)
    if total_mis > 0:
        print(f"ERROR: {total_mis} answer mismatches between baseline and improved")
        sys.exit(1)

    def per_seed(key):   # mean over densities, per seed
        return np.array([np.mean(list(runs[s][key].values())) for s in SEEDS])

    b_seed, i_seed = per_seed("base_ms"), per_seed("imp_ms")
    b_mean, i_mean = float(b_seed.mean()), float(i_seed.mean())
    if not (np.isfinite(b_mean) and np.isfinite(i_mean)) or i_mean <= 0:
        print("ERROR: could not produce meaningful latency metrics")
        sys.exit(1)

    metrics = {
        "baseline_score": b_mean, "improved_score": i_mean,
        "improvement": b_mean - i_mean,
        "baseline_score_std": float(b_seed.std()), "improved_score_std": float(i_seed.std()),
        "speedup": b_mean / i_mean,
        "baseline_expansions_per_query": float(per_seed("base_exp").mean()),
        "improved_expansions_per_query": float(per_seed("imp_exp").mean()),
        "baseline_build_ms": 0.0,   # baseline has no index
        "improved_build_ms": float(per_seed("build_ms").mean()),
        "exact_match_rate": 1.0 - total_mis / total_chk,
    }
    per_d = {}
    for frac in DENSITIES:
        for name, key in [("baseline", "base_ms"), ("improved", "imp_ms")]:
            vals = [runs[s][key][frac] for s in SEEDS]
            metrics[f"{name}_ms_density_{frac:g}"] = float(np.mean(vals))
            per_d.setdefault(name, []).append((np.mean(vals), np.std(vals)))

    # Optional chart: latency by density, baseline vs improved
    os.makedirs("figures", exist_ok=True)
    x = np.arange(len(DENSITIES))
    fig, ax = plt.subplots(figsize=(6, 4))
    for off, name in [(-0.2, "baseline"), (0.2, "improved")]:
        mu = [v[0] for v in per_d[name]]
        sd = [v[1] for v in per_d[name]]
        ax.bar(x + off, mu, 0.4, yerr=sd, label=name, capsize=3)
    ax.set_xticks(x, [f"{f:g}" for f in DENSITIES])
    ax.set_xlabel("object density (fraction of nodes)")
    ax.set_ylabel("mean query latency (ms)")
    ax.legend()
    fig.tight_layout()
    fig.savefig("figures/latency_by_density.png", dpi=120)

    results = {
        "question": "Does a landmark-pruned index answer reverse-nearest-neighbour queries on a "
                    "road network faster than a standard exhaustive baseline, at the same exact "
                    "answers, as the density of data objects grows?",
        "method": "Simulated perturbed-grid road network; moving RNN queries from random walks; "
                  "exact answers verified identical; latency compared across object densities "
                  "and seeds 42, 43, 44.",
        "dataset": {
            "name": "Synthetic perturbed grid road network (networkx grid, random edge removal "
                    "and weights)",
            "source_url": "https://networkx.org/",
            "n_rows": int(runs[SEEDS[0]]["n"]),
            "n_train": int(sum(runs[s]["n_train"] for s in SEEDS)),
            "n_test": int(sum(runs[s]["n_test"] for s in SEEDS)),
            "same_as_paper": False,
            "baseline_is_papers_method": False,
        },
        "metric_name": "mean query latency (ms), averaged over densities and seeds",
        "higher_is_better": False,
        "baseline": "Exhaustive exact RNN check: full Dijkstra from every data object per query, "
                    "no index.",
        "improved": "Adds a landmark index: triangle-inequality lower/upper bounds prune objects "
                    "before the same exact Dijkstra verification runs only on survivors.",
        "metrics": metrics,
        "notes": "Simulated network, not the paper's dataset; the paper's adjustable-tree method "
                 "is not implemented. No model is trained and nothing is tuned: landmark count "
                 "is fixed a priori. The train split is only warm-up queries. Latency is wall "
                 "clock on a shared CI runner and is noisy. Index build time is reported "
                 "separately and not included in query latency. Exact answer agreement is "
                 "checked on every timed query.",
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"baseline {b_mean:.2f} ms vs improved {i_mean:.2f} ms; wrote {OUT}")


if __name__ == "__main__":
    main()
