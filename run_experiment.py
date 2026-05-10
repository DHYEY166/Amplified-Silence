#!/usr/bin/env python3
"""End-to-end experiment: Last.fm 360K slice -> ALS -> fairness + mitigation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent  # post_processing/
REPO_ROOT = ROOT.parent  # repository root (contains fairness_reg_als/, .hf_cache/, …)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from implicit.als import AlternatingLeastSquares  # type: ignore

from src.data import build_bundle, load_raw_slice
from src.evaluate import FairnessAccumulator, _precision_recall_ndcg, train_exposure_share
from src.labels import load_or_fetch_labels


def _item_label_map(orig_mbids: np.ndarray, mbid_to_label: dict[str, str]) -> dict[int, str]:
    return {int(idx): mbid_to_label.get(str(mb), "unknown") for idx, mb in enumerate(orig_mbids)}


def _user_relevant_map(test_df) -> dict[int, set[int]]:
    rel: dict[int, set[int]] = {}
    for row in test_df.itertuples(index=False):
        rel.setdefault(int(row.u), set()).add(int(row.i))
    return rel


def apply_boost_and_rank(
    ids: np.ndarray,
    raw_scores: np.ndarray,
    k: int,
    item_boost: np.ndarray | None,
) -> list[int]:
    """Apply per-item score multipliers to a precomputed candidate pool and return top-k."""
    scores = raw_scores.copy()
    if item_boost is not None:
        scores *= item_boost[ids]
    order = np.argsort(-scores)
    return [int(ids[i]) for i in order[:k]]


def constrained_rerank(
    ids: np.ndarray,
    scores: np.ndarray,
    item_labels: dict[int, str],
    k: int,
    min_female: int,
) -> list[int]:
    """Pull high-scoring female artists into the head of the list, then fill by score."""
    order = np.argsort(-scores)
    ranked = [int(ids[i]) for i in order]
    females = [it for it in ranked if item_labels.get(it) == "female"]
    target_f = min(min_female, k, len(females))
    out: list[int] = []
    out.extend(females[:target_f])
    for it in ranked:
        if len(out) >= k:
            break
        if it in out:
            continue
        out.append(it)
    return out[:k]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="train[:1200000]", help="HF datasets slice")
    p.add_argument("--streaming", action="store_true", help="Stream the HF split (avoids full-shard download)")
    p.add_argument("--min-user", type=int, default=25)
    p.add_argument("--min-artist", type=int, default=15)
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--pool", type=int, default=400, help="Candidate pool size before re-ranking")
    p.add_argument("--factors", type=int, default=64)
    p.add_argument("--als-iters", type=int, default=20)
    p.add_argument("--reg", type=float, default=0.08)
    p.add_argument("--max-mb-fetch", type=int, default=2500, help="Cap MusicBrainz requests (incremental cache)")
    p.add_argument("--skip-mb-fetch", action="store_true", help="Use existing cache only; no network calls")
    p.add_argument("--boost-a", type=float, default=0.12, help="Multiplicative boost for female scores (mitigation A)")
    p.add_argument("--boost-b", type=float, default=0.28, help="Stronger boost (mitigation B)")
    p.add_argument("--min-female-topk", type=int, default=4, help="Mitigation C: min females in top-K (if available)")
    p.add_argument("--out-dir", default=str(ROOT / "data" / "outputs"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = ROOT / "data" / "cache" / "musicbrainz_artist_labels.json"

    print("Loading interactions:", args.split)
    raw = load_raw_slice(args.split, streaming=args.streaming)
    bundle = build_bundle(
        raw,
        min_user_interactions=args.min_user,
        min_artist_interactions=args.min_artist,
        test_frac=args.test_frac,
    )
    print(
        f"Filtered: users={bundle.n_users:,} items={bundle.n_items:,} "
        f"train_edges={bundle.train_df.shape[0]:,} test_edges={bundle.test_df.shape[0]:,}"
    )

    # Prioritize MusicBrainz fetches by play mass so the most-heard artists get labels first
    play_mass = np.zeros(bundle.n_items, dtype=np.float64)
    for part in (bundle.train_df, bundle.test_df):
        g = part.groupby("i", as_index=False)["play_count"].sum()
        for row in g.itertuples(index=False):
            play_mass[int(row.i)] += float(row.play_count)
    order = np.argsort(-play_mass)
    seen: set[str] = set()
    mbids_by_plays: list[str] = []
    for idx in order:
        mb = str(bundle.orig_artist_mbids[int(idx)])
        if not mb or mb in seen:
            continue
        seen.add(mb)
        mbids_by_plays.append(mb)
    print(f"Unique MusicBrainz artist ids in filtered slice: {len(mbids_by_plays):,}")

    fetch_limit = 0 if args.skip_mb_fetch else args.max_mb_fetch
    mbid_to_label = load_or_fetch_labels(mbids_by_plays, cache_path, max_fetch=fetch_limit)
    item_labels = _item_label_map(bundle.orig_artist_mbids, mbid_to_label)

    labeled_items = sum(1 for i in range(bundle.n_items) if item_labels.get(i) in ("male", "female"))
    print(f"Items with male/female label: {labeled_items:,}")

    train_share = train_exposure_share(bundle.train_matrix, item_labels)

    model = AlternatingLeastSquares(
        factors=args.factors,
        regularization=args.reg,
        iterations=args.als_iters,
        random_state=42,
    )
    print("Fitting ALS…")
    model.fit(bundle.train_matrix)

    rel = _user_relevant_map(bundle.test_df)
    test_users = sorted(rel.keys())
    K = args.k

    # Precompute candidate pools ONCE for all test users.
    # All scenario/sensitivity loops reuse these — eliminates redundant model.recommend() calls.
    print("Precomputing candidate pools…")
    user_candidates: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for u in tqdm(test_users, desc="precompute"):
        if not rel[u]:
            continue
        ids, scores = model.recommend(
            u, bundle.train_matrix[u], N=args.pool, filter_already_liked_items=True
        )
        user_candidates[u] = (
            np.asarray(ids, dtype=np.int32),
            np.asarray(scores, dtype=np.float64),
        )
    n_eval = len(user_candidates)

    boost_vec_a = np.ones(bundle.n_items, dtype=np.float64)
    boost_vec_b = np.ones(bundle.n_items, dtype=np.float64)
    for it, lab in item_labels.items():
        if lab == "female":
            boost_vec_a[it] = 1.0 + args.boost_a
            boost_vec_b[it] = 1.0 + args.boost_b

    scenarios: dict[str, np.ndarray | None] = {
        "baseline": None,
        "mitigation_a": boost_vec_a,
        "mitigation_b": boost_vec_b,
    }

    results: dict[str, dict] = {}

    for name, boost in scenarios.items():
        prec_sum = rec_sum = ndcg_sum = 0.0
        fair = FairnessAccumulator()
        for u, (ids, raw_scores) in tqdm(user_candidates.items(), desc=name, total=n_eval):
            rec = apply_boost_and_rank(ids, raw_scores, K, boost)
            p, r, n = _precision_recall_ndcg(rec, rel[u], K)
            prec_sum += p
            rec_sum += r
            ndcg_sum += n
            fair.add_recommendation(rec, item_labels)
        results[name] = {
            "precision_at_k": prec_sum / n_eval,
            "recall_at_k": rec_sum / n_eval,
            "ndcg_at_k": ndcg_sum / n_eval,
            "exposure_male_share": fair.exposure_gap(),
            "avg_rank_gap_m_minus_f": fair.avg_rank_gap(),
            "coverage_gap_m_minus_f": fair.coverage_gap(),
            "users_evaluated": n_eval,
        }

    # Mitigation C: constrained minimum female artists in top-K
    fair_c = FairnessAccumulator()
    prec_sum = rec_sum = ndcg_sum = 0.0
    min_f = min(args.min_female_topk, K)
    for u, (ids, raw_scores) in tqdm(user_candidates.items(), desc="mitigation_c", total=n_eval):
        rec = constrained_rerank(ids, raw_scores, item_labels, K, min_female=min_f)
        p, r, n = _precision_recall_ndcg(rec, rel[u], K)
        prec_sum += p
        rec_sum += r
        ndcg_sum += n
        fair_c.add_recommendation(rec, item_labels)
    results["mitigation_c_constrained"] = {
        "precision_at_k": prec_sum / n_eval,
        "recall_at_k": rec_sum / n_eval,
        "ndcg_at_k": ndcg_sum / n_eval,
        "exposure_male_share": fair_c.exposure_gap(),
        "avg_rank_gap_m_minus_f": fair_c.avg_rank_gap(),
        "coverage_gap_m_minus_f": fair_c.coverage_gap(),
        "users_evaluated": n_eval,
    }

    # Sensitivity: vary female boost over a range; reuse precomputed pools (no extra model calls)
    sens_x: list[float] = []
    sens_fair: list[float | None] = []
    sens_ndcg: list[float] = []
    for b in [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.45]:
        boost_vec = np.ones(bundle.n_items, dtype=np.float64)
        for it, lab in item_labels.items():
            if lab == "female":
                boost_vec[it] = 1.0 + b
        fair_s = FairnessAccumulator()
        nd_sum = 0.0
        for u, (ids, raw_scores) in user_candidates.items():
            rec = apply_boost_and_rank(ids, raw_scores, K, boost_vec)
            _, _, n_s = _precision_recall_ndcg(rec, rel[u], K)
            fair_s.add_recommendation(rec, item_labels)
            nd_sum += n_s
        sens_x.append(b)
        sens_fair.append(fair_s.exposure_gap())
        sens_ndcg.append(nd_sum / n_eval)

    summary = {
        "config": vars(args),
        "dataset": {
            "users": bundle.n_users,
            "items": bundle.n_items,
            "train_interactions": int(bundle.train_df.shape[0]),
            "test_interactions": int(bundle.test_df.shape[0]),
            "labeled_mf_items": labeled_items,
            "train_male_share_among_labeled": train_share["male_share"],
        },
        "metrics": results,
        "sensitivity": {"boost": sens_x, "exposure_male_share": sens_fair, "ndcg_at_k": sens_ndcg},
    }
    (out_dir / "results.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(results, indent=2))
    print("Wrote", out_dir / "results.json")

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        male_share = [float(x) if x is not None else np.nan for x in sens_fair]
        ax.plot(sens_x, male_share, marker="o", label="Male exposure share ↓ is fairer")
        ax.set_xlabel("Female score boost (additive factor on 1 + boost)")
        ax.set_ylabel("Male share of weighted exposure (top-K recs)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "tradeoff_exposure_vs_boost.png", dpi=160)

        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.plot(sens_x, sens_ndcg, marker="o", color="darkorange")
        ax2.set_xlabel("Female score boost")
        ax2.set_ylabel(f"NDCG@{K}")
        ax2.grid(True, alpha=0.3)
        fig2.tight_layout()
        fig2.savefig(out_dir / "tradeoff_ndcg_vs_boost.png", dpi=160)
        print("Saved trade-off plots to", out_dir)
    except Exception as exc:  # noqa: BLE001
        print("Plotting skipped:", exc)


if __name__ == "__main__":
    os.environ.setdefault("HF_HOME", str(REPO_ROOT / ".hf_cache"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(REPO_ROOT / ".hf_cache"))
    main()
