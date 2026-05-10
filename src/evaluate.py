"""Ranking utility + provider-side fairness metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.sparse import csr_matrix


def _precision_recall_ndcg(
    recommended: list[int], relevant: set[int], k: int
) -> tuple[float, float, float]:
    rec = recommended[:k]
    if not rec:
        return 0.0, 0.0, 0.0
    hits = [1 if it in relevant else 0 for it in rec]
    prec = sum(hits) / k
    rel_count = len(relevant)
    recall = (sum(hits) / rel_count) if rel_count else 0.0
    dcg = 0.0
    for rank, h in enumerate(hits, start=1):
        if h:
            dcg += 1.0 / np.log2(rank + 1)
    ideal_hits = min(rel_count, k)
    idcg = sum(1.0 / np.log2(i + 1) for i in range(1, ideal_hits + 1)) if ideal_hits else 1.0
    ndcg = dcg / idcg if idcg > 0 else 0.0
    return float(prec), float(recall), float(ndcg)


def position_weight(rank_1based: int) -> float:
    return 1.0 / np.log2(rank_1based + 1)


@dataclass
class FairnessAccumulator:
    weighted_m: float = 0.0
    weighted_f: float = 0.0
    sum_rank_m: float = 0.0
    sum_rank_f: float = 0.0
    count_m: int = 0
    count_f: int = 0
    distinct_m: set[int] | None = None
    distinct_f: set[int] | None = None

    def __post_init__(self) -> None:
        if self.distinct_m is None:
            self.distinct_m = set()
        if self.distinct_f is None:
            self.distinct_f = set()

    def add_recommendation(self, items: Iterable[int], labels: dict[int, str]) -> None:
        for rank, it in enumerate(items, start=1):
            lab = labels.get(it, "unknown")
            w = position_weight(rank)
            if lab == "male":
                self.weighted_m += w
                self.sum_rank_m += rank
                self.count_m += 1
                self.distinct_m.add(it)
            elif lab == "female":
                self.weighted_f += w
                self.sum_rank_f += rank
                self.count_f += 1
                self.distinct_f.add(it)

    def exposure_gap(self) -> float | None:
        """Male share of position-weighted exposure among labeled M/F recs."""
        denom = self.weighted_m + self.weighted_f
        if denom <= 0:
            return None
        return float(self.weighted_m / denom)

    def avg_rank_gap(self) -> float | None:
        """Mean rank (1-based) difference: male minus female (lower is better for F)."""
        if self.count_m == 0 or self.count_f == 0:
            return None
        mean_m = self.sum_rank_m / self.count_m
        mean_f = self.sum_rank_f / self.count_f
        return float(mean_m - mean_f)

    def coverage_gap(self) -> float | None:
        """Difference in distinct artist counts (male - female) in rec lists."""
        if self.distinct_m is None or self.distinct_f is None:
            return None
        return float(len(self.distinct_m) - len(self.distinct_f))


def train_exposure_share(
    train: csr_matrix,
    item_labels: dict[int, str],
) -> dict[str, float | None]:
    """Position-agnostic exposure share on training interactions (M/F only)."""
    train_coo = train.tocoo()
    col_labels = np.array([item_labels.get(int(i), "unknown") for i in train_coo.col])
    w_m = float(train_coo.data[col_labels == "male"].sum())
    w_f = float(train_coo.data[col_labels == "female"].sum())
    denom = w_m + w_f
    if denom <= 0:
        return {"male_share": None, "female_share": None}
    return {"male_share": w_m / denom, "female_share": w_f / denom}
