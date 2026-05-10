"""Load Last.fm 360K (HF mirror), filter, and build train/test implicit splits."""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


def _ensure_hf_cache() -> None:
    root = os.environ.get("HF_HOME")
    if root:
        return
    # Repo root (parent of post_processing/) — share HF cache with other tooling
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    cache = os.path.join(repo_root, ".hf_cache")
    os.makedirs(cache, exist_ok=True)
    os.environ["HF_HOME"] = cache
    os.environ["HF_DATASETS_CACHE"] = cache


@dataclass
class InteractionBundle:
    """Contiguous user/item ids with train/test masks."""

    train_matrix: csr_matrix
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    user_meta: pd.DataFrame
    n_users: int
    n_items: int
    orig_artist_mbids: np.ndarray  # length n_items, aligned to new item id


def load_raw_slice(split_spec: str = "train[:1500000]", *, streaming: bool = False) -> pd.DataFrame:
    _ensure_hf_cache()
    from datasets import load_dataset

    if streaming:
        # IterableDataset does not accept HF slicing syntax; parse train[:N] and use .take(N).
        if "[" in split_spec:
            base, rest = split_spec.split("[", 1)
            inside = rest.rstrip("]")
            if ":" not in inside:
                raise ValueError("Use split like train[:800000] when streaming=True")
            _, cap_s = inside.split(":", 1)
            if not cap_s:
                raise ValueError("Streaming requires an explicit row cap, e.g. train[:800000]")
            cap = int(cap_s)
        else:
            base, cap = split_spec, None
        if cap is None:
            raise ValueError("streaming=True needs a capped split (e.g. train[:500000])")
        stream = load_dataset(
            "matthewfranglen/lastfm-360k",
            split=base,
            streaming=True,
        )
        stream = stream.take(cap)
        rows = list(stream)
        return pd.DataFrame(rows)
    ds = load_dataset("matthewfranglen/lastfm-360k", split=split_spec)
    return ds.to_pandas()


def build_bundle(
    df: pd.DataFrame,
    *,
    min_user_interactions: int = 20,
    min_artist_interactions: int = 10,
    test_frac: float = 0.2,
    random_state: int = 42,
) -> InteractionBundle:
    """Filter by activity, remap ids, per-user train/test split on interactions."""
    rng = np.random.default_rng(random_state)
    counts_u = df.groupby("user_index").size()
    counts_a = df.groupby("artist_index").size()
    keep_u = set(counts_u[counts_u >= min_user_interactions].index)
    keep_a = set(counts_a[counts_a >= min_artist_interactions].index)
    f = df[df["user_index"].isin(keep_u) & df["artist_index"].isin(keep_a)].copy()

    # Collapse duplicate (user, artist) rows by summing plays
    f = (
        f.groupby(["user_index", "artist_index"], as_index=False)
        .agg({"play_count": "sum", "musicbrainz_artist_id": "first"})
        .reset_index(drop=True)
    )

    user_ids, user_inv = np.unique(f["user_index"].to_numpy(), return_inverse=True)
    item_ids, item_inv = np.unique(f["artist_index"].to_numpy(), return_inverse=True)
    n_users, n_items = len(user_ids), len(item_ids)

    # Map original artist_index -> mbid for contiguous item id
    mb_series = f.groupby("artist_index")["musicbrainz_artist_id"].first()
    orig_artist_mbids = np.array(
        [mb_series.loc[int(orig)] for orig in item_ids], dtype=object
    )

    f["u"] = user_inv.astype(np.int32)
    f["i"] = item_inv.astype(np.int32)

    train_rows: list[pd.Series] = []
    test_rows: list[pd.Series] = []
    for u, part in f.groupby("u"):
        idx = part.index.to_numpy()
        if len(idx) < 5:
            continue
        perm = rng.permutation(idx)
        n_test = max(1, int(round(len(perm) * test_frac)))
        test_idx = perm[:n_test]
        train_idx = perm[n_test:]
        test_rows.append(part.loc[test_idx])
        train_rows.append(part.loc[train_idx])

    train_df = pd.concat(train_rows, ignore_index=True)
    test_df = pd.concat(test_rows, ignore_index=True)

    confidence = np.log1p(train_df["play_count"].to_numpy(dtype=np.float64)).astype(np.float32)
    rows = train_df["u"].to_numpy()
    cols = train_df["i"].to_numpy()
    train_matrix = csr_matrix(
        (confidence, (rows, cols)), shape=(n_users, n_items), dtype=np.float32
    )

    user_meta = (
        df.groupby("user_index")
        .first()
        .reset_index()[["user_index", "gender", "age", "country"]]
    )
    user_meta = user_meta[user_meta["user_index"].isin(user_ids)].drop_duplicates("user_index")

    return InteractionBundle(
        train_matrix=train_matrix,
        train_df=train_df,
        test_df=test_df,
        user_meta=user_meta,
        n_users=n_users,
        n_items=n_items,
        orig_artist_mbids=orig_artist_mbids,
    )
