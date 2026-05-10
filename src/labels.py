"""Fetch and cache artist gender / type from MusicBrainz (rate-limited)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Literal

import requests

ArtistLabel = Literal["male", "female", "group", "unknown"]

SESSION = requests.Session()
SESSION.headers.update(
    {
        "User-Agent": "DSCI531AmplifiedSilence/1.0 (course project; contact: dhyeydes@usc.edu)",
        "Accept": "application/json",
    }
)


def _normalize_gender(name: str | None) -> ArtistLabel:
    if not name:
        return "unknown"
    n = name.lower()
    if n == "male":
        return "male"
    if n == "female":
        return "female"
    return "unknown"


def fetch_artist_label(mbid: str) -> ArtistLabel:
    """Return coarse artist label using MusicBrainz artist lookup."""
    if not mbid or mbid == "00000000-0000-0000-0000-000000000000":
        return "unknown"
    url = f"https://musicbrainz.org/ws/2/artist/{mbid}"
    for attempt in range(3):
        try:
            resp = SESSION.get(url, params={"fmt": "json"}, timeout=30)
            if resp.status_code == 404:
                return "unknown"
            resp.raise_for_status()
            payload = resp.json()
            break
        except (requests.RequestException, ValueError):
            time.sleep(2.0 * (attempt + 1))
            payload = None
    if not payload:
        return "unknown"

    artist_type = (payload.get("type") or "").lower()
    if artist_type == "group":
        return "group"
    if artist_type not in ("person", ""):
        return "unknown"

    gender_obj = payload.get("gender")
    if isinstance(gender_obj, dict):
        return _normalize_gender(gender_obj.get("name"))
    if isinstance(gender_obj, str):
        return _normalize_gender(gender_obj)
    return "unknown"


def load_or_fetch_labels(
    mbids: list[str],
    cache_path: Path,
    *,
    max_fetch: int | None = None,
    sleep_s: float = 1.05,
) -> dict[str, ArtistLabel]:
    """Load JSON cache {mbid: label}; fetch missing up to max_fetch."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cached: dict[str, ArtistLabel] = {}
    if cache_path.exists():
        cached = json.loads(cache_path.read_text())

    to_fetch = [m for m in mbids if m not in cached]
    fetched = 0
    for mbid in to_fetch:
        if max_fetch is not None and fetched >= max_fetch:
            break
        label = fetch_artist_label(mbid)
        cached[mbid] = label
        fetched += 1
        time.sleep(sleep_s)
        if fetched % 25 == 0:
            cache_path.write_text(json.dumps(cached))

    cache_path.write_text(json.dumps(cached))
    return cached
