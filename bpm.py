"""GetSongBPM tempo lookups for TuneSage.

Fills the BPM gap Last.fm leaves: candidates get a real tempo (beats per
minute) so the UI can offer tempo-based streams (upbeat / slow burn).

Design notes:
- Results are cached persistently in SQLite (db.bpm_cache) — tempo never
  changes, so each unique track is looked up once, ever. This keeps us far
  under the free quota no matter how often the user refreshes.
- Matching is defensive: we search by title, then prefer the result whose
  artist name matches ours; otherwise we take the top hit. A miss returns
  None rather than a wrong number — no tempo beats a wrong tempo.
- One bad lookup never kills a batch: fill_tempos() catches per-track.

Get a free key at https://getsongbpm.com/api (terms require a backlink —
see the dashboard footer).
"""

from __future__ import annotations

import time

import requests

import db

API_URL = "https://api.getsong.co/search/"
TIMEOUT = 10  # seconds


class BpmError(RuntimeError):
    """Raised when a GetSongBPM request fails."""


def _parse_tempo(value) -> float | None:
    try:
        tempo = float(str(value))
    except (TypeError, ValueError):
        return None
    return tempo if tempo > 0 else None


def get_tempo(api_key: str, title: str, artist: str) -> float | None:
    """Look up a track's tempo. Returns BPM or None when unknown.

    Raises BpmError on transport/API failures (unknown track -> None).
    """
    if not api_key or not title:
        raise BpmError("API key and title are required.")
    try:
        r = requests.get(
            API_URL,
            params={"api_key": api_key, "type": "song",
                    "lookup": f"song:{title}"},
            timeout=TIMEOUT,
        )
    except requests.RequestException as e:
        raise BpmError(f"GetSongBPM request failed: {e}") from e
    if r.status_code == 429:
        raise BpmError("GetSongBPM rate limit hit (HTTP 429).")
    if r.status_code != 200:
        raise BpmError(
            f"GetSongBPM error (HTTP {r.status_code}): {r.text[:200]}")
    try:
        body = r.json()
    except ValueError as e:
        raise BpmError("GetSongBPM returned non-JSON response.") from e
    if isinstance(body, dict) and body.get("error"):
        # e.g. {"error": "Invalid API Key, or inactive."} — a key problem,
        # not an unknown track. Must raise, not return None.
        raise BpmError(f"GetSongBPM: {body['error']}")
    results = body.get("search") or []
    if not results:
        return None  # unknown track — not an error
    want = (artist or "").strip().lower()

    def _artist_of(item: dict) -> str:
        a = item.get("artist")
        if isinstance(a, dict):
            return str(a.get("name", "") or "").strip().lower()
        return str(a or "").strip().lower()

    for item in results:
        if want and want in _artist_of(item):
            if (t := _parse_tempo(item.get("tempo"))) is not None:
                return t
    # no artist match: fall back to the top hit's tempo, if it has one
    for item in results:
        if (t := _parse_tempo(item.get("tempo"))) is not None:
            return t
    return None


def fill_tempos(api_key: str, candidates: list[dict],
                sleep: float = 0.2) -> int:
    """Attach tempo to candidates in place. Returns # of API lookups made.

    Checks the persistent cache first; only cache misses hit the network.
    Per-track failures are swallowed (tempo stays None) so one bad lookup
    can't break a refresh.
    """
    if not api_key:
        return 0
    lookups = 0
    for c in candidates:
        if c.get("tempo"):
            continue
        title, artist = c.get("title", ""), c.get("artist", "")
        try:
            cached = db.get_tempo(title, artist)
            if cached is not None:
                c["tempo"] = cached
                continue
            tempo = get_tempo(api_key, title, artist)
            lookups += 1
            time.sleep(sleep)
        except BpmError as e:
            if "api key" in str(e).lower():
                raise  # key-level problem: abort so the caller surfaces it
            continue  # one bad track must not break the batch
        if tempo is not None:
            db.set_tempo(title, artist, tempo)
            c["tempo"] = tempo
    return lookups
