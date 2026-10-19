"""Backup artwork source: iTunes Search API (free, no key).

Used when Last.fm has no real artwork for a track — Last.fm serves a
generic white-star placeholder in that case, which lastfm.py filters out.
Best-effort by design: any failure returns "" and the UI renders its own
placeholder instead.
"""

from __future__ import annotations

import requests

ITUNES_URL = "https://itunes.apple.com/search"
TIMEOUT = 8  # seconds


def fetch_itunes_artwork(title: str, artist: str) -> str:
    """Artwork URL for a track via iTunes Search, or "" when unknown.

    Returns the 300x300 variant (Apple serves larger sizes through the same
    URL pattern as the 100x100 artworkUrl100).
    """
    if not title or not artist:
        return ""
    try:
        r = requests.get(
            ITUNES_URL,
            params={"term": f"{artist} {title}", "media": "music",
                    "entity": "song", "limit": 1},
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        results = r.json().get("results") or []
    except Exception:
        # Best-effort by contract: artwork must never break a mine.
        return ""
    if not results:
        return ""
    url = str(results[0].get("artworkUrl100") or "")
    return url.replace("100x100bb", "300x300bb")
