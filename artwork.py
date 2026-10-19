"""Artwork source: iTunes Search API (free, no key).

Primary artwork for mined candidates: Last.fm's generic placeholder can't
be detected reliably, while iTunes has real artwork for virtually all
commercial tracks. Best-effort by design: any failure returns "" and
callers fall back to whatever art they have.
"""

from __future__ import annotations

import re

import requests

ITUNES_URL = "https://itunes.apple.com/search"
TIMEOUT = 8  # seconds
MAX_RESULTS = 5


def _norm_title(t: str) -> str:
    """Lowercase, drop (feat. ...) / [...] segments, collapse whitespace."""
    t = t.lower()
    t = re.sub(r"\([^)]*\)", "", t)
    t = re.sub(r"\[[^\]]*\]", "", t)
    return re.sub(r"\s+", " ", t).strip()


def _artist_match(found: str, want: str) -> bool:
    found, want = found.strip().lower(), want.strip().lower()
    return bool(found) and (want in found or found in want)


def fetch_itunes_artwork(title: str, artist: str) -> str:
    """Artwork URL for a track via iTunes Search, or "" when unknown.

    Scans the top hits for one matching both artist and (normalized) title —
    otherwise we'd display another track's artwork. Returns the 300x300
    variant (Apple serves larger sizes through the same URL pattern as the
    100x100 artworkUrl100).
    """
    if not title or not artist:
        return ""
    try:
        r = requests.get(
            ITUNES_URL,
            params={"term": f"{artist} {_norm_title(title)}", "media": "music",
                    "entity": "song", "limit": MAX_RESULTS},
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        results = r.json().get("results") or []
    except Exception:
        # Best-effort by contract: artwork must never break a mine.
        return ""
    want_title = _norm_title(title)
    for top in results:
        if not _artist_match(str(top.get("artistName") or ""), artist):
            continue
        found_title = _norm_title(str(top.get("trackName") or ""))
        if found_title and (found_title == want_title
                            or want_title in found_title
                            or found_title in want_title):
            url = str(top.get("artworkUrl100") or "")
            return url.replace("100x100bb", "300x300bb")
    return ""
