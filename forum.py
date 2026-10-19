"""Candidate mining for TuneSage: find songs worth recommending.

Live path: Last.fm API (see lastfm.mine_candidates) — top tracks of artists
similar to the user's most-played artists. Listener counts become the
social-proof signal (log-scaled onto the old mention-count scale, so the
scorer in recommender.py is untouched).

History: this module originally mined Reddit's public JSON API for
"songs like X" threads, but Reddit now returns HTTP 403 to unauthenticated
script clients, so that path (and its regex extraction helpers) was removed
on 2026-10-11.

Resilience: on any failure we fall back to FALLBACK_CANDIDATES, a small
built-in sample dataset, so /api/recommendations ALWAYS returns data.
"""

from __future__ import annotations

import lastfm as lastfm_client


def mine_recommendations(seed_songs: list[dict], api_key: str = "",
                         progress_cb=None) -> dict:
    """Mine candidate songs similar to the user's top seeds.

    Returns {"candidates": [...], "threads": [...],
             "source": "lastfm"|"fallback", "error": str|None}.
    "threads" keeps its name for API/UI compatibility but now carries source
    provenance entries (Last.fm similar-artist pages). Never raises: any
    failure anywhere -> built-in fallback dataset, with the cause captured
    in "error" so the UI can say *why* instead of just "unreachable".
    progress_cb(stage, frac) is called with 0-1 progress for the UI.
    """
    if api_key:
        try:
            artists = []
            for s in seed_songs:
                a = (s.get("artist") or "").strip()
                if a and a not in artists:
                    artists.append(a)
            mined = lastfm_client.mine_candidates(api_key, artists,
                                                  progress_cb=progress_cb)
            threads = [{"title": s["title"], "subreddit": "",
                        "url": s["url"]} for s in mined["sources"]]
            return {"candidates": mined["candidates"],
                    "threads": threads,
                    "source": "lastfm", "error": None}
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
    else:
        error = "LASTFM_API_KEY is not set"
    if progress_cb:
        progress_cb("Using offline sample data", 1.0)
    return {"candidates": [dict(c) for c in FALLBACK_CANDIDATES],
            "threads": [{"title": "(fallback sample dataset — "
                                 "live mining unreachable)",
                         "subreddit": "", "url": ""}],
            "source": "fallback", "error": error}


# ---------------------------------------------------------------------------
# Built-in fallback dataset: realistic recommendation content so the pipeline,
# scoring, and UI always have something to work with offline.
# ---------------------------------------------------------------------------
FALLBACK_CANDIDATES = [
    {"title": "Blinding Lights", "artist": "The Weeknd", "genre": "pop",
     "tags": ["energetic", "night-drive"], "mentions": 9, "listeners": 4200000,
     "sources": ["https://www.last.fm/music/The+Weeknd (sample)"],
     "snippets": ["I love Blinding Lights by The Weeknd, absolute masterpiece "
                  "for night drives, such a catchy synth gem"]},
    {"title": "Levitating", "artist": "Dua Lipa", "genre": "pop",
     "tags": ["energetic", "dance"], "mentions": 7, "listeners": 2100000,
     "sources": ["https://www.last.fm/music/Dua+Lipa (sample)"],
     "snippets": ["Levitating by Dua Lipa is fantastic, incredible groove, "
                  "highly recommend if you like dance pop"]},
    {"title": "Heat Waves", "artist": "Glass Animals", "genre": "indie",
     "tags": ["chill", "dreamy"], "mentions": 6, "listeners": 1600000,
     "sources": ["https://www.last.fm/music/Glass+Animals (sample)"],
     "snippets": ["Heat Waves by Glass Animals is beautiful and haunting, "
                  "love the dreamy vibe"]},
    {"title": "Sweater Weather", "artist": "The Neighbourhood", "genre": "indie rock",
     "tags": ["chill", "moody"], "mentions": 5, "listeners": 900000,
     "sources": ["https://www.last.fm/music/The+Neighbourhood (sample)"],
     "snippets": ["Sweater Weather by The Neighbourhood, great moody indie "
                  "classic, always recommend"]},
    {"title": "HUMBLE.", "artist": "Kendrick Lamar", "genre": "hip-hop",
     "tags": ["energetic", "bold"], "mentions": 8, "listeners": 2800000,
     "sources": ["https://www.last.fm/music/Kendrick+Lamar (sample)"],
     "snippets": ["HUMBLE. by Kendrick Lamar is brilliant, powerful beat, "
                  "an amazing hip-hop track"]},
    {"title": "SICKO MODE", "artist": "Travis Scott", "genre": "hip-hop",
     "tags": ["energetic", "hype"], "mentions": 6, "listeners": 1900000,
     "sources": ["https://www.last.fm/music/Travis+Scott (sample)"],
     "snippets": ["SICKO MODE by Travis Scott goes so hard, awesome production, "
                  "love the beat switches"]},
    {"title": "Take Five", "artist": "Dave Brubeck Quartet", "genre": "jazz",
     "tags": ["chill", "sophisticated"], "mentions": 4, "listeners": 320000,
     "sources": ["https://www.last.fm/music/Dave+Brubeck+Quartet (sample)"],
     "snippets": ["Take Five by Dave Brubeck Quartet is a gorgeous jazz "
                  "masterpiece, timeless and brilliant"]},
    {"title": "So What", "artist": "Miles Davis", "genre": "jazz",
     "tags": ["chill", "cool"], "mentions": 4, "listeners": 280000,
     "sources": ["https://www.last.fm/music/Miles+Davis (sample)"],
     "snippets": ["So What by Miles Davis, the best modal jazz track, "
                  "beautiful and classic"]},
    {"title": "Strobe", "artist": "deadmau5", "genre": "electronic",
     "tags": ["chill", "epic"], "mentions": 5, "listeners": 750000,
     "sources": ["https://www.last.fm/music/deadmau5 (sample)"],
     "snippets": ["Strobe by deadmau5 is incredible, such a powerful build, "
                  "an electronic gem, gives me chills"]},
    {"title": "Midnight City", "artist": "M83", "genre": "electronic",
     "tags": ["dreamy", "night-drive"], "mentions": 5, "listeners": 1100000,
     "sources": ["https://www.last.fm/music/M83 (sample)"],
     "snippets": ["Midnight City by M83 is fantastic synthwave, love it, "
                  "perfect night drive song"]},
    {"title": "505", "artist": "Arctic Monkeys", "genre": "indie rock",
     "tags": ["moody", "romantic"], "mentions": 5, "listeners": 1300000,
     "sources": ["https://www.last.fm/music/Arctic+Monkeys (sample)"],
     "snippets": ["505 by Arctic Monkeys is amazing, haunting and beautiful, "
                  "highly recommend"]},
    {"title": "Fluorescent Adolescent", "artist": "Arctic Monkeys",
     "genre": "indie rock", "tags": ["energetic", "fun"], "mentions": 3, "listeners": 95000,
     "sources": ["https://www.last.fm/music/Arctic+Monkeys (sample)"],
     "snippets": ["Fluorescent Adolescent by Arctic Monkeys is great, catchy "
                  "and fun indie rock"]},
]
