"""Forum mining for TuneSage: find "songs like X" recommendation threads.

Strategy:
  1. Query Reddit's public JSON search (no auth needed) across
     r/MusicSuggestions, r/ifyoulikeblank, r/LetsTalkMusic.
  2. Extract candidate song mentions from thread titles/selftext/comments
     with regex heuristics ("X by Y", quoted titles).
  3. Aggregate: {title, artist, mentions, sources, snippets}.

Resilience: Reddit rate-limits aggressively (429/403 are common from cloud
IPs). EVERY network call is wrapped in try/except with short timeouts; on any
failure we fall back to FALLBACK_CANDIDATES, a small built-in sample dataset,
so the demo and the /api/recommendations endpoint ALWAYS return data.
"""

from __future__ import annotations

import re
import time

import requests

USER_AGENT = "TuneSage/1.0 (music recommendation demo; contact: demo@example.com)"
TIMEOUT = 8  # seconds per request
SUBREDDITS = ["MusicSuggestions", "ifyoulikeblank", "LetsTalkMusic"]

# "Title by Artist" pattern, e.g.  "Blinding Lights by The Weeknd"
BY_PATTERN = re.compile(
    r"""(?P<title>[A-Z0-9][\w'&\-\.\(\) ]{1,60}?)\s+by\s+
        (?P<artist>[A-Z][\w'&\-\. ]{1,50})""",
    re.VERBOSE,
)
# Quoted "Song Title" followed by artist-ish words
QUOTED_PATTERN = re.compile(r'"([A-Z][\w\'\-\.\(\) ]{2,60})"\s*(?:by\s+([A-Z][\w\'\-\. ]{2,50}))?')


def search_reddit(query: str, subreddit: str, limit: int = 5) -> list[dict]:
    """Search one subreddit via the public JSON API. Returns post dicts.

    Raises on network/HTTP errors — callers catch and fall back.
    """
    url = f"https://www.reddit.com/r/{subreddit}/search.json"
    resp = requests.get(
        url,
        params={"q": query, "restrict_sr": "1", "sort": "relevance", "limit": limit},
        headers={"User-Agent": USER_AGENT},
        timeout=TIMEOUT,
    )
    resp.raise_for_status()  # 429/403/etc. -> exception -> fallback path
    data = resp.json()
    return data.get("data", {}).get("children", [])


def fetch_comments(post_id: str, subreddit: str, limit: int = 10) -> list[str]:
    """Fetch top-level comment bodies for a post (best-effort)."""
    url = f"https://www.reddit.com/r/{subreddit}/comments/{post_id}.json"
    resp = requests.get(url, headers={"User-Agent": USER_AGENT},
                        params={"limit": limit}, timeout=TIMEOUT)
    resp.raise_for_status()
    payload = resp.json()
    if len(payload) < 2:
        return []
    comments = payload[1].get("data", {}).get("children", [])
    bodies = []
    for c in comments:
        body = (c.get("data") or {}).get("body") or ""
        if body and body not in ("[deleted]", "[removed]"):
            bodies.append(body)
    return bodies


def extract_candidates(text: str, source_url: str) -> list[dict]:
    """Regex heuristics to pull 'Song by Artist' mentions out of free text."""
    found: dict[tuple[str, str], dict] = {}
    for m in BY_PATTERN.finditer(text):
        title, artist = m.group("title").strip(), m.group("artist").strip()
        if len(title) < 2 or len(artist) < 2:
            continue
        key = (title.lower(), artist.lower())
        snippet = text[max(0, m.start() - 60): m.end() + 60].strip()
        entry = found.setdefault(key, {"title": title, "artist": artist,
                                       "mentions": 0, "sources": [],
                                       "snippets": [], "genre": "", "tags": []})
        entry["mentions"] += 1
        entry["snippets"].append(snippet)
        if source_url not in entry["sources"]:
            entry["sources"].append(source_url)
    for m in QUOTED_PATTERN.finditer(text):
        title = m.group(1).strip()
        artist = (m.group(2) or "Unknown Artist").strip()
        key = (title.lower(), artist.lower())
        if key in found:
            continue
        snippet = text[max(0, m.start() - 60): m.end() + 60].strip()
        found[key] = {"title": title, "artist": artist, "mentions": 1,
                      "sources": [source_url], "snippets": [snippet],
                      "genre": "", "tags": []}
    return list(found.values())


def _merge(into: dict, new: list[dict]) -> None:
    for c in new:
        key = (c["title"].lower(), c["artist"].lower())
        if key in into:
            e = into[key]
            e["mentions"] += c["mentions"]
            e["snippets"].extend(c["snippets"])
            for s in c["sources"]:
                if s not in e["sources"]:
                    e["sources"].append(s)
        else:
            into[key] = c


def mine_recommendations(seed_songs: list[dict],
                         max_threads_per_song: int = 2) -> dict:
    """Mine forums for songs similar to the user's top seeds.

    Returns {"candidates": [...], "threads": [...], "source": "reddit"|"fallback"}.
    Never raises: any failure anywhere -> built-in fallback dataset.
    """
    aggregate: dict[tuple[str, str], dict] = {}
    threads: list[dict] = []
    try:
        for song in seed_songs[:5]:  # cap seeds to stay polite
            queries = [f"songs like {song['title']} {song['artist']}",
                       f"similar to {song['artist']}"]
            for sub in SUBREDDITS:
                for q in queries:
                    posts = search_reddit(q, sub, limit=max_threads_per_song)
                    time.sleep(1.0)  # be polite: 1 req/sec
                    for post in posts:
                        pdata = post.get("data", {})
                        url = "https://www.reddit.com" + pdata.get("permalink", "")
                        threads.append({"title": pdata.get("title", ""),
                                        "subreddit": sub, "url": url})
                        text = (pdata.get("title", "") + "\n"
                                + (pdata.get("selftext") or ""))
                        _merge(aggregate, extract_candidates(text, url))
                        try:
                            for body in fetch_comments(pdata.get("id", ""), sub):
                                _merge(aggregate,
                                       extract_candidates(body, url))
                            time.sleep(1.0)
                        except Exception:
                            continue  # comments are best-effort
        if not aggregate:
            raise RuntimeError("no candidates extracted")
        return {"candidates": sorted(aggregate.values(),
                                     key=lambda c: c["mentions"], reverse=True),
                "threads": threads, "source": "reddit"}
    except Exception:
        # Network blocked / rate-limited / parse failure -> demo still works.
        return {"candidates": [dict(c) for c in FALLBACK_CANDIDATES],
                "threads": [{"title": "(fallback sample threads — Reddit unreachable)",
                             "subreddit": "demo", "url": ""}],
                "source": "fallback"}


# ---------------------------------------------------------------------------
# Built-in fallback dataset: realistic recommendation-thread content so the
# pipeline, scoring, and UI always have something to work with offline.
# ---------------------------------------------------------------------------
FALLBACK_CANDIDATES = [
    {"title": "Blinding Lights", "artist": "The Weeknd", "genre": "pop",
     "tags": ["energetic", "night-drive"], "mentions": 9,
     "sources": ["https://www.reddit.com/r/MusicSuggestions/ (sample)"],
     "snippets": ["I love Blinding Lights by The Weeknd, absolute masterpiece "
                  "for night drives, such a catchy synth gem"]},
    {"title": "Levitating", "artist": "Dua Lipa", "genre": "pop",
     "tags": ["energetic", "dance"], "mentions": 7,
     "sources": ["https://www.reddit.com/r/MusicSuggestions/ (sample)"],
     "snippets": ["Levitating by Dua Lipa is fantastic, incredible groove, "
                  "highly recommend if you like dance pop"]},
    {"title": "Heat Waves", "artist": "Glass Animals", "genre": "indie",
     "tags": ["chill", "dreamy"], "mentions": 6,
     "sources": ["https://www.reddit.com/r/ifyoulikeblank/ (sample)"],
     "snippets": ["Heat Waves by Glass Animals is beautiful and haunting, "
                  "love the dreamy vibe"]},
    {"title": "Sweater Weather", "artist": "The Neighbourhood", "genre": "indie rock",
     "tags": ["chill", "moody"], "mentions": 5,
     "sources": ["https://www.reddit.com/r/ifyoulikeblank/ (sample)"],
     "snippets": ["Sweater Weather by The Neighbourhood, great moody indie "
                  "classic, always recommend"]},
    {"title": "HUMBLE.", "artist": "Kendrick Lamar", "genre": "hip-hop",
     "tags": ["energetic", "bold"], "mentions": 8,
     "sources": ["https://www.reddit.com/r/LetsTalkMusic/ (sample)"],
     "snippets": ["HUMBLE. by Kendrick Lamar is brilliant, powerful beat, "
                  "an amazing hip-hop track"]},
    {"title": "SICKO MODE", "artist": "Travis Scott", "genre": "hip-hop",
     "tags": ["energetic", "hype"], "mentions": 6,
     "sources": ["https://www.reddit.com/r/MusicSuggestions/ (sample)"],
     "snippets": ["SICKO MODE by Travis Scott goes so hard, awesome production, "
                  "love the beat switches"]},
    {"title": "Take Five", "artist": "Dave Brubeck Quartet", "genre": "jazz",
     "tags": ["chill", "sophisticated"], "mentions": 4,
     "sources": ["https://www.reddit.com/r/LetsTalkMusic/ (sample)"],
     "snippets": ["Take Five by Dave Brubeck Quartet is a gorgeous jazz "
                  "masterpiece, timeless and brilliant"]},
    {"title": "So What", "artist": "Miles Davis", "genre": "jazz",
     "tags": ["chill", "cool"], "mentions": 4,
     "sources": ["https://www.reddit.com/r/LetsTalkMusic/ (sample)"],
     "snippets": ["So What by Miles Davis, the best modal jazz track, "
                  "beautiful and classic"]},
    {"title": "Strobe", "artist": "deadmau5", "genre": "electronic",
     "tags": ["chill", "epic"], "mentions": 5,
     "sources": ["https://www.reddit.com/r/ifyoulikeblank/ (sample)"],
     "snippets": ["Strobe by deadmau5 is incredible, such a powerful build, "
                  "an electronic gem, gives me chills"]},
    {"title": "Midnight City", "artist": "M83", "genre": "electronic",
     "tags": ["dreamy", "night-drive"], "mentions": 5,
     "sources": ["https://www.reddit.com/r/MusicSuggestions/ (sample)"],
     "snippets": ["Midnight City by M83 is fantastic synthwave, love it, "
                  "perfect night drive song"]},
    {"title": "505", "artist": "Arctic Monkeys", "genre": "indie rock",
     "tags": ["moody", "romantic"], "mentions": 5,
     "sources": ["https://www.reddit.com/r/ifyoulikeblank/ (sample)"],
     "snippets": ["505 by Arctic Monkeys is amazing, haunting and beautiful, "
                  "highly recommend"]},
    {"title": "Fluorescent Adolescent", "artist": "Arctic Monkeys",
     "genre": "indie rock", "tags": ["energetic", "fun"], "mentions": 3,
     "sources": ["https://www.reddit.com/r/MusicSuggestions/ (sample)"],
     "snippets": ["Fluorescent Adolescent by Arctic Monkeys is great, catchy "
                  "and fun indie rock"]},
]
