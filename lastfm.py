"""Last.fm integration for TuneSage.

Imports the user's recent scrobbles (user.getrecenttracks) into the
listening history instead of logging listens by hand. Last.fm scrobbling
apps (including the official one, which can scrobble from Spotify) keep a
play-by-play record of what you listen to.

Why Last.fm and not Spotify: Spotify's Developer Policy explicitly
prohibits using its platform/content to train ML/AI models — which is
exactly what TuneSage does with listening history. Last.fm's API has no
such restriction (non-commercial use with attribution).

Config comes from the environment (never hardcoded, never logged):
  LASTFM_API_KEY, LASTFM_USERNAME

Notes on the API:
- Errors come back as HTTP 200 with an {"error": code, "message": ...}
  body, so we must check the body, not just the status.
- The currently-playing track has no "date" (it hasn't finished yet) —
  we skip it.
- Rate guidance is ~5 requests/sec; pagination sleeps between pages.
"""

from __future__ import annotations

import math
import time
from datetime import datetime, timezone
from urllib.parse import quote_plus

import artwork

# Last.fm's generic "no artwork" placeholder (white star box). Filtered out
# in _largest_image so the UI renders its own placeholder instead.
_LASTFM_DEFAULT_ART_HASH = "2a96cbd8b46e442fc41c2b86b821b4ce"

import requests

API_URL = "https://ws.audioscrobbler.com/2.0/"
TIMEOUT = 10  # seconds
MAX_REQ_PER_SEC = 5


class LastfmError(RuntimeError):
    """Raised when a Last.fm request fails (network, HTTP, or API error)."""


def _text(value) -> str:
    """Last.fm nests strings as {"#text": "..."}; tolerate plain strings."""
    if isinstance(value, dict):
        return str(value.get("#text", "") or "")
    return str(value or "")


def _largest_image(item: dict) -> str:
    """Pick the biggest artwork URL from a Last.fm image list.

    Items look like {"image": [{"#text": url, "size": "small"}, ...]} with
    sizes small < medium < large < extralarge. Returns "" when absent —
    callers render a placeholder.

    Last.fm serves a generic white-star placeholder for tracks with no real
    artwork; its URL always contains _LASTFM_DEFAULT_ART_HASH, so we treat
    those as missing too (otherwise every obscure track shows a white box).
    """
    images = _as_list(item.get("image"))
    rank = {"small": 0, "medium": 1, "large": 2, "extralarge": 3, "mega": 4}
    best, best_rank = "", -1
    for img in images:
        if not isinstance(img, dict):
            continue
        url = _text(img.get("#text")).strip()
        if not url or _LASTFM_DEFAULT_ART_HASH in url:
            continue
        r = rank.get(str(img.get("size") or "").lower(), -1)
        if url and r >= best_rank:
            best, best_rank = url, r
    return best


def _request(api_key: str, params: dict) -> dict:
    """Single API call. Raises LastfmError on any failure.

    The key is passed as a request param (Last.fm's design) and is never
    included in exception messages or logs.
    """
    try:
        r = requests.get(API_URL,
                         params={"api_key": api_key, "format": "json",
                                 **params},
                         timeout=TIMEOUT)
    except requests.RequestException as e:
        raise LastfmError(f"Last.fm request failed: {e}") from e
    if r.status_code == 429:
        raise LastfmError(
            "Last.fm rate limit hit (HTTP 429). Wait a minute and retry.")
    if r.status_code != 200:
        raise LastfmError(
            f"Last.fm API error (HTTP {r.status_code}): {r.text[:200]}")
    try:
        body = r.json()
    except ValueError as e:
        raise LastfmError("Last.fm returned non-JSON response.") from e
    # Last.fm reports API errors as HTTP 200 with an {error, message} body.
    if isinstance(body, dict) and "error" in body:
        raise LastfmError(
            f"Last.fm error {body.get('error')}: {body.get('message')}")
    return body


def _parse_track(item: dict) -> dict | None:
    """Normalize one recenttracks entry. Returns None for the now-playing
    entry (no play timestamp yet) or malformed entries."""
    date = item.get("date") or {}
    uts = date.get("uts")
    if not uts:
        return None  # now-playing or otherwise incomplete
    try:
        played_at = datetime.fromtimestamp(int(uts), tz=timezone.utc)
    except (TypeError, ValueError):
        return None
    title = _text(item.get("name")).strip()
    artist = _text(item.get("artist")).strip()
    if not title or not artist:
        return None
    return {
        "title": title,
        "artist": artist,
        "album": _text(item.get("album")).strip(),
        # ISO-8601 UTC; lexicographic order == chronological order.
        "played_at": played_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "image": _largest_image(item),
    }


def get_recent_tracks(api_key: str, username: str,
                      limit: int = 50, page: int = 1) -> list[dict]:
    """One page of recent scrobbles, newest first, normalized to our shape."""
    if not api_key or not username:
        raise LastfmError(
            "LASTFM_API_KEY and LASTFM_USERNAME must be set "
            "(see README 'Last.fm integration').")
    body = _request(api_key, {
        "method": "user.getrecenttracks",
        "user": username,
        "limit": max(1, min(200, limit)),
        "page": max(1, page),
    })
    items = (body.get("recenttracks") or {}).get("track") or []
    if isinstance(items, dict):  # single track comes back as an object
        items = [items]
    tracks = [_parse_track(i) for i in items]
    return [t for t in tracks if t is not None]


def get_all_recent_tracks(api_key: str, username: str,
                          total_limit: int = 200,
                          per_page: int = 200) -> list[dict]:
    """Paginate recent scrobbles up to total_limit, newest first.

    Stops early when a page comes back short (no more history). Sleeps
    between pages to stay under Last.fm's ~5 req/sec guidance.
    """
    per_page = max(1, min(200, per_page))
    tracks: list[dict] = []
    page = 1
    while len(tracks) < total_limit:
        batch = get_recent_tracks(api_key, username,
                                  limit=per_page, page=page)
        if not batch:
            break
        tracks.extend(batch)
        if len(batch) < per_page:
            break  # last page
        page += 1
        time.sleep(1.0 / MAX_REQ_PER_SEC)
    return tracks[:total_limit]


# ---------------------------------------------------------------------------
# Candidate mining: turn "artists like mine" into recommendable tracks.
# Used by forum.mine_recommendations as the live candidate source (Reddit's
# anonymous JSON API now 403s script clients). Same politeness rules as the
# scrobble import: short sleeps between calls, ~5 req/sec max.
# ---------------------------------------------------------------------------

def _as_list(value) -> list:
    """Last.fm returns a single object instead of a list when there's one."""
    if not value:
        return []
    return value if isinstance(value, list) else [value]


def _to_int(value, default: int = 0) -> int:
    try:
        return int(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return default


def _to_float(value, default: float = 0.0) -> float:
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return default


def get_similar_artists(api_key: str, artist: str,
                        limit: int = 5) -> list[dict]:
    """Artists Last.fm considers similar: [{name, match}].

    `match` is a 0–1 similarity score. Raises LastfmError on any failure.
    """
    if not api_key or not artist:
        raise LastfmError("API key and artist are required.")
    body = _request(api_key, {
        "method": "artist.getSimilar",
        "artist": artist,
        "limit": max(1, min(100, limit)),
    })
    items = _as_list((body.get("similarartists") or {}).get("artist"))
    out = []
    for item in items:
        name = _text(item.get("name")).strip()
        if not name:
            continue
        out.append({"name": name,
                    "match": _to_float(item.get("match"))})
    return out


def get_artist_top_tracks(api_key: str, artist: str,
                          limit: int = 5) -> list[dict]:
    """An artist's most-played tracks: [{title, artist, listeners, playcount}].

    Raises LastfmError on any failure.
    """
    if not api_key or not artist:
        raise LastfmError("API key and artist are required.")
    body = _request(api_key, {
        "method": "artist.getTopTracks",
        "artist": artist,
        "limit": max(1, min(50, limit)),
    })
    items = _as_list((body.get("toptracks") or {}).get("track"))
    out = []
    for item in items:
        title = _text(item.get("name")).strip()
        if not title:
            continue
        # Nested artist is {"name": ...} here (unlike recenttracks' #text).
        nested = item.get("artist")
        nested_name = (_text(nested)
                       or (nested.get("name") if isinstance(nested, dict)
                           else ""))
        out.append({
            "title": title,
            "artist": nested_name.strip() or artist,
            "listeners": _to_int(item.get("listeners")),
            "playcount": _to_int(item.get("playcount")),
            "image": _largest_image(item),
        })
    return out


def _mentions_from_listeners(listeners: int) -> int:
    """Bucket raw listener counts onto the old mention-count scale.

    forum_score() log-scales `mentions` and caps at ~10, so feeding it raw
    listener counts (millions) would pin every candidate at the max and kill
    differentiation. log10 bucketing keeps the same ordering and spread on a
    1–7ish scale: 10M listeners -> 7, 100k -> 5, 1k -> 3.
    """
    return max(1, round(math.log10(max(1, listeners))))


def mine_candidates(api_key: str, seed_artists: list[str],
                    max_artists: int = 4, similar_per_artist: int = 5,
                    tracks_per_artist: int = 2, hop2_seeds: int = 2,
                    hop2_similar: int = 3, sleep: float = 0.25) -> dict:
    """Build recommendation candidates from Last.fm similarity data.

    Two-hop traversal: for each seed artist, take similar artists (hop 1),
    then similar artists *of the closest hop-1 matches* (hop 2, match score
    decayed 0.8x per hop). Hop 2 is where real discovery happens — artists
    two steps removed from your listening, which a flat similar-artists
    list would never surface. Then top tracks for every similar artist found.

    Returns {"candidates": [...], "sources": [{"title", "url"}]}.
    Candidate dicts match the shape recommender.py expects
    (title, artist, genre, tags, mentions, listeners, via, image,
    sources, snippets).

    Raises LastfmError (or RuntimeError when nothing comes back) — the
    caller falls back to the built-in sample dataset.
    """
    if not api_key:
        raise LastfmError(
            "LASTFM_API_KEY must be set (see README 'Last.fm integration').")
    artists = [a for a in dict.fromkeys(seed_artists) if a][:max_artists]
    if not artists:
        raise RuntimeError("no seed artists to mine from")
    seed_set = {a.lower() for a in artists}

    # Hop 1 + hop 2 similar-artist expansion. Every network call is
    # individually guarded: one flaky artist (Last.fm has been serving
    # intermittent 503s) must degrade the mine, never kill it.
    similar: dict[str, dict] = {}  # name -> {"match", "via"}
    sources: list[dict] = []
    for seed in artists:
        try:
            hop1 = get_similar_artists(api_key, seed, limit=similar_per_artist)
        except LastfmError:
            continue  # one bad seed must not kill the whole mine
        time.sleep(sleep)
        for sim in hop1:
            name = sim["name"]
            if name.lower() in seed_set:
                continue
            if name not in similar or sim["match"] > similar[name]["match"]:
                similar[name] = {"match": sim["match"], "via": seed}
        for hop1sim in sorted(hop1, key=lambda s: s["match"],
                              reverse=True)[:hop2_seeds]:
            try:
                hop2 = get_similar_artists(api_key, hop1sim["name"],
                                           limit=hop2_similar)
            except LastfmError:
                continue  # one bad hop must not kill the whole mine
            time.sleep(sleep)
            for sim in hop2:
                name = sim["name"]
                if name.lower() in seed_set or name in similar:
                    continue
                similar[name] = {
                    "match": sim["match"] * 0.8,  # similarity dilutes w/ distance
                    "via": f"{seed} → {hop1sim['name']}",
                }
    for sim_name, info in similar.items():
        sources.append({
            "title": (f"Last.fm: top tracks of {sim_name} "
                      f"(similar to {info['via']}, match {info['match']:.2f})"),
            "url": "https://www.last.fm/music/" + quote_plus(sim_name),
        })

    aggregate: dict[tuple[str, str], dict] = {}
    for sim_name, info in similar.items():
        match = info["match"]
        try:
            tracks = get_artist_top_tracks(api_key, sim_name,
                                           limit=tracks_per_artist)
        except LastfmError:
            continue  # skip the flaky artist, keep the rest
        time.sleep(sleep)
        for t in tracks:
            key = (t["title"].lower(), t["artist"].lower())
            mentions = _mentions_from_listeners(t["listeners"])
            snippet = (f"Top track of {t['artist']} — similar to {info['via']} "
                       f"on Last.fm (match {match:.2f}); "
                       f"{t['listeners']:,} listeners")
            existing = aggregate.get(key)
            if existing:
                # Same track via two similar artists: keep the stronger
                # signal, merge provenance.
                if mentions > existing["mentions"]:
                    existing["mentions"] = mentions
                    existing["listeners"] = t["listeners"]
                    existing["snippets"] = [snippet]
                for s in (f"https://www.last.fm/music/"
                          + quote_plus(t["artist"]),):
                    if s not in existing["sources"]:
                        existing["sources"].append(s)
            else:
                aggregate[key] = {
                    "title": t["title"], "artist": t["artist"],
                    "genre": "", "tags": [],
                    "mentions": mentions,
                    "listeners": t["listeners"],
                    "via": info["via"],  # which of your artists led us here
                    # Last.fm art, else iTunes fallback, else UI placeholder.
                    "image": (t.get("image", "")
                              or artwork.fetch_itunes_artwork(
                                  t["title"], t["artist"])),
                    "sources": ["https://www.last.fm/music/"
                                + quote_plus(t["artist"])],
                    "snippets": [snippet],
                }
    candidates = sorted(aggregate.values(),
                        key=lambda c: c["mentions"], reverse=True)
    if not candidates:
        raise RuntimeError("Last.fm mining returned no candidates")
    return {"candidates": candidates, "sources": sources}
