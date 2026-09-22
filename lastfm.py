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

import time
from datetime import datetime, timezone

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
