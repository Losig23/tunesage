"""Spotify playback integration (Web API) — playback only.

Recommendations still come from Last.fm; nothing Spotify returns feeds the
taste model. (Spotify's Developer Policy bans training AI/ML on its
content; searching + playing tracks for the user's own listening is
ordinary API use, which is why this module exists but an import never did.)

OAuth 2.0 authorization-code flow:
  1. /api/spotify/login redirects the browser to authorize_url().
  2. Spotify redirects back to the redirect URI with ?code=...
  3. exchange_code() trades the code for access + refresh tokens.
  4. Access tokens live ~1h; refresh_access_token() renews them silently.

Playback needs Spotify Premium (the API returns 403 PREMIUM_REQUIRED
otherwise) and an available device — if the phone's Spotify app is open,
it shows up in get_devices() and can be targeted directly.
"""

from __future__ import annotations

from urllib.parse import urlencode

import requests

from artwork import _artist_match, _norm_title

ACCOUNTS_URL = "https://accounts.spotify.com"
API_URL = "https://api.spotify.com/v1"
SCOPES = "user-modify-playback-state user-read-playback-state"
TIMEOUT = 10  # seconds


class SpotifyError(Exception):
    """Any Spotify transport/API failure, with a user-readable message."""


def authorize_url(client_id: str, redirect_uri: str, state: str) -> str:
    """The accounts.spotify.com URL the user approves access on."""
    params = {
        "response_type": "code",
        "client_id": client_id,
        "scope": SCOPES,
        "redirect_uri": redirect_uri,
        "state": state,
    }
    return f"{ACCOUNTS_URL}/authorize?{urlencode(params)}"


def _token_request(client_id: str, client_secret: str,
                   data: dict) -> dict:
    """POST to the token endpoint; returns the parsed token payload."""
    try:
        r = requests.post(
            f"{ACCOUNTS_URL}/api/token",
            data=data,
            auth=(client_id, client_secret),
            timeout=TIMEOUT,
        )
    except requests.RequestException as e:
        raise SpotifyError(f"Spotify token request failed: {e}") from e
    try:
        body = r.json()
    except ValueError:
        body = {}
    if r.status_code != 200:
        desc = body.get("error_description") or body.get("error") or r.text[:200]
        raise SpotifyError(f"Spotify token error: {desc}")
    return body


def exchange_code(client_id: str, client_secret: str, code: str,
                  redirect_uri: str) -> dict:
    """Trade an authorization code for access + refresh tokens."""
    return _token_request(client_id, client_secret, {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri,
    })


def refresh_access_token(client_id: str, client_secret: str,
                         refresh_token: str) -> dict:
    """Renew an access token. Spotify may rotate the refresh token."""
    return _token_request(client_id, client_secret, {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
    })


def _api(access_token: str, method: str, path: str, **kwargs) -> dict | None:
    """One authenticated Web API call; None on 204 No Content."""
    try:
        r = requests.request(
            method, f"{API_URL}{path}",
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=TIMEOUT, **kwargs)
    except requests.RequestException as e:
        raise SpotifyError(f"Spotify request failed: {e}") from e
    if r.status_code in (200, 201, 202, 204):
        if r.status_code == 204 or not r.content:
            return None
        try:
            return r.json()
        except ValueError:
            return None
    # Error shape: {"error": {"status": n, "message": "...", "reason": ...}}
    message, reason = "", ""
    try:
        err = r.json().get("error") or {}
        if isinstance(err, dict):
            message = err.get("message") or ""
            reason = err.get("reason") or ""
    except ValueError:
        pass
    if r.status_code == 401:
        raise SpotifyError("Spotify session expired — reconnect Spotify.")
    if r.status_code == 403:
        if reason == "PREMIUM_REQUIRED":
            raise SpotifyError(
                "Spotify Premium is required for playback control.")
        raise SpotifyError(f"Spotify refused the request: {message}")
    if r.status_code == 404:
        raise SpotifyError(
            "No active Spotify device — open Spotify on your phone first.")
    if r.status_code == 429:
        raise SpotifyError("Spotify rate limit hit — try again in a moment.")
    raise SpotifyError(
        f"Spotify error (HTTP {r.status_code}){': ' + message if message else ''}")


def search_track(access_token: str, title: str, artist: str) -> str | None:
    """Find a track's Spotify URI, or None when nothing confidently matches.

    Same philosophy as the artwork matching: a wrong song is worse than
    no song, so a hit must match both artist and (normalized) title.
    """
    q = f"track:{title} artist:{artist}"
    body = _api(access_token, "GET", "/search",
                params={"q": q, "type": "track", "limit": 5}) or {}
    items = ((body.get("tracks") or {}).get("items")) or []

    def _artist_ok(item: dict) -> bool:
        names = [a.get("name", "") for a in item.get("artists") or []]
        return any(_artist_match(n, artist) for n in names)

    # Exact title first: "(Remix)" normalizes away, but a remix is a
    # different track — only fall back to the normalized match after.
    for item in items:
        if _artist_ok(item) \
                and item.get("name", "").strip().lower() == title.strip().lower():
            return item.get("uri")
    want_title = _norm_title(title)
    for item in items:
        if _artist_ok(item) and _norm_title(item.get("name", "")) == want_title:
            return item.get("uri")
    return None


def get_devices(access_token: str) -> list[dict]:
    """The user's available Spotify Connect devices."""
    body = _api(access_token, "GET", "/me/player/devices") or {}
    return body.get("devices") or []


def play_track(access_token: str, track_uri: str,
               device_id: str | None = None) -> None:
    """Start playing a track (optionally on a specific device)."""
    params = {"device_id": device_id} if device_id else {}
    _api(access_token, "PUT", "/me/player/play",
         params=params, json={"uris": [track_uri]})


def transfer_playback(access_token: str, device_id: str) -> None:
    """Move playback to a device (wakes it up so play can target it)."""
    _api(access_token, "PUT", "/me/player",
         json={"device_ids": [device_id], "play": False})


def play_recommendation(access_token: str, title: str,
                        artist: str) -> str:
    """Search + play one track; returns the device name it started on.

    Device choice: the active device, else a smartphone, else the first
    available device (transferred to first, if it wasn't active).
    Raises SpotifyError with a user-facing message on any failure.
    """
    uri = search_track(access_token, title, artist)
    if not uri:
        raise SpotifyError(f"Couldn't find '{title}' by {artist} on Spotify.")
    devices = get_devices(access_token)
    if not devices:
        raise SpotifyError(
            "No Spotify device found — open Spotify on your phone first.")
    device = (next((d for d in devices if d.get("is_active")), None)
              or next((d for d in devices
                       if (d.get("type") or "").lower() == "smartphone"), None)
              or devices[0])
    if not device.get("is_active"):
        transfer_playback(access_token, device["id"])
    play_track(access_token, uri, device_id=device["id"])
    return device.get("name") or "Spotify"
