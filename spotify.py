"""Spotify integration for TuneSage.

OAuth Authorization Code flow + Web API client. Lets the user connect their
own Spotify account and import recently-played tracks instead of logging
listens by hand.

Config comes from the environment (never hardcoded, never logged):
  SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET, SPOTIFY_REDIRECT_URI

Every request has a timeout and raises SpotifyError with a clear message on
failure. Tokens are returned to the caller for DB storage — they are never
printed or logged here.
"""

from __future__ import annotations

import os
import urllib.parse

import requests

AUTHORIZE_URL = "https://accounts.spotify.com/authorize"
TOKEN_URL = "https://accounts.spotify.com/api/token"
API_BASE = "https://api.spotify.com/v1"
SCOPE = "user-read-recently-played"
TIMEOUT = 10  # seconds


class SpotifyError(RuntimeError):
    """Raised when a Spotify request fails (config, network, or API error)."""


def _client_id() -> str:
    cid = os.environ.get("SPOTIFY_CLIENT_ID", "").strip()
    if not cid:
        raise SpotifyError(
            "SPOTIFY_CLIENT_ID is not set. Create an app at "
            "https://developer.spotify.com/dashboard and set "
            "SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET and "
            "SPOTIFY_REDIRECT_URI (see README 'Spotify integration')."
        )
    return cid


def _client_secret() -> str:
    secret = os.environ.get("SPOTIFY_CLIENT_SECRET", "").strip()
    if not secret:
        raise SpotifyError(
            "SPOTIFY_CLIENT_SECRET is not set. See README "
            "'Spotify integration' for setup."
        )
    return secret


def _redirect_uri() -> str:
    return os.environ.get("SPOTIFY_REDIRECT_URI", "").strip()


def build_authorize_url(state: str) -> str:
    """Authorization URL to send the user to. `state` is a CSRF token we
    generated; Spotify echoes it back on the redirect so we can verify it."""
    params = {
        "response_type": "code",
        "client_id": _client_id(),
        "scope": SCOPE,
        "redirect_uri": _redirect_uri(),
        "state": state,
    }
    return AUTHORIZE_URL + "?" + urllib.parse.urlencode(params)


def _token_request(data: dict) -> dict:
    """POST to the token endpoint (used by both exchange and refresh)."""
    try:
        r = requests.post(
            TOKEN_URL,
            data=data,
            auth=(_client_id(), _client_secret()),
            timeout=TIMEOUT,
        )
    except requests.RequestException as e:
        raise SpotifyError(f"Spotify token request failed: {e}") from e
    if r.status_code != 200:
        # Spotify error bodies are JSON like {"error": "invalid_grant", ...}
        # — they never contain tokens.
        raise SpotifyError(
            f"Spotify token request failed (HTTP {r.status_code}): "
            f"{r.text[:200]}"
        )
    body = r.json()
    if "access_token" not in body:
        raise SpotifyError("Spotify token response had no access_token.")
    return {
        "access_token": body["access_token"],
        # Refresh responses may omit a new refresh token — caller keeps the old.
        "refresh_token": body.get("refresh_token"),
        "expires_in": int(body.get("expires_in", 3600)),
    }


def exchange_code(code: str) -> dict:
    """Swap the authorization `code` from the callback for tokens."""
    return _token_request({
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": _redirect_uri(),
    })


def refresh_access_token(refresh_token: str) -> dict:
    """Get a fresh access token. Response may omit `refresh_token`; when it
    does, keep using the previous one (see db.get_valid_access_token)."""
    return _token_request({
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
    })


def _api_get(path: str, access_token: str, params: dict | None = None) -> dict:
    try:
        r = requests.get(
            API_BASE + path,
            headers={"Authorization": f"Bearer {access_token}"},
            params=params or {},
            timeout=TIMEOUT,
        )
    except requests.RequestException as e:
        raise SpotifyError(f"Spotify API request failed: {e}") from e
    if r.status_code != 200:
        raise SpotifyError(
            f"Spotify API error (HTTP {r.status_code}): {r.text[:200]}"
        )
    return r.json()


def get_recently_played(access_token: str, limit: int = 50) -> list[dict]:
    """Recently-played tracks, newest first, normalized to our song shape."""
    data = _api_get("/me/player/recently-played", access_token,
                    {"limit": max(1, min(50, limit))})
    tracks = []
    for item in data.get("items", []):
        track = item.get("track") or {}
        artists = [a.get("name", "") for a in track.get("artists", [])]
        tracks.append({
            "spotify_id": track.get("id", ""),
            "title": track.get("name", ""),
            "artist": ", ".join(a for a in artists if a),
            "album": (track.get("album") or {}).get("name", ""),
            "played_at": item.get("played_at", ""),
        })
    # Drop malformed entries (no title/artist can't be matched to songs).
    return [t for t in tracks if t["title"] and t["artist"]]


def get_profile(access_token: str) -> dict:
    """Who just connected — stored so the UI can show their display name."""
    data = _api_get("/me", access_token)
    return {
        "spotify_user_id": data.get("id", ""),
        "display_name": data.get("display_name") or data.get("id", ""),
    }
