"""Tests for the Spotify integration.

No real HTTP anywhere: every network call is mocked. DB tests run against a
throwaway SQLite file — the fixture rebinds the db helpers' `path` default to
it, so the Flask routes under test hit the temp DB, not tunesage.db.
"""

import functools
import time
import urllib.parse
from unittest.mock import MagicMock, patch

import pytest

import db
import spotify


@pytest.fixture
def tmpdb(tmp_path, monkeypatch):
    """Temp DB + rebind db helpers used by the routes to use it."""
    p = tmp_path / "spotify_test.db"
    db.init_db(p)

    def _bind(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            kwargs.setdefault("path", p)
            return fn(*args, **kwargs)
        return wrapper

    for name in ("save_spotify_auth", "get_spotify_auth", "clear_spotify_auth",
                 "get_valid_access_token", "log_listen", "top_songs",
                 "spotify_already_imported", "record_spotify_import",
                 "last_spotify_import_at"):
        monkeypatch.setattr(db, name, _bind(getattr(db, name)))
    return p


@pytest.fixture
def app_client(tmpdb, monkeypatch):
    monkeypatch.setenv("SECRET_KEY", "test-secret-key")
    import app as app_module  # imported after SECRET_KEY is set
    return app_module.app.test_client()


@pytest.fixture
def spotify_env(monkeypatch):
    monkeypatch.setenv("SPOTIFY_CLIENT_ID", "cid123")
    monkeypatch.setenv("SPOTIFY_CLIENT_SECRET", "csecret")
    monkeypatch.setenv("SPOTIFY_REDIRECT_URI",
                       "http://localhost:5000/api/spotify/callback")


# ---- spotify.py -----------------------------------------------------------

def test_build_authorize_url(spotify_env):
    url = spotify.build_authorize_url("state-xyz")
    parts = urllib.parse.urlparse(url)
    q = urllib.parse.parse_qs(parts.query)
    assert parts.scheme == "https" and parts.netloc == "accounts.spotify.com"
    assert q["response_type"] == ["code"]
    assert q["client_id"] == ["cid123"]
    assert q["scope"] == ["user-read-recently-played"]
    assert q["state"] == ["state-xyz"]
    assert q["redirect_uri"] == ["http://localhost:5000/api/spotify/callback"]


def test_build_authorize_url_needs_client_id(monkeypatch):
    monkeypatch.delenv("SPOTIFY_CLIENT_ID", raising=False)
    with pytest.raises(spotify.SpotifyError):
        spotify.build_authorize_url("s")


def test_exchange_code_parses_tokens(spotify_env):
    resp = MagicMock(status_code=200)
    resp.json.return_value = {"access_token": "a", "refresh_token": "r",
                             "expires_in": 3600}
    with patch("spotify.requests.post", return_value=resp) as mock_post:
        tokens = spotify.exchange_code("code123")
    assert tokens == {"access_token": "a", "refresh_token": "r",
                      "expires_in": 3600}
    assert mock_post.call_args[0][0] == \
        "https://accounts.spotify.com/api/token"
    assert mock_post.call_args[1]["data"]["grant_type"] == "authorization_code"


def test_refresh_access_token_omitted_refresh_token(spotify_env):
    """Spotify may not return a new refresh token — caller keeps the old one."""
    resp = MagicMock(status_code=200)
    resp.json.return_value = {"access_token": "new-a", "expires_in": 3600}
    with patch("spotify.requests.post", return_value=resp) as mock_post:
        tokens = spotify.refresh_access_token("old-refresh")
    assert tokens["access_token"] == "new-a"
    assert tokens["refresh_token"] is None
    assert mock_post.call_args[1]["data"]["grant_type"] == "refresh_token"
    assert mock_post.call_args[1]["data"]["refresh_token"] == "old-refresh"


def test_token_request_http_error(spotify_env):
    resp = MagicMock(status_code=400, text='{"error":"invalid_grant"}')
    with patch("spotify.requests.post", return_value=resp):
        with pytest.raises(spotify.SpotifyError):
            spotify.exchange_code("bad-code")


def test_get_recently_played_mapping():
    payload = {"items": [
        {"played_at": "2026-09-21T21:00:00.000Z",
         "track": {"id": "abc", "name": "Blinding Lights",
                   "artists": [{"name": "The Weeknd"}],
                   "album": {"name": "After Hours"}}},
        {"played_at": "2026-09-21T20:56:00.000Z",
         "track": {"id": "def", "name": "HUMBLE.",
                   "artists": [{"name": "Kendrick Lamar"},
                               {"name": "Featured Artist"}],
                   "album": {"name": "DAMN."}}},
        # malformed entry (no artists) is dropped, not crashed on
        {"played_at": "2026-09-21T20:50:00.000Z",
         "track": {"id": "zzz", "name": "Untitled", "artists": [],
                   "album": {"name": "X"}}},
    ]}
    resp = MagicMock(status_code=200)
    resp.json.return_value = payload
    with patch("spotify.requests.get", return_value=resp) as mock_get:
        tracks = spotify.get_recently_played("tok", limit=50)
    assert mock_get.call_args[0][0].endswith("/me/player/recently-played")
    assert mock_get.call_args[1]["params"] == {"limit": 50}
    assert len(tracks) == 2
    assert tracks[0]["title"] == "Blinding Lights"
    assert tracks[0]["artist"] == "The Weeknd"
    assert tracks[0]["album"] == "After Hours"
    assert tracks[0]["played_at"] == "2026-09-21T21:00:00.000Z"
    assert tracks[1]["artist"] == "Kendrick Lamar, Featured Artist"


# ---- db spotify helpers -----------------------------------------------------

def test_get_valid_access_token_not_connected(tmpdb):
    assert db.get_valid_access_token() is None


def test_get_valid_access_token_unexpired_no_refresh(tmpdb):
    db.save_spotify_auth("tok-now", "ref-now", int(time.time()) + 3600,
                         "u1", "Ani")
    with patch("spotify.refresh_access_token") as mock_refresh:
        assert db.get_valid_access_token() == "tok-now"
    mock_refresh.assert_not_called()


def test_get_valid_access_token_expired_refreshes(tmpdb):
    db.save_spotify_auth("tok-old", "ref-old", int(time.time()) - 10,
                         "u1", "Ani")
    new = {"access_token": "tok-new", "refresh_token": None,
           "expires_in": 3600}
    with patch("spotify.refresh_access_token",
               return_value=new) as mock_refresh:
        assert db.get_valid_access_token() == "tok-new"
    mock_refresh.assert_called_once_with("ref-old")
    auth = db.get_spotify_auth()
    assert auth["access_token"] == "tok-new"
    # response omitted a new refresh token → old one kept
    assert auth["refresh_token"] == "ref-old"
    assert auth["expires_at"] > int(time.time())


def test_get_valid_access_token_refresh_failure_returns_none(tmpdb):
    db.save_spotify_auth("tok-old", "ref-old", int(time.time()) - 10,
                         "u1", "Ani")
    with patch("spotify.refresh_access_token",
               side_effect=spotify.SpotifyError("network down")):
        assert db.get_valid_access_token() is None


def test_import_dedupe_same_played_at_counted_once(tmpdb):
    song = db.log_listen("Blinding Lights", "The Weeknd")
    db.record_spotify_import("2026-09-21T21:00:00.000Z", song["id"])
    assert db.spotify_already_imported("2026-09-21T21:00:00.000Z")
    assert not db.spotify_already_imported("2026-09-21T20:00:00.000Z")
    # a second import pass would skip it, so play_count stays 1
    assert db.top_songs()[0]["play_count"] == 1
    assert db.last_spotify_import_at() == "2026-09-21T21:00:00.000Z"


# ---- app routes --------------------------------------------------------------

def test_connect_without_config_returns_helpful_400(app_client, monkeypatch):
    monkeypatch.delenv("SPOTIFY_CLIENT_ID", raising=False)
    r = app_client.get("/api/spotify/connect")
    assert r.status_code == 400
    body = r.get_json()
    assert "not configured" in body["error"]
    assert "SPOTIFY_CLIENT_ID" in body["setup"]


def test_connect_redirects_with_state(app_client, spotify_env):
    r = app_client.get("/api/spotify/connect")
    assert r.status_code == 302
    loc = r.headers["Location"]
    assert loc.startswith("https://accounts.spotify.com/authorize")
    assert "user-read-recently-played" in urllib.parse.unquote(loc)
    with app_client.session_transaction() as sess:
        state = sess["spotify_oauth_state"]
    assert f"state={state}" in loc


def test_callback_state_mismatch_returns_400(app_client):
    with app_client.session_transaction() as sess:
        sess["spotify_oauth_state"] = "expected-state"
    r = app_client.get("/api/spotify/callback?state=wrong-state&code=abc")
    assert r.status_code == 400
    assert "state" in r.get_json()["error"].lower()


def test_callback_missing_state_returns_400(app_client):
    r = app_client.get("/api/spotify/callback?state=x&code=abc")
    assert r.status_code == 400


def test_callback_success_stores_tokens(app_client, spotify_env):
    with app_client.session_transaction() as sess:
        sess["spotify_oauth_state"] = "good-state"
    with patch("spotify.exchange_code",
               return_value={"access_token": "tok", "refresh_token": "ref",
                             "expires_in": 3600}), \
         patch("spotify.get_profile",
               return_value={"spotify_user_id": "user1",
                             "display_name": "Ani"}):
        r = app_client.get(
            "/api/spotify/callback?state=good-state&code=authcode")
    assert r.status_code == 302
    assert r.headers["Location"].endswith("/?spotify=connected")
    auth = db.get_spotify_auth()
    assert auth["access_token"] == "tok"
    assert auth["refresh_token"] == "ref"
    assert auth["spotify_user_id"] == "user1"
    assert auth["display_name"] == "Ani"


def test_status_and_disconnect(app_client):
    assert app_client.get("/api/spotify/status").get_json() == {
        "connected": False}
    db.save_spotify_auth("t", "r", int(time.time()) + 3600, "u", "Ani Y")
    body = app_client.get("/api/spotify/status").get_json()
    assert body["connected"] is True
    assert body["display_name"] == "Ani Y"
    r = app_client.post("/api/spotify/disconnect")
    assert r.get_json() == {"ok": True, "connected": False}
    assert db.get_spotify_auth() is None


def test_import_route_401_when_not_connected(app_client):
    r = app_client.post("/api/spotify/import")
    assert r.status_code == 401


def test_import_route_dedupes_across_calls(app_client):
    db.save_spotify_auth("tok", "ref", int(time.time()) + 3600, "u1", "Ani")
    tracks = [
        {"spotify_id": "a", "title": "Blinding Lights",
         "artist": "The Weeknd", "album": "After Hours",
         "played_at": "2026-09-21T21:00:00.000Z"},
        {"spotify_id": "b", "title": "HUMBLE.",
         "artist": "Kendrick Lamar", "album": "DAMN.",
         "played_at": "2026-09-21T20:56:00.000Z"},
    ]
    with patch("spotify.get_recently_played", return_value=tracks):
        r1 = app_client.post("/api/spotify/import")
        r2 = app_client.post("/api/spotify/import")  # re-import: all skipped
    assert r1.status_code == 200 and r2.status_code == 200
    b1, b2 = r1.get_json(), r2.get_json()
    assert b1["imported"] == 2 and b1["skipped"] == 0
    assert b2["imported"] == 0 and b2["skipped"] == 2
    assert b1["songs"] == ["Blinding Lights — The Weeknd",
                           "HUMBLE. — Kendrick Lamar"]
    counts = {s["title"]: s["play_count"] for s in db.top_songs()}
    assert counts["Blinding Lights"] == 1  # counted once, not twice
    assert counts["HUMBLE."] == 1
