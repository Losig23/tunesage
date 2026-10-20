"""Tests for the Spotify playback integration. No real HTTP."""

import time
from unittest.mock import MagicMock, patch
from urllib.parse import parse_qs, urlparse

import pytest

import db
import spotify


def _resp(status=200, body=None):
    r = MagicMock(status_code=status)
    r.json.return_value = body or {}
    r.content = b"{}" if body else b""
    return r


def _track(name, artist, uri="spotify:track:abc"):
    return {"name": name, "uri": uri, "artists": [{"name": artist}]}


# ---- OAuth ---------------------------------------------------------------

def test_authorize_url_carries_params():
    url = spotify.authorize_url("cid", "http://127.0.0.1:5001/cb", "st8")
    q = parse_qs(urlparse(url).query)
    assert q["response_type"] == ["code"]
    assert q["client_id"] == ["cid"]
    assert q["state"] == ["st8"]
    assert "user-modify-playback-state" in q["scope"][0]


def test_exchange_code_posts_form_and_parses():
    with patch("spotify.requests.post",
               return_value=_resp(body={"access_token": "at",
                                        "refresh_token": "rt",
                                        "expires_in": 3600})) as mock_post:
        tokens = spotify.exchange_code("cid", "sec", "code1",
                                       "http://127.0.0.1:5001/cb")
    assert tokens["access_token"] == "at"
    data = mock_post.call_args[1]["data"]
    assert data["grant_type"] == "authorization_code"
    assert data["code"] == "code1"
    assert mock_post.call_args[1]["auth"] == ("cid", "sec")


def test_exchange_code_error_raises():
    with patch("spotify.requests.post",
               return_value=_resp(400, {"error": "invalid_grant",
                                        "error_description": "bad code"})):
        with pytest.raises(spotify.SpotifyError, match="bad code"):
            spotify.exchange_code("cid", "sec", "x", "http://127.0.0.1/cb")


# ---- search --------------------------------------------------------------

def _search_body(items):
    return {"tracks": {"items": items}}


def test_search_track_matches_artist_and_title():
    body = _search_body([_track("Borderline (Remix)", "Tame Impala", "u1"),
                         _track("Borderline", "Tame Impala", "u2")])
    with patch("spotify.requests.request", return_value=_resp(body=body)):
        assert spotify.search_track("at", "Borderline", "Tame Impala") == "u2"


def test_search_track_no_confident_match_returns_none():
    body = _search_body([_track("Borderline", "Someone Else")])
    with patch("spotify.requests.request", return_value=_resp(body=body)):
        assert spotify.search_track("at", "Borderline", "Tame Impala") is None


# ---- playback --------------------------------------------------------------

def _api_router(search_items, devices):
    def _route(method, url, **kwargs):
        if "/search" in url:
            return _resp(body=_search_body(search_items))
        if url.endswith("/devices"):
            return _resp(body={"devices": devices})
        return _resp(204)
    return _route


def test_play_recommendation_uses_active_device_without_transfer():
    calls = []

    def _route(method, url, **kwargs):
        calls.append((method, url, kwargs))
        return _api_router([_track("Song", "Artist")],
                           [{"id": "d1", "name": "Phone", "type": "Smartphone",
                             "is_active": True}])(method, url, **kwargs)

    with patch("spotify.requests.request", side_effect=_route):
        assert spotify.play_recommendation("at", "Song", "Artist") == "Phone"
    play = [c for c in calls if c[1].endswith("/play")][0]
    assert play[2]["json"] == {"uris": ["spotify:track:abc"]}
    assert not any(c[1].endswith("/player") for c in calls)  # no transfer


def test_play_recommendation_transfers_to_phone_when_inactive():
    calls = []

    def _route(method, url, **kwargs):
        calls.append((method, url))
        return _api_router([_track("Song", "Artist")],
                           [{"id": "d9", "name": "Ani's iPhone",
                             "type": "Smartphone", "is_active": False}]
                           )(method, url, **kwargs)

    with patch("spotify.requests.request", side_effect=_route):
        name = spotify.play_recommendation("at", "Song", "Artist")
    assert name == "Ani's iPhone"
    assert any(u.endswith("/player") for _, u in calls)  # transfer happened


def test_play_recommendation_no_device_is_a_clear_error():
    with patch("spotify.requests.request",
               side_effect=_api_router([_track("Song", "Artist")], [])):
        with pytest.raises(spotify.SpotifyError, match="open Spotify"):
            spotify.play_recommendation("at", "Song", "Artist")


def test_play_recommendation_no_match_is_a_clear_error():
    with patch("spotify.requests.request", side_effect=_api_router([], [])):
        with pytest.raises(spotify.SpotifyError, match="Couldn't find"):
            spotify.play_recommendation("at", "Nope", "Nobody")


def test_api_403_premium_required_message():
    body = {"error": {"status": 403, "message": "Premium required",
                      "reason": "PREMIUM_REQUIRED"}}
    with patch("spotify.requests.request", return_value=_resp(403, body)):
        with pytest.raises(spotify.SpotifyError, match="Premium"):
            spotify.get_devices("at")


def test_api_401_message_says_reconnect():
    with patch("spotify.requests.request",
               return_value=_resp(401, {"error": {"message": "expired"}})):
        with pytest.raises(spotify.SpotifyError, match="reconnect"):
            spotify.get_devices("at")


# ---- routes ---------------------------------------------------------------

@pytest.fixture
def spotify_env(monkeypatch):
    monkeypatch.setenv("SPOTIFY_CLIENT_ID", "cid")
    monkeypatch.setenv("SPOTIFY_CLIENT_SECRET", "sec")
    monkeypatch.setenv("SPOTIFY_REDIRECT_URI",
                       "http://127.0.0.1:5001/api/spotify/callback")


def test_status_route(app_client, spotify_env):
    body = app_client.get("/api/spotify/status").get_json()
    assert body == {"configured": True, "connected": False}


def test_login_redirects_to_spotify(app_client, spotify_env, tmpdb):
    r = app_client.get("/api/spotify/login")
    assert r.status_code == 302
    assert r.headers["Location"].startswith(
        "https://accounts.spotify.com/authorize?")
    assert db.pop_spotify_state()  # state stored for the callback check


def test_callback_bad_state_goes_home_with_error(app_client, spotify_env):
    r = app_client.get("/api/spotify/callback?code=x&state=wrong")
    assert r.status_code == 302
    assert "spotify=error" in r.headers["Location"]


def test_callback_success_stores_tokens(app_client, spotify_env, tmpdb):
    app_client.get("/api/spotify/login")
    state = db.pop_spotify_state()
    db.save_spotify_state(state)  # login consumed it in a real flow too;
    # here we re-save to simulate the round-trip value Spotify echoes back
    with patch("spotify.requests.post",
               return_value=_resp(body={"access_token": "at",
                                        "refresh_token": "rt",
                                        "expires_in": 3600})):
        r = app_client.get(f"/api/spotify/callback?code=ok&state={state}")
    assert "spotify=connected" in r.headers["Location"]
    tokens = db.get_spotify_tokens()
    assert tokens["access_token"] == "at"
    assert tokens["refresh_token"] == "rt"


def test_play_route_plays_via_stored_token(app_client, spotify_env, tmpdb):
    db.save_spotify_tokens("at", "rt", int(time.time()) + 3600)
    with patch("spotify.requests.request",
               side_effect=_api_router(
                   [_track("Song", "Artist")],
                   [{"id": "d1", "name": "Phone", "type": "Smartphone",
                     "is_active": True}])):
        body = app_client.post("/api/spotify/play",
                               json={"title": "Song",
                                     "artist": "Artist"}).get_json()
    assert body["ok"] is True
    assert body["device"] == "Phone"


def test_play_route_refreshes_expired_token(app_client, spotify_env, tmpdb):
    db.save_spotify_tokens("old", "rt", int(time.time()) - 10)  # expired
    with patch("spotify.requests.post",
               return_value=_resp(body={"access_token": "new",
                                        "expires_in": 3600})) as mock_post, \
         patch("spotify.requests.request",
               side_effect=_api_router(
                   [_track("Song", "Artist")],
                   [{"id": "d1", "name": "Phone", "type": "Smartphone",
                     "is_active": True}])) as mock_req:
        body = app_client.post("/api/spotify/play",
                               json={"title": "Song",
                                     "artist": "Artist"}).get_json()
    assert body["ok"] is True
    assert mock_post.call_args[1]["data"]["grant_type"] == "refresh_token"
    auth = mock_req.call_args_list[0][1]["headers"]["Authorization"]
    assert auth == "Bearer new"  # refreshed token is the one used
    assert db.get_spotify_tokens()["access_token"] == "new"


def test_disconnect_clears_tokens(app_client, spotify_env, tmpdb):
    db.save_spotify_tokens("at", "rt", int(time.time()) + 3600)
    app_client.post("/api/spotify/disconnect")
    assert db.get_spotify_tokens() is None
