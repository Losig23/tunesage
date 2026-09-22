"""Tests for the Last.fm integration.

No real HTTP anywhere: every network call is mocked. DB tests run against a
throwaway SQLite file — the fixture rebinds the db helpers' `path` default to
it, so the Flask routes under test hit the temp DB, not tunesage.db.
"""

import functools
from unittest.mock import MagicMock, patch

import pytest

import db
import lastfm


@pytest.fixture
def tmpdb(tmp_path, monkeypatch):
    """Temp DB + rebind db helpers used by the routes to use it."""
    p = tmp_path / "lastfm_test.db"
    db.init_db(p)

    def _bind(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            kwargs.setdefault("path", p)
            return fn(*args, **kwargs)
        return wrapper

    for name in ("log_listen", "top_songs", "already_imported",
                 "record_import", "last_import_at"):
        monkeypatch.setattr(db, name, _bind(getattr(db, name)))
    return p


@pytest.fixture
def app_client(tmpdb, monkeypatch):
    monkeypatch.setenv("SECRET_KEY", "test-secret-key")
    import app as app_module  # imported after SECRET_KEY is set
    return app_module.app.test_client()


@pytest.fixture
def lastfm_env(monkeypatch):
    monkeypatch.setenv("LASTFM_API_KEY", "key123")
    monkeypatch.setenv("LASTFM_USERNAME", "testuser")


def _recent_tracks_payload(*tracks):
    return {"recenttracks": {
        "track": tracks,
        "@attr": {"user": "testuser", "page": "1", "totalPages": "1"},
    }}


def _track(name, artist, uts, album="Some Album"):
    return {"name": name,
            "artist": {"#text": artist, "mbid": ""},
            "album": {"#text": album},
            "date": {"uts": str(uts), "#text": "21 Sep 2026, 21:00"}}


# ---- lastfm.py: parsing ------------------------------------------------------

def test_get_recent_tracks_parses_and_skips_now_playing():
    payload = _recent_tracks_payload(
        _track("Blinding Lights", "The Weeknd", 1780000000),
        _track("HUMBLE.", "Kendrick Lamar", 1779999960),
        # now-playing entry: no "date" → skipped, not crashed on
        {"name": "Currently Playing", "artist": {"#text": "Someone"},
         "album": {"#text": "X"}, "@attr": {"nowplaying": "true"}},
        # malformed entry (no artist text) → skipped
        {"name": "Untitled", "artist": {"#text": ""},
         "album": {"#text": "X"},
         "date": {"uts": "1779999900", "#text": "x"}},
    )
    resp = MagicMock(status_code=200)
    resp.json.return_value = payload
    with patch("lastfm.requests.get", return_value=resp) as mock_get:
        tracks = lastfm.get_recent_tracks("key123", "testuser", limit=50)
    assert mock_get.call_args[0][0] == "https://ws.audioscrobbler.com/2.0/"
    params = mock_get.call_args[1]["params"]
    assert params["method"] == "user.getrecenttracks"
    assert params["user"] == "testuser"
    assert params["limit"] == 50
    assert params["api_key"] == "key123"
    assert len(tracks) == 2
    assert tracks[0]["title"] == "Blinding Lights"
    assert tracks[0]["artist"] == "The Weeknd"
    assert tracks[0]["album"] == "Some Album"
    assert tracks[0]["played_at"] == "2026-05-28T20:26:40Z"
    assert tracks[1]["title"] == "HUMBLE."


def test_get_recent_tracks_single_object_response():
    """One scrobble comes back as an object, not a list."""
    payload = {"recenttracks": {
        "track": _track("Solo", "Artist", 1780000000)}}
    resp = MagicMock(status_code=200)
    resp.json.return_value = payload
    with patch("lastfm.requests.get", return_value=resp):
        tracks = lastfm.get_recent_tracks("k", "u")
    assert len(tracks) == 1 and tracks[0]["title"] == "Solo"


def test_get_recent_tracks_error_body_raises():
    """Last.fm reports errors as HTTP 200 with {error, message}."""
    resp = MagicMock(status_code=200)
    resp.json.return_value = {"error": 6, "message": "User not found"}
    with patch("lastfm.requests.get", return_value=resp):
        with pytest.raises(lastfm.LastfmError, match="User not found"):
            lastfm.get_recent_tracks("k", "nosuchuser")


def test_get_recent_tracks_bad_api_key_error():
    resp = MagicMock(status_code=200)
    resp.json.return_value = {"error": 10,
                             "message": "Invalid API key - You must be "
                                        "granted a valid key by last.fm"}
    with patch("lastfm.requests.get", return_value=resp):
        with pytest.raises(lastfm.LastfmError, match="Invalid API key"):
            lastfm.get_recent_tracks("badkey", "u")


def test_error_messages_never_contain_api_key():
    resp = MagicMock(status_code=200)
    resp.json.return_value = {"error": 10, "message": "Invalid API key"}
    with patch("lastfm.requests.get", return_value=resp):
        try:
            lastfm.get_recent_tracks("supersecretkey", "u")
            pytest.fail("expected LastfmError")
        except lastfm.LastfmError as e:
            assert "supersecretkey" not in str(e)


def test_get_recent_tracks_http_error():
    resp = MagicMock(status_code=503, text="Service Unavailable")
    with patch("lastfm.requests.get", return_value=resp):
        with pytest.raises(lastfm.LastfmError, match="HTTP 503"):
            lastfm.get_recent_tracks("k", "u")


def test_get_recent_tracks_missing_config():
    with pytest.raises(lastfm.LastfmError, match="LASTFM_API_KEY"):
        lastfm.get_recent_tracks("", "u")


def test_get_all_recent_tracks_paginates_until_short_page(monkeypatch):
    """Full page → fetch next; short page → stop (no more history)."""
    page1 = [_track(f"Song {i}", "Artist", 1780000000 - i) for i in range(3)]
    page2 = [_track("Last Song", "Artist", 1779900000)]

    def fake_get(api_key, username, limit=50, page=1):
        assert limit == 3  # per_page passed through
        raw = page1 if page == 1 else page2
        return [t for t in (lastfm._parse_track(i) for i in raw) if t]

    monkeypatch.setattr(lastfm, "get_recent_tracks", fake_get)
    monkeypatch.setattr(lastfm.time, "sleep", lambda s: None)
    tracks = lastfm.get_all_recent_tracks("k", "u", total_limit=200,
                                          per_page=3)
    assert [t["title"] for t in tracks] == \
        ["Song 0", "Song 1", "Song 2", "Last Song"]


def test_get_all_recent_tracks_respects_total_limit(monkeypatch):
    batch = [_track(f"Song {i}", "Artist", 1780000000 - i) for i in range(5)]
    calls = []

    def fake_get(api_key, username, limit=50, page=1):
        calls.append(page)
        return list(batch)

    monkeypatch.setattr(lastfm, "get_recent_tracks", fake_get)
    monkeypatch.setattr(lastfm.time, "sleep", lambda s: None)
    tracks = lastfm.get_all_recent_tracks("k", "u", total_limit=7,
                                          per_page=5)
    assert len(tracks) == 7
    assert calls == [1, 2]  # stopped once the limit was reached


# ---- db import helpers ---------------------------------------------------------

def test_import_dedupe_same_scrobble_counted_once(tmpdb):
    song = db.log_listen("Blinding Lights", "The Weeknd")
    db.record_import("lastfm", "2026-09-21T21:00:00Z", song["id"])
    assert db.already_imported("lastfm", "2026-09-21T21:00:00Z")
    assert not db.already_imported("lastfm", "2026-09-21T20:00:00Z")
    # a different source with the same timestamp is a different play
    assert not db.already_imported("other", "2026-09-21T21:00:00Z")
    # a second import pass would skip it, so play_count stays 1
    assert db.top_songs()[0]["play_count"] == 1
    assert db.last_import_at("lastfm") == "2026-09-21T21:00:00Z"
    assert db.last_import_at("other") is None


# ---- app routes ------------------------------------------------------------------

def test_status_unconfigured_returns_helpful_json(app_client, monkeypatch):
    monkeypatch.delenv("LASTFM_API_KEY", raising=False)
    monkeypatch.delenv("LASTFM_USERNAME", raising=False)
    r = app_client.get("/api/lastfm/status")
    assert r.status_code == 200
    body = r.get_json()
    assert body["configured"] is False
    assert "LASTFM_API_KEY" in body["setup"]


def test_status_configured(app_client, lastfm_env):
    body = app_client.get("/api/lastfm/status").get_json()
    assert body["configured"] is True
    assert body["username"] == "testuser"
    assert body["last_import_at"] is None


def test_import_unconfigured_returns_clean_400(app_client, monkeypatch):
    monkeypatch.delenv("LASTFM_API_KEY", raising=False)
    r = app_client.post("/api/lastfm/import")
    assert r.status_code == 400
    body = r.get_json()
    assert "not configured" in body["error"]
    assert "LASTFM_USERNAME" in body["setup"]


def test_import_route_dedupes_across_calls(app_client, lastfm_env):
    tracks = [
        {"title": "Blinding Lights", "artist": "The Weeknd",
         "album": "After Hours", "played_at": "2026-09-21T21:00:00Z"},
        {"title": "HUMBLE.", "artist": "Kendrick Lamar",
         "album": "DAMN.", "played_at": "2026-09-21T20:56:00Z"},
    ]
    with patch("lastfm.get_all_recent_tracks", return_value=tracks):
        r1 = app_client.post("/api/lastfm/import")
        r2 = app_client.post("/api/lastfm/import")  # re-import: all skipped
    assert r1.status_code == 200 and r2.status_code == 200
    b1, b2 = r1.get_json(), r2.get_json()
    assert b1["imported"] == 2 and b1["skipped"] == 0
    assert b2["imported"] == 0 and b2["skipped"] == 2
    assert b1["songs"] == ["Blinding Lights — The Weeknd",
                           "HUMBLE. — Kendrick Lamar"]
    counts = {s["title"]: s["play_count"] for s in db.top_songs()}
    assert counts["Blinding Lights"] == 1  # counted once, not twice
    assert counts["HUMBLE."] == 1


def test_import_route_lastfm_error_returns_502(app_client, lastfm_env):
    with patch("lastfm.get_all_recent_tracks",
               side_effect=lastfm.LastfmError("Last.fm error 6: "
                                              "User not found")):
        r = app_client.post("/api/lastfm/import")
    assert r.status_code == 502
    assert "User not found" in r.get_json()["error"]


def test_status_reflects_last_import(app_client, lastfm_env):
    tracks = [{"title": "Song A", "artist": "Artist A", "album": "",
               "played_at": "2026-09-21T21:00:00Z"}]
    with patch("lastfm.get_all_recent_tracks", return_value=tracks):
        app_client.post("/api/lastfm/import")
    body = app_client.get("/api/lastfm/status").get_json()
    assert body["last_import_at"] == "2026-09-21T21:00:00Z"
