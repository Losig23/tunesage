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


@pytest.fixture(autouse=True)
def _stub_itunes_artwork(monkeypatch):
    """mine_candidates falls back to iTunes when Last.fm art is missing;
    stub it so mining unit tests never touch the network."""
    import artwork
    monkeypatch.setattr(artwork, "fetch_itunes_artwork",
                        lambda title, artist: "")


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
                 "record_import", "last_import_at",
                 "get_cached_recommendations", "save_recommendations",
                 "all_song_keys"):
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


# ---- lastfm.py: candidate mining ---------------------------------------------

def _similar_payload(*artists):
    return {"similarartists": {"artist": [
        {"name": name, "match": str(match)} for name, match in artists]}}


def _toptracks_payload(artist, *tracks):
    return {"toptracks": {"track": [
        {"name": title, "playcount": str(pc), "listeners": str(listeners),
         "artist": {"name": artist}}
        for title, listeners, pc in tracks]}}


def _mock_get(payloads):
    """requests.get mock serving one payload per call, in order."""
    resps = []
    for p in payloads:
        r = MagicMock(status_code=200)
        r.json.return_value = p
        resps.append(r)
    return MagicMock(side_effect=resps)


def test_get_similar_artists_parses_match_scores():
    payload = _similar_payload(("Tame Impala", 0.92), ("MGMT", 0.71),
                               ("", 0.5))  # blank name skipped
    resp = MagicMock(status_code=200)
    resp.json.return_value = payload
    with patch("lastfm.requests.get", return_value=resp):
        sims = lastfm.get_similar_artists("key123", "Arctic Monkeys", limit=5)
    assert sims == [{"name": "Tame Impala", "match": 0.92},
                    {"name": "MGMT", "match": 0.71}]


def test_get_artist_top_tracks_parses_listeners():
    payload = _toptracks_payload("Tame Impala",
                                 ("The Less I Know The Better", 2500000, 180000000),
                                 ("Borderline", 900000, 60000000))
    resp = MagicMock(status_code=200)
    resp.json.return_value = payload
    with patch("lastfm.requests.get", return_value=resp):
        tracks = lastfm.get_artist_top_tracks("key123", "Tame Impala", limit=2)
    assert tracks[0]["title"] == "The Less I Know The Better"
    assert tracks[0]["listeners"] == 2500000
    assert tracks[0]["artist"] == "Tame Impala"


def test_mentions_bucketing_keeps_scale_sane():
    assert lastfm._mentions_from_listeners(10_000_000) == 7
    assert lastfm._mentions_from_listeners(100_000) == 5
    assert lastfm._mentions_from_listeners(1_000) == 3
    assert lastfm._mentions_from_listeners(0) == 1  # never zero/negative


def test_mine_candidates_builds_ranked_unique_candidates():
    similar = _similar_payload(("Tame Impala", 0.9), ("MGMT", 0.7))
    tame_tracks = _toptracks_payload(
        "Tame Impala", ("The Less I Know The Better", 2500000, 1),
        ("Borderline", 900000, 1))
    mgmt_tracks = _toptracks_payload(
        "MGMT", ("Kids", 1800000, 1),
        # same title+artist as Tame Impala's track above -> deduped
        ("Borderline", 5000, 1))
    mgmt_tracks["toptracks"]["track"][1]["artist"] = {"name": "Tame Impala"}
    mock = _mock_get([similar, tame_tracks, mgmt_tracks])
    with patch("lastfm.requests.get", mock), \
         patch("lastfm.time.sleep"):  # don't actually sleep in tests
        mined = lastfm.mine_candidates("key123", ["Arctic Monkeys"],
                                       max_artists=1, similar_per_artist=2,
                                       tracks_per_artist=2, hop2_seeds=0)
    titles = [c["title"] for c in mined["candidates"]]
    assert titles.count("Borderline") == 1  # deduped, kept max listeners
    assert mined["candidates"][0]["mentions"] >= \
        mined["candidates"][-1]["mentions"]  # sorted desc
    assert all(c["mentions"] >= 1 for c in mined["candidates"])
    assert mined["sources"]  # provenance entries for the UI
    assert all("Last.fm" in s["title"] for s in mined["sources"])


def test_mine_candidates_requires_api_key():
    with pytest.raises(lastfm.LastfmError):
        lastfm.mine_candidates("", ["Arctic Monkeys"])


def test_mine_candidates_raises_when_nothing_found():
    empty = {"similarartists": {"artist": []}}
    with patch("lastfm.requests.get", _mock_get([empty])), \
         patch("lastfm.time.sleep"):
        with pytest.raises(RuntimeError, match="no candidates"):
            lastfm.mine_candidates("key123", ["Nobody Ever Heard Of Them"])


# ---- album artwork -----------------------------------------------------------

def _track_with_images(name, artist, uts):
    t = _track(name, artist, uts)
    t["image"] = [
        {"#text": "http://img/small.png", "size": "small"},
        {"#text": "http://img/xl.png", "size": "extralarge"},
        {"#text": "http://img/medium.png", "size": "medium"},
    ]
    return t


def test_parse_track_picks_largest_image():
    t = lastfm._parse_track(_track_with_images("A", "B", 1780000000))
    assert t["image"] == "http://img/xl.png"


def test_parse_track_no_images_gives_blank():
    t = lastfm._parse_track(_track("A", "B", 1780000000))
    assert t["image"] == ""


def test_top_tracks_extracts_image_when_present():
    payload = {"toptracks": {"track": [
        {"name": "Borderline", "listeners": "900000", "playcount": "1",
         "artist": {"name": "Tame Impala"},
         "image": [{"#text": "http://img/s.png", "size": "small"},
                   {"#text": "http://img/l.png", "size": "large"}]}]}}
    resp = MagicMock(status_code=200)
    resp.json.return_value = payload
    with patch("lastfm.requests.get", return_value=resp):
        tracks = lastfm.get_artist_top_tracks("key123", "Tame Impala")
    assert tracks[0]["image"] == "http://img/l.png"


def test_top_tracks_missing_image_gives_blank():
    payload = {"toptracks": {"track": [
        {"name": "Borderline", "listeners": "900000", "playcount": "1",
         "artist": {"name": "Tame Impala"}}]}}
    resp = MagicMock(status_code=200)
    resp.json.return_value = payload
    with patch("lastfm.requests.get", return_value=resp):
        tracks = lastfm.get_artist_top_tracks("key123", "Tame Impala")
    assert tracks[0]["image"] == ""


def test_mine_candidates_carries_image_through():
    similar = _similar_payload(("Tame Impala", 0.9))
    tracks = {"toptracks": {"track": [
        {"name": "Borderline", "listeners": "900000", "playcount": "1",
         "artist": {"name": "Tame Impala"},
         "image": [{"#text": "http://img/xl.png", "size": "extralarge"}]}]}}
    with patch("lastfm.requests.get", _mock_get([similar, tracks])), \
         patch("lastfm.time.sleep"):
        mined = lastfm.mine_candidates("key123", ["Arctic Monkeys"],
                                       max_artists=1, similar_per_artist=1,
                                       tracks_per_artist=1, hop2_seeds=0)
    assert mined["candidates"][0]["image"] == "http://img/xl.png"


def test_log_listen_stores_and_preserves_image(tmpdb):
    row = db.log_listen("A", "B", image_url="http://img/xl.png")
    assert row["image_url"] == "http://img/xl.png"
    # a later manual log with no art must not blank the stored art
    row = db.log_listen("A", "B")
    assert row["image_url"] == "http://img/xl.png"
    # ...but a newer non-empty URL wins
    row = db.log_listen("A", "B", image_url="http://img/new.png")
    assert row["image_url"] == "http://img/new.png"


def test_migrate_adds_image_url_to_legacy_db(tmp_path):
    import sqlite3
    p = tmp_path / "legacy.db"
    conn = sqlite3.connect(p)
    # realistic pre-migration schema: every original column, no image_url
    conn.execute("""CREATE TABLE songs (id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL, artist TEXT NOT NULL, genre TEXT DEFAULT '',
        tags TEXT DEFAULT '[]', play_count INTEGER NOT NULL DEFAULT 0,
        updated_at INTEGER NOT NULL, UNIQUE(title, artist))""")
    conn.commit()
    conn.close()
    db.init_db(p)  # CREATE IF NOT EXISTS + _migrate
    cols = {r[1] for r in sqlite3.connect(p).execute(
        "PRAGMA table_info(songs)")}
    assert "image_url" in cols
    row = db.log_listen("X", "Y", image_url="http://img", path=p)
    assert row["image_url"] == "http://img"


def test_import_route_stores_artwork(app_client, lastfm_env):
    tracks = [{"title": "Blinding Lights", "artist": "The Weeknd",
               "album": "After Hours", "played_at": "2026-09-21T21:00:00Z",
               "image": "http://img/xl.png"}]
    with patch("lastfm.get_all_recent_tracks", return_value=tracks):
        r = app_client.post("/api/lastfm/import")
    assert r.status_code == 200
    assert r.get_json()["imported"] == 1
    assert db.top_songs()[0]["image_url"] == "http://img/xl.png"


def test_mine_candidates_carries_listeners_and_via():
    similar = _similar_payload(("Tame Impala", 0.9))
    tracks = {"toptracks": {"track": [
        {"name": "Borderline", "listeners": "900000", "playcount": "1",
         "artist": {"name": "Tame Impala"}}]}}
    with patch("lastfm.requests.get", _mock_get([similar, tracks])), \
         patch("lastfm.time.sleep"):
        mined = lastfm.mine_candidates("key123", ["Arctic Monkeys"],
                                       max_artists=1, similar_per_artist=1,
                                       tracks_per_artist=1, hop2_seeds=0)
    cand = mined["candidates"][0]
    assert cand["listeners"] == 900000
    assert cand["via"] == "Arctic Monkeys"


def test_mine_candidates_hop2_discovers_deeper_artists():
    sim_am = _similar_payload(("Tame Impala", 0.9), ("MGMT", 0.7))
    sim_tame = _similar_payload(("Pond", 0.8), ("Unknown Mortal Orchestra", 0.6))
    def _tt(artist, title):
        return _toptracks_payload(artist, (title, 50000, 1))
    # call order: hop1 similar, hop2 similar, then top tracks per artist
    mock = _mock_get([sim_am, sim_tame,
                      _tt("Tame Impala", "Borderline"),
                      _tt("MGMT", "Kids"),
                      _tt("Pond", "Paint Me Silver"),
                      _tt("Unknown Mortal Orchestra", "Multi-Love")])
    with patch("lastfm.requests.get", mock), \
         patch("lastfm.time.sleep"):
        mined = lastfm.mine_candidates(
            "key123", ["Arctic Monkeys"],
            max_artists=1, similar_per_artist=2, tracks_per_artist=1,
            hop2_seeds=1, hop2_similar=2)
    by_title = {c["title"]: c for c in mined["candidates"]}
    # hop-2 artist surfaced with provenance chain
    assert by_title["Paint Me Silver"]["via"] == "Arctic Monkeys → Tame Impala"
    assert by_title["Borderline"]["via"] == "Arctic Monkeys"
    assert len(mined["candidates"]) == 4


def test_mine_candidates_hop2_failure_does_not_kill_mine():
    sim_resp = MagicMock(status_code=200)
    sim_resp.json.return_value = _similar_payload(("Tame Impala", 0.9))
    err_resp = MagicMock(status_code=500)
    err_resp.text = "server exploded"
    tracks_resp = MagicMock(status_code=200)
    tracks_resp.json.return_value = _toptracks_payload(
        "Tame Impala", ("Borderline", 50000, 1))
    mock = MagicMock(side_effect=[sim_resp, err_resp, tracks_resp])
    with patch("lastfm.requests.get", mock), \
         patch("lastfm.time.sleep"):
        mined = lastfm.mine_candidates(
            "key123", ["Arctic Monkeys"],
            max_artists=1, similar_per_artist=1, tracks_per_artist=1,
            hop2_seeds=1, hop2_similar=1)
    # hop-1 results survive the failed hop-2 call
    assert [c["title"] for c in mined["candidates"]] == ["Borderline"]


def _err_resp():
    r = MagicMock(status_code=503)
    r.text = "upstream connect error"
    return r


def _ok_resp(payload):
    r = MagicMock(status_code=200)
    r.json.return_value = payload
    return r


def test_mine_candidates_skips_seed_whose_similar_call_fails():
    mock = MagicMock(side_effect=[
        _err_resp(),  # Arctic Monkeys hop1 -> 503, skipped
        _ok_resp(_similar_payload(("Pond", 0.8))),
        _ok_resp(_toptracks_payload("Pond", ("Paint Me Silver", 50000, 1))),
    ])
    with patch("lastfm.requests.get", mock), \
         patch("lastfm.time.sleep"):
        mined = lastfm.mine_candidates(
            "key123", ["Arctic Monkeys", "Tame Impala"],
            max_artists=2, similar_per_artist=1, tracks_per_artist=1,
            hop2_seeds=0)
    assert [c["title"] for c in mined["candidates"]] == ["Paint Me Silver"]
    assert mined["candidates"][0]["via"] == "Tame Impala"


def test_mine_candidates_skips_artist_whose_tracks_call_fails():
    mock = MagicMock(side_effect=[
        _ok_resp(_similar_payload(("Tame Impala", 0.9), ("MGMT", 0.7))),
        _err_resp(),  # Tame Impala top tracks -> 503, skipped
        _ok_resp(_toptracks_payload("MGMT", ("Kids", 60000, 1))),
    ])
    with patch("lastfm.requests.get", mock), \
         patch("lastfm.time.sleep"):
        mined = lastfm.mine_candidates(
            "key123", ["Arctic Monkeys"],
            max_artists=1, similar_per_artist=2, tracks_per_artist=1,
            hop2_seeds=0)
    assert [c["title"] for c in mined["candidates"]] == ["Kids"]


def test_mine_candidates_raises_when_everything_fails():
    import pytest as _pytest
    mock = MagicMock(side_effect=[_err_resp()])
    with patch("lastfm.requests.get", mock), \
         patch("lastfm.time.sleep"):
        with _pytest.raises(RuntimeError, match="no candidates"):
            lastfm.mine_candidates(
                "key123", ["Arctic Monkeys"],
                max_artists=1, similar_per_artist=1, tracks_per_artist=1,
                hop2_seeds=0)


def test_largest_image_skips_lastfm_default_placeholder():
    item = {"image": [
        {"#text": "https://lastfm.freetls.fastly.net/i/u/300x300/"
                  "2a96cbd8b46e442fc41c2b86b821b4ce.png", "size": "extralarge"},
        {"#text": "https://example.com/real.png", "size": "small"},
    ]}
    # default placeholder ignored even though it's the biggest entry
    assert lastfm._largest_image(item) == "https://example.com/real.png"
    only_default = {"image": [
        {"#text": "https://lastfm.freetls.fastly.net/i/u/300x300/"
                  "2a96cbd8b46e442fc41c2b86b821b4ce.png", "size": "large"}]}
    assert lastfm._largest_image(only_default) == ""


def test_blank_default_artwork_shared_across_artists():
    import artwork as artwork_mod
    cands = [
        {"title": "A", "artist": "Art1", "image": "https://x/star.png"},
        {"title": "B", "artist": "Art2", "image": "https://x/star.png"},
        {"title": "C", "artist": "Art3", "image": "https://x/star.png"},
        {"title": "D", "artist": "Art4", "image": "https://x/real.png"},
    ]
    with patch.object(artwork_mod, "fetch_itunes_artwork",
                      return_value="https://itunes/d.jpg"):
        lastfm._blank_default_artwork(cands)
    assert [c["image"] for c in cands[:3]] == ["https://itunes/d.jpg"] * 3
    assert cands[3]["image"] == "https://x/real.png"


def test_blank_default_artwork_keeps_shared_album_art():
    # same artist, same image (one album) -> not the default placeholder
    cands = [
        {"title": "A", "artist": "Art1", "image": "https://x/album.png"},
        {"title": "B", "artist": "Art1", "image": "https://x/album.png"},
    ]
    lastfm._blank_default_artwork(cands)
    assert all(c["image"] == "https://x/album.png" for c in cands)
