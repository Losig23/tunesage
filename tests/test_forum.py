"""Tests for forum.mine_recommendations orchestration.

The live Last.fm HTTP layer is tested in test_lastfm.py; here we check the
routing: live path when a key is set, graceful fallback otherwise.
"""

from unittest.mock import patch

import pytest

import forum


@pytest.fixture(autouse=True)
def _stub_itunes_artwork(monkeypatch):
    """Route tests exercise mine_candidates; keep artwork lookups off-net."""
    import artwork
    import lastfm
    monkeypatch.setattr(artwork, "fetch_itunes_artwork",
                        lambda title, artist: "")
    monkeypatch.setattr(lastfm, "get_track_album_art",
                        lambda api_key, title, artist: "")


def _seeds():
    return [{"title": "505", "artist": "Arctic Monkeys", "play_count": 10},
            {"title": "Blinding Lights", "artist": "The Weeknd",
             "play_count": 5}]


def test_falls_back_without_api_key():
    mined = forum.mine_recommendations(_seeds(), api_key="")
    assert mined["source"] == "fallback"
    assert len(mined["candidates"]) == len(forum.FALLBACK_CANDIDATES)
    assert mined["threads"]  # UI still has something to render


def test_live_path_uses_lastfm_when_key_set():
    fake = {"candidates": [{"title": "Kids", "artist": "MGMT", "genre": "",
                            "tags": [], "mentions": 6, "sources": ["u"],
                            "snippets": ["s"]}],
            "sources": [{"title": "Last.fm: similar to Arctic Monkeys",
                         "url": "https://www.last.fm/music/MGMT"}]}
    with patch("forum.lastfm_client.mine_candidates",
               return_value=fake) as mock_mine:
        mined = forum.mine_recommendations(_seeds(), api_key="key123")
    assert mined["source"] == "lastfm"
    assert mined["candidates"] == fake["candidates"]
    # seed artists passed through (deduped, in order)
    assert mock_mine.call_args[0][1] == ["Arctic Monkeys", "The Weeknd"]
    # threads carry provenance for the UI
    assert mined["threads"][0]["url"].startswith("https://www.last.fm")


def test_falls_back_when_lastfm_raises():
    with patch("forum.lastfm_client.mine_candidates",
               side_effect=RuntimeError("boom")):
        mined = forum.mine_recommendations(_seeds(), api_key="key123")
    assert mined["source"] == "fallback"
    assert mined["candidates"]


def test_never_raises_on_empty_seeds():
    # app.py guards this, but the contract is "never raises"
    mined = forum.mine_recommendations([], api_key="key123")
    assert mined["source"] == "fallback"


# ---- end-to-end wiring: /api/recommendations with live mining ---------------

from test_lastfm import tmpdb, app_client  # noqa: F401  (reuse fixtures)
import db as db_module
from unittest.mock import MagicMock


def _fake_lastfm_get(url, params=None, **kwargs):
    method = (params or {}).get("method")
    resp = MagicMock(status_code=200)
    if method == "artist.getSimilar":
        resp.json.return_value = {"similarartists": {"artist": [
            {"name": "Tame Impala", "match": "0.90"}]}}
    elif method == "artist.getTopTracks":
        resp.json.return_value = {"toptracks": {"track": [
            {"name": "Borderline", "listeners": "900000", "playcount": "1",
             "artist": {"name": "Tame Impala"}}]}}
    else:
        raise AssertionError(f"unexpected Last.fm method: {method}")
    return resp


def test_recommendations_endpoint_mines_lastfm_live(app_client, monkeypatch):
    monkeypatch.setenv("LASTFM_API_KEY", "key123")
    db_module.log_listen("505", "Arctic Monkeys")
    db_module.log_listen("Blinding Lights", "The Weeknd")
    with patch("lastfm.requests.get", _fake_lastfm_get), \
         patch("lastfm.time.sleep"):
        r = app_client.get("/api/recommendations?limit=5&refresh=1")
    assert r.status_code == 200
    body = r.get_json()
    assert body["source"] == "lastfm"
    titles = [s["title"] for s in body["songs"]]
    assert "Borderline" in titles
    assert all(s["why"] for s in body["songs"])  # explanations attached


def test_recommendations_endpoint_falls_back_without_key(app_client,
                                                         monkeypatch):
    monkeypatch.delenv("LASTFM_API_KEY", raising=False)
    db_module.log_listen("505", "Arctic Monkeys")
    r = app_client.get("/api/recommendations?limit=5&refresh=1")
    assert r.status_code == 200
    assert r.get_json()["source"] == "fallback"


def test_recommendations_payload_includes_streams(app_client, monkeypatch):
    monkeypatch.setenv("LASTFM_API_KEY", "key123")
    db_module.log_listen("505", "Arctic Monkeys")
    with patch("lastfm.requests.get", _fake_lastfm_get), \
         patch("lastfm.time.sleep"):
        r = app_client.get("/api/recommendations?limit=5&refresh=1")
    body = r.get_json()
    assert set(body["streams"]) == {"mainstream", "niche", "underground"}
    total = sum(len(v) for v in body["streams"].values())
    assert total == len(body["songs"])  # every song lands in exactly one tier


def test_recommendations_tempo_unavailable_without_key(app_client, monkeypatch):
    monkeypatch.delenv("GETSONGBPM_API_KEY", raising=False)
    monkeypatch.setenv("LASTFM_API_KEY", "key123")
    db_module.log_listen("505", "Arctic Monkeys")
    with patch("lastfm.requests.get", _fake_lastfm_get), \
         patch("lastfm.time.sleep"):
        body = app_client.get("/api/recommendations?limit=5&refresh=1").get_json()
    assert body["tempo_available"] is False
    assert "upbeat" not in body["streams"]


def test_recommendations_tempo_streams_with_key(app_client, monkeypatch):
    monkeypatch.setenv("GETSONGBPM_API_KEY", "bpmkey")
    monkeypatch.setenv("LASTFM_API_KEY", "key123")
    db_module.log_listen("505", "Arctic Monkeys")
    fake_tempos = {"Borderline": 140.0}

    def _fake_fill(api_key, cands, sleep=0.2):
        for c in cands:
            if c["title"] in fake_tempos:
                c["tempo"] = fake_tempos[c["title"]]
        return len(fake_tempos)

    with patch("lastfm.requests.get", _fake_lastfm_get), \
         patch("lastfm.time.sleep"), \
         patch("bpm.fill_tempos", side_effect=_fake_fill):
        body = app_client.get("/api/recommendations?limit=5&refresh=1").get_json()
    assert body["tempo_available"] is True
    assert [s["title"] for s in body["streams"]["upbeat"]] == ["Borderline"]
    assert body["streams"]["slowburn"] == []


def test_fallback_carries_error_cause():
    mined = forum.mine_recommendations(_seeds(), api_key="")
    assert mined["error"] == "LASTFM_API_KEY is not set"
    with patch("forum.lastfm_client.mine_candidates",
               side_effect=RuntimeError("boom")):
        mined = forum.mine_recommendations(_seeds(), api_key="<redacted>")
    assert mined["source"] == "fallback"
    assert mined["error"] == "RuntimeError: boom"


def test_live_path_error_is_none():
    fake = {"candidates": [], "sources": []}
    with patch("forum.lastfm_client.mine_candidates", return_value=fake):
        mined = forum.mine_recommendations(_seeds(), api_key="<redacted>")
    assert mined["source"] == "lastfm"
    assert mined["error"] is None


def test_empty_seeds_route_reports_empty_source(app_client):
    # no listens logged in this fixture's db -> distinct "empty" source
    # (not "fallback"), so the UI doesn't blame mining.
    body = app_client.get("/api/recommendations?limit=5").get_json()
    assert body["source"] == "empty"
    assert body["songs"] == []


def test_recommendations_exclude_entire_history_not_just_top10(
        app_client, monkeypatch):
    monkeypatch.setenv("LASTFM_API_KEY", "key123")
    # 10 songs with 3 plays each fill the top-10; Borderline (1 play) sits
    # at #11 — the old top-10-only exclusion would still recommend it.
    for i in range(10):
        for _ in range(3):
            db_module.log_listen(f"Song{i}", f"Artist{i}")
    db_module.log_listen("Borderline", "Tame Impala")
    with patch("lastfm.requests.get", _fake_lastfm_get), \
         patch("lastfm.time.sleep"):
        body = app_client.get(
            "/api/recommendations?limit=10&refresh=1").get_json()
    titles = [s["title"] for s in body["songs"]]
    assert "Borderline" not in titles


def test_all_song_keys_roundtrip(tmpdb):
    db_module.log_listen("  Blinding Lights ", "THE WEEKND")
    assert db_module.all_song_keys() == {("blinding lights", "the weeknd")}


def test_tempo_available_only_with_real_data(app_client, monkeypatch):
    monkeypatch.setenv("GETSONGBPM_API_KEY", "bpmkey")
    monkeypatch.setenv("LASTFM_API_KEY", "key123")
    db_module.log_listen("505", "Arctic Monkeys")
    # key is set but lookups find nothing -> tabs stay hidden, no crash
    with patch("lastfm.requests.get", _fake_lastfm_get), \
         patch("lastfm.time.sleep"), \
         patch("bpm.requests.get",
               return_value=MagicMock(status_code=200,
                                      **{"json.return_value": {"search": []}})), \
         patch("bpm.time.sleep"):
        body = app_client.get(
            "/api/recommendations?limit=5&refresh=1").get_json()
    assert body["tempo_available"] is False
    assert "upbeat" not in body["streams"]
    assert body["tempo_error"] is None


def test_tempo_error_surfaced_on_bad_key(app_client, monkeypatch):
    monkeypatch.setenv("GETSONGBPM_API_KEY", "badkey")
    monkeypatch.setenv("LASTFM_API_KEY", "key123")
    db_module.log_listen("505", "Arctic Monkeys")
    bad = MagicMock(status_code=200)
    bad.json.return_value = {"error": "Invalid API Key, or inactive."}
    with patch("lastfm.requests.get", _fake_lastfm_get), \
         patch("lastfm.time.sleep"), \
         patch("bpm.requests.get", return_value=bad), \
         patch("bpm.time.sleep"):
        body = app_client.get(
            "/api/recommendations?limit=5&refresh=1").get_json()
    assert body["tempo_available"] is False
    assert "Invalid API Key" in (body["tempo_error"] or "")
