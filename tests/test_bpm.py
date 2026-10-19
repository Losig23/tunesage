"""Tests for the GetSongBPM tempo integration.

No real HTTP anywhere: requests.get is mocked. DB tests run against a
throwaway SQLite file via the tmpdb fixture pattern.
"""

import functools
from unittest.mock import MagicMock, patch

import pytest

import bpm
import db


@pytest.fixture
def tmpdb(tmp_path, monkeypatch):
    p = tmp_path / "bpm_test.db"
    db.init_db(p)

    def _bind(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            kwargs.setdefault("path", p)
            return fn(*args, **kwargs)
        return wrapper

    for name in ("get_tempo", "set_tempo"):
        monkeypatch.setattr(db, name, _bind(getattr(db, name)))
    return p


def _resp(search):
    r = MagicMock(status_code=200)
    r.json.return_value = {"search": search}
    return r


def _hit(title, artist, tempo):
    return {"title": title, "artist": {"name": artist}, "tempo": tempo}


def test_get_tempo_prefers_artist_match():
    payload = [_hit("Borderline", "Tame Impala", "110"),
               _hit("Borderline", "Some Cover Band", "140")]
    with patch("bpm.requests.get", return_value=_resp(payload)):
        assert bpm.get_tempo("k", "Borderline", "Tame Impala") == 110.0


def test_get_tempo_falls_back_to_top_hit():
    payload = [_hit("Borderline", "Unknown Artist", "125")]
    with patch("bpm.requests.get", return_value=_resp(payload)):
        assert bpm.get_tempo("k", "Borderline", "Nobody") == 125.0


def test_get_tempo_none_when_no_results():
    with patch("bpm.requests.get", return_value=_resp([])):
        assert bpm.get_tempo("k", "Nonexistent Song", "Nobody") is None


def test_get_tempo_none_on_garbage_tempo():
    payload = [_hit("X", "Y", "not-a-number")]
    with patch("bpm.requests.get", return_value=_resp(payload)):
        assert bpm.get_tempo("k", "X", "Y") is None


def test_get_tempo_raises_on_http_error():
    r = MagicMock(status_code=500)
    r.text = "boom"
    with patch("bpm.requests.get", return_value=r):
        with pytest.raises(bpm.BpmError):
            bpm.get_tempo("k", "X", "Y")


def test_get_tempo_requires_key():
    with pytest.raises(bpm.BpmError):
        bpm.get_tempo("", "X", "Y")


def test_db_tempo_roundtrip(tmpdb):
    assert db.get_tempo("A", "B") is None
    db.set_tempo("A", "B", 128.0)
    assert db.get_tempo("A", "B") == 128.0


def test_fill_tempos_uses_cache_not_network(tmpdb):
    db.set_tempo("Cached Song", "Cached Artist", 100.0)
    cands = [{"title": "Cached Song", "artist": "Cached Artist"}]
    with patch("bpm.requests.get") as mock_get:
        n = bpm.fill_tempos("k", cands)
    assert n == 0  # no HTTP — served from cache
    mock_get.assert_not_called()
    assert cands[0]["tempo"] == 100.0


def test_fill_tempos_caches_misses_and_swallows_errors(tmpdb):
    ok = _resp([_hit("Good", "Artist", "120")])
    bad = MagicMock(status_code=500)
    bad.text = "boom"
    cands = [{"title": "Good", "artist": "Artist"},
             {"title": "Bad", "artist": "Artist"}]
    # max_workers=1: deterministic request order for the side_effect list
    with patch("bpm.requests.get", side_effect=[ok, bad]):
        n = bpm.fill_tempos("k", cands, max_workers=1)
    assert n == 1  # only the successful lookup counted
    assert cands[0]["tempo"] == 120.0
    assert "tempo" not in cands[1]  # failure swallowed, no tempo key
    assert db.get_tempo("Good", "Artist") == 120.0  # persisted


def test_fill_tempos_no_key_does_nothing(tmpdb):
    cands = [{"title": "X", "artist": "Y"}]
    with patch("bpm.requests.get") as mock_get:
        assert bpm.fill_tempos("", cands) == 0
    mock_get.assert_not_called()


def test_get_tempo_raises_on_api_error_body():
    r = MagicMock(status_code=200)
    r.json.return_value = {"error": "Invalid API Key, or inactive."}
    with patch("bpm.requests.get", return_value=r):
        with pytest.raises(bpm.BpmError, match="Invalid API Key"):
            bpm.get_tempo("badkey", "X", "Y")


def test_fill_tempos_reraises_key_errors(tmpdb):
    r = MagicMock(status_code=200)
    r.json.return_value = {"error": "Invalid API Key, or inactive."}
    cands = [{"title": "X", "artist": "Y"}]
    with patch("bpm.requests.get", return_value=r):
        with pytest.raises(bpm.BpmError, match="Invalid API Key"):
            bpm.fill_tempos("badkey", cands)


def test_get_tempo_uses_getsong_co_endpoint():
    with patch("bpm.requests.get", return_value=_resp([])) as mock_get:
        bpm.get_tempo("k", "X", "Y")
    url = mock_get.call_args[0][0]
    assert url == "https://api.getsong.co/search/"


def test_get_tempo_uses_both_type_with_artist_refinement():
    hit = _resp([{"tempo": "172", "artist": {"name": "The Weeknd"}}])
    with patch("bpm.requests.get", return_value=hit) as mock_get:
        assert bpm.get_tempo("k", "Blinding Lights", "The Weeknd") == 172.0
    params = mock_get.call_args[1]["params"]
    assert params["type"] == "both"
    assert params["lookup"] == "song:Blinding Lights artist:The Weeknd"


def test_get_tempo_falls_back_to_plain_title_search():
    empty = MagicMock(status_code=200)
    empty.json.return_value = {"search": []}
    hit = _resp([{"tempo": "128", "artist": {"name": "X"}}])
    with patch("bpm.requests.get", side_effect=[empty, hit]) as mock_get:
        assert bpm.get_tempo("k", "X", "Y") == 128.0
    calls = mock_get.call_args_list
    assert calls[0][1]["params"]["type"] == "both"
    assert calls[1][1]["params"]["type"] == "song"
    assert calls[1][1]["params"]["lookup"] == "X"


def test_get_tempo_treats_non_list_search_as_miss():
    r = MagicMock(status_code=200)
    r.json.return_value = {"search": "no results"}
    with patch("bpm.requests.get", return_value=r):
        assert bpm.get_tempo("k", "X", "Y") is None
