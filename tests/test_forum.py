"""Tests for forum.mine_recommendations orchestration.

The live Last.fm HTTP layer is tested in test_lastfm.py; here we check the
routing: live path when a key is set, graceful fallback otherwise.
"""

from unittest.mock import patch

import forum


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
