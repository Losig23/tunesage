"""Tests for the iTunes artwork fallback (artwork.py). No real HTTP."""

from unittest.mock import MagicMock, patch

import artwork


def _itunes_resp(artwork_url=None):
    r = MagicMock(status_code=200)
    r.json.return_value = ({"results": [{"artworkUrl100": artwork_url}]}
                           if artwork_url else {"results": []})
    return r


def test_fetch_itunes_artwork_returns_300px():
    with patch("artwork.requests.get",
               return_value=_itunes_resp(
                   "https://is1-ssl.mzstatic.com/image/100x100bb.jpg")):
        url = artwork.fetch_itunes_artwork("Borderline", "Tame Impala")
    assert url == "https://is1-ssl.mzstatic.com/image/300x300bb.jpg"


def test_fetch_itunes_artwork_empty_on_no_results():
    with patch("artwork.requests.get",
               return_value=_itunes_resp()):
        assert artwork.fetch_itunes_artwork("X", "Y") == ""


def test_fetch_itunes_artwork_empty_on_http_error():
    import requests as _requests
    r = MagicMock(status_code=500)
    r.raise_for_status.side_effect = _requests.HTTPError("500")
    with patch("artwork.requests.get", return_value=r):
        assert artwork.fetch_itunes_artwork("X", "Y") == ""
    with patch("artwork.requests.get",
               side_effect=_requests.RequestException("down")):
        assert artwork.fetch_itunes_artwork("X", "Y") == ""


def test_mine_candidates_falls_back_to_itunes_art():
    import lastfm
    similar = {"similarartists": {"artist": [{"name": "MGMT", "match": "0.7"}]}}
    tracks = {"toptracks": {"track": [
        {"name": "Kids", "listeners": "60000", "playcount": "1",
         "artist": {"name": "MGMT"},
         # only the default placeholder -> treated as missing
         "image": [{"#text": "https://lastfm.freetls.fastly.net/i/u/300x300/"
                             "2a96cbd8b46e442fc41c2b86b821b4ce.png",
                    "size": "large"}]}]}}
    def _fake_get(url, params=None, **kwargs):
        r = MagicMock(status_code=200)
        method = (params or {}).get("method")
        r.json.return_value = similar if method == "artist.getSimilar" else tracks
        return r
    with patch("lastfm.requests.get", _fake_get), \
         patch("lastfm.time.sleep"), \
         patch("artwork.fetch_itunes_artwork",
               return_value="https://itunes/art.jpg") as mock_art:
        mined = lastfm.mine_candidates(
            "key123", ["Arctic Monkeys"],
            max_artists=1, similar_per_artist=1, tracks_per_artist=1,
            hop2_seeds=0)
    cand = mined["candidates"][0]
    assert cand["image"] == "https://itunes/art.jpg"
    mock_art.assert_called_once_with("Kids", "MGMT")
