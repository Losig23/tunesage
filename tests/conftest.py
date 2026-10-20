"""Shared fixtures: temp DB with db helpers rebound to it.

Test-isolation norm: every db.py helper that takes a `path` arg must be in
the bind list below, or unbound calls hit the real dev DB.
"""

import functools

import pytest

import db


@pytest.fixture
def tmpdb(tmp_path, monkeypatch):
    """Temp DB + rebind db helpers used by the routes to use it."""
    p = tmp_path / "test.db"
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
                 "all_song_keys", "voted_keys", "record_feedback",
                 "get_tempo", "set_tempo",
                 "get_lastfm_cached", "set_lastfm_cached",
                 "get_artwork", "set_artwork",
                 "save_spotify_tokens", "get_spotify_tokens",
                 "clear_spotify_tokens", "save_spotify_state",
                 "pop_spotify_state"):
        monkeypatch.setattr(db, name, _bind(getattr(db, name)))
    return p


@pytest.fixture
def app_client(tmpdb, monkeypatch):
    monkeypatch.setenv("SECRET_KEY", "test-secret-key")
    import app as app_module  # imported after SECRET_KEY is set
    return app_module.app.test_client()
