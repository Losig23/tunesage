"""SQLite persistence layer for TuneSage.

Tables:
  songs            - one row per unique (title, artist); play_count is the
                     aggregated listen count ("most listened").
  recommendations  - cache of the last pipeline run (JSON payload + timestamp).
  feedback         - one row per (song_title, artist) with like/dislike counts.
  weights          - single-row table holding the hybrid ranker weights.
  external_imports - (source, played_at) -> song_id; dedupe log so a
                     re-import (e.g. Last.fm scrobbles) never double-counts.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

DB_PATH = Path(__file__).with_name("tunesage.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS songs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    artist TEXT NOT NULL,
    genre TEXT DEFAULT '',
    tags TEXT DEFAULT '[]',           -- JSON list of mood tags
    image_url TEXT DEFAULT '',        -- album art (Last.fm); backfilled by _migrate
    play_count INTEGER NOT NULL DEFAULT 0,
    updated_at INTEGER NOT NULL,
    UNIQUE(title, artist)
);
CREATE TABLE IF NOT EXISTS recommendations (
    id INTEGER PRIMARY KEY CHECK (id = 1),  -- single cache row
    payload TEXT NOT NULL,                  -- JSON
    created_at INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    song_title TEXT NOT NULL,
    artist TEXT NOT NULL,
    liked INTEGER NOT NULL,                 -- 1 = like, 0 = dislike
    created_at INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS weights (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    w_content REAL NOT NULL DEFAULT 0.5,
    w_forum REAL NOT NULL DEFAULT 0.35,
    w_feedback REAL NOT NULL DEFAULT 0.15
);
CREATE INDEX IF NOT EXISTS idx_songs_playcount ON songs(play_count DESC);
-- Removed legacy tables from the old Spotify integration (dropped in favor
-- of the generic external_imports table used by the Last.fm importer).
DROP TABLE IF EXISTS spotify_auth;
DROP TABLE IF EXISTS spotify_imported;
CREATE TABLE IF NOT EXISTS external_imports (
    source TEXT NOT NULL,               -- e.g. 'lastfm'
    played_at TEXT NOT NULL,            -- ISO-8601 play timestamp; dedupe key
    song_id INTEGER NOT NULL,           -- FK -> songs(id)
    PRIMARY KEY (source, played_at)
);
-- Tempo lookups (GetSongBPM) are cached forever: BPM never changes, so each
-- unique track is looked up once no matter how often mining refreshes.
CREATE TABLE IF NOT EXISTS bpm_cache (
    title TEXT NOT NULL,
    artist TEXT NOT NULL,
    tempo REAL NOT NULL,
    fetched_at INTEGER NOT NULL,
    PRIMARY KEY (title, artist)
);
-- Last.fm API responses: similar artists and top tracks change slowly, so
-- mining reuses them instead of re-fetching every refresh. TTLs are enforced
-- by the caller (get_lastfm_cached takes max_age_seconds).
CREATE TABLE IF NOT EXISTS lastfm_cache (
    method TEXT NOT NULL,
    cache_key TEXT NOT NULL,
    response TEXT NOT NULL,
    fetched_at INTEGER NOT NULL,
    PRIMARY KEY (method, cache_key)
);
-- Resolved artwork URLs (iTunes + Last.fm album art): art never changes for
-- a given track, so each unique track is resolved once, like tempo.
CREATE TABLE IF NOT EXISTS art_cache (
    title TEXT NOT NULL,
    artist TEXT NOT NULL,
    url TEXT NOT NULL,
    fetched_at INTEGER NOT NULL,
    PRIMARY KEY (title, artist)
);
-- Spotify playback tokens (single row, id=1): OAuth access/refresh tokens
-- plus the pending OAuth state for the login round-trip. Playback only —
-- no Spotify data is stored or fed to the recommender.
CREATE TABLE IF NOT EXISTS spotify_tokens (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    access_token TEXT,
    refresh_token TEXT,
    expires_at INTEGER,
    oauth_state TEXT
);
"""


def get_conn(path: Path | str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(path: Path | str = DB_PATH) -> None:
    conn = get_conn(path)
    try:
        conn.executescript(SCHEMA)
        _migrate(conn)
        conn.execute(
            "INSERT OR IGNORE INTO weights (id, w_content, w_forum, w_feedback)"
            " VALUES (1, 0.5, 0.35, 0.15)"
        )
        conn.commit()
    finally:
        conn.close()


def _migrate(conn: sqlite3.Connection) -> None:
    """Lightweight migrations for DBs created before a column existed.

    CREATE TABLE IF NOT EXISTS won't add columns to an existing table, so
    each addition gets an explicit guarded ALTER TABLE here.
    """
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(songs)")}
    if "image_url" not in cols:
        conn.execute("ALTER TABLE songs ADD COLUMN image_url TEXT DEFAULT ''")


# ---- songs / listens -------------------------------------------------------

def log_listen(title: str, artist: str, genre: str = "",
               tags: list | None = None, image_url: str = "",
               path: Path | str = DB_PATH) -> dict:
    """Upsert a song and increment its play_count. Returns the row as a dict.

    image_url is only overwritten when a non-empty one is provided, so a
    manual log can't blank art that a scrobble import already stored (and
    vice versa).
    """
    title, artist = title.strip(), artist.strip()
    genre = (genre or "").strip()
    tags = tags or []
    image_url = (image_url or "").strip()
    now = int(time.time())
    conn = get_conn(path)
    try:
        conn.execute(
            """INSERT INTO songs (title, artist, genre, tags, image_url,
                                  play_count, updated_at)
               VALUES (?, ?, ?, ?, ?, 1, ?)
               ON CONFLICT(title, artist) DO UPDATE SET
                   play_count = play_count + 1,
                   genre = excluded.genre,
                   tags = excluded.tags,
                   image_url = CASE WHEN excluded.image_url <> ''
                                    THEN excluded.image_url
                                    ELSE songs.image_url END,
                   updated_at = excluded.updated_at""",
            (title, artist, genre, json.dumps(tags), image_url, now),
        )
        conn.commit()
        row = conn.execute(
            "SELECT * FROM songs WHERE title = ? AND artist = ?",
            (title, artist),
        ).fetchone()
        return _row_to_song(dict(row))
    finally:
        conn.close()


def top_songs(limit: int = 10, path: Path | str = DB_PATH) -> list[dict]:
    """Most-listened songs, play_count DESC. (Top-k via ORDER BY/LIMIT;
    in-process equivalent is a heap — see STUDY_GUIDE.md.)"""
    conn = get_conn(path)
    try:
        rows = conn.execute(
            "SELECT * FROM songs ORDER BY play_count DESC, updated_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [_row_to_song(dict(r)) for r in rows]
    finally:
        conn.close()


def _row_to_song(row: dict) -> dict:
    row["tags"] = json.loads(row.get("tags") or "[]")
    return row


# ---- recommendation cache ---------------------------------------------------

CACHE_TTL_SECONDS = 6 * 3600  # 6 hours


def get_cached_recommendations(path: Path | str = DB_PATH) -> dict | None:
    conn = get_conn(path)
    try:
        row = conn.execute(
            "SELECT payload, created_at FROM recommendations WHERE id = 1"
        ).fetchone()
        if row is None:
            return None
        if int(time.time()) - row["created_at"] > CACHE_TTL_SECONDS:
            return None
        data = json.loads(row["payload"])
        data["_cache_age_seconds"] = int(time.time()) - row["created_at"]
        return data
    finally:
        conn.close()


def save_recommendations(payload: dict, path: Path | str = DB_PATH) -> None:
    conn = get_conn(path)
    try:
        conn.execute(
            "INSERT INTO recommendations (id, payload, created_at) VALUES (1, ?, ?)"
            " ON CONFLICT(id) DO UPDATE SET payload = excluded.payload,"
            " created_at = excluded.created_at",
            (json.dumps(payload), int(time.time())),
        )
        conn.commit()
    finally:
        conn.close()


# ---- feedback ----------------------------------------------------------------

def record_feedback(song_title: str, artist: str, liked: bool,
                    path: Path | str = DB_PATH) -> None:
    conn = get_conn(path)
    try:
        conn.execute(
            "INSERT INTO feedback (song_title, artist, liked, created_at)"
            " VALUES (?, ?, ?, ?)",
            (song_title.strip(), artist.strip(), 1 if liked else 0,
             int(time.time())),
        )
        conn.commit()
    finally:
        conn.close()


def feedback_counts(path: Path | str = DB_PATH) -> dict[tuple[str, str], dict]:
    """{(title, artist): {'likes': n, 'dislikes': n}} — drives feedback_boost."""
    conn = get_conn(path)
    try:
        rows = conn.execute(
            """SELECT song_title, artist,
                      SUM(liked) AS likes,
                      SUM(1 - liked) AS dislikes,
                      COUNT(*) AS total
               FROM feedback GROUP BY song_title, artist"""
        ).fetchall()
        return {
            (r["song_title"].lower(), r["artist"].lower()): {
                "likes": r["likes"] or 0,
                "dislikes": r["dislikes"] or 0,
                "total": r["total"],
            }
            for r in rows
        }
    finally:
        conn.close()


# ---- adaptive weights ---------------------------------------------------------

def get_weights(path: Path | str = DB_PATH) -> dict:
    conn = get_conn(path)
    try:
        row = conn.execute(
            "SELECT w_content, w_forum, w_feedback FROM weights WHERE id = 1"
        ).fetchone()
        return dict(row)
    finally:
        conn.close()


def nudge_weights(liked: bool, path: Path | str = DB_PATH) -> dict:
    """Simple online learning: a like slightly increases the weight of the
    content and forum signals that produced a good recommendation; a dislike
    slightly decreases them, pushing mass toward the other signals.

    We renormalize so weights always sum to 1. Deltas are small (0.02) to keep
    learning stable — one data point shouldn't swing the model.
    """
    w = get_weights(path)
    delta = 0.02 if liked else -0.02
    w["w_content"] = max(0.05, w["w_content"] + delta)
    w["w_forum"] = max(0.05, w["w_forum"] + delta * 0.5)
    total = w["w_content"] + w["w_forum"] + w["w_feedback"]
    w = {k: round(v / total, 4) for k, v in w.items()}
    conn = get_conn(path)
    try:
        conn.execute(
            "UPDATE weights SET w_content = ?, w_forum = ?, w_feedback = ?"
            " WHERE id = 1",
            (w["w_content"], w["w_forum"], w["w_feedback"]),
        )
        conn.commit()
    finally:
        conn.close()
    return w


# ---- external imports (Last.fm scrobbles, etc.) ------------------------------

def already_imported(source: str, played_at: str,
                     path: Path | str = DB_PATH) -> bool:
    """True if this (source, played_at) was imported before (dedupe check)."""
    conn = get_conn(path)
    try:
        return conn.execute(
            "SELECT 1 FROM external_imports WHERE source = ? AND played_at = ?",
            (source, played_at)).fetchone() is not None
    finally:
        conn.close()


def record_import(source: str, played_at: str, song_id: int,
                  path: Path | str = DB_PATH) -> None:
    """Record that (source, played_at) was imported as song_id.

    INSERT OR IGNORE — a second import of the same play is a no-op, never a
    double count.
    """
    conn = get_conn(path)
    try:
        conn.execute(
            "INSERT OR IGNORE INTO external_imports (source, played_at, song_id)"
            " VALUES (?, ?, ?)",
            (source, played_at, song_id),
        )
        conn.commit()
    finally:
        conn.close()


def last_import_at(source: str, path: Path | str = DB_PATH) -> str | None:
    """Newest imported played_at for a source (ISO-8601 sorts lexicographically)."""
    conn = get_conn(path)
    try:
        row = conn.execute(
            "SELECT MAX(played_at) AS m FROM external_imports"
            " WHERE source = ?",
            (source,)).fetchone()
        return row["m"] if row and row["m"] else None
    finally:
        conn.close()


# ---- tempo cache (GetSongBPM) --------------------------------------------------

def get_tempo(title: str, artist: str,
              path: Path | str = DB_PATH) -> float | None:
    """Cached BPM for a track, or None if never looked up."""
    conn = get_conn(path)
    try:
        row = conn.execute(
            "SELECT tempo FROM bpm_cache WHERE title = ? AND artist = ?",
            (title.strip(), artist.strip())).fetchone()
        return float(row["tempo"]) if row else None
    finally:
        conn.close()


def set_tempo(title: str, artist: str, tempo: float,
              path: Path | str = DB_PATH) -> None:
    """Cache a track's BPM (INSERT OR REPLACE — tempo never changes)."""
    conn = get_conn(path)
    try:
        conn.execute(
            "INSERT OR REPLACE INTO bpm_cache (title, artist, tempo, fetched_at)"
            " VALUES (?, ?, ?, ?)",
            (title.strip(), artist.strip(), float(tempo), int(time.time())),
        )
        conn.commit()
    finally:
        conn.close()


# ---- Last.fm response cache ----------------------------------------------------

def get_lastfm_cached(method: str, cache_key: str, max_age_seconds: int,
                      path: Path | str = DB_PATH):
    """Cached Last.fm API response (parsed JSON), or None if missing/stale."""
    conn = get_conn(path)
    try:
        row = conn.execute(
            "SELECT response, fetched_at FROM lastfm_cache"
            " WHERE method = ? AND cache_key = ?",
            (method, cache_key)).fetchone()
        if not row:
            return None
        if int(time.time()) - int(row["fetched_at"]) > max_age_seconds:
            return None
        return json.loads(row["response"])
    finally:
        conn.close()


def set_lastfm_cached(method: str, cache_key: str, response,
                      path: Path | str = DB_PATH) -> None:
    """Cache a Last.fm API response (stored as JSON)."""
    conn = get_conn(path)
    try:
        conn.execute(
            "INSERT OR REPLACE INTO lastfm_cache"
            " (method, cache_key, response, fetched_at)"
            " VALUES (?, ?, ?, ?)",
            (method, cache_key, json.dumps(response), int(time.time())),
        )
        conn.commit()
    finally:
        conn.close()


# ---- artwork cache -------------------------------------------------------------

def get_artwork(title: str, artist: str,
                path: Path | str = DB_PATH) -> str | None:
    """Cached artwork URL for a track, or None if never resolved."""
    conn = get_conn(path)
    try:
        row = conn.execute(
            "SELECT url FROM art_cache WHERE title = ? AND artist = ?",
            (title.strip(), artist.strip())).fetchone()
        return row["url"] if row else None
    finally:
        conn.close()


def set_artwork(title: str, artist: str, url: str,
                path: Path | str = DB_PATH) -> None:
    """Cache a track's artwork URL (INSERT OR REPLACE — art never changes)."""
    conn = get_conn(path)
    try:
        conn.execute(
            "INSERT OR REPLACE INTO art_cache (title, artist, url, fetched_at)"
            " VALUES (?, ?, ?, ?)",
            (title.strip(), artist.strip(), url, int(time.time())),
        )
        conn.commit()
    finally:
        conn.close()


# ---- Spotify playback tokens (single row) ---------------------------------------

def save_spotify_tokens(access_token: str, refresh_token: str | None,
                        expires_at: int,
                        path: Path | str = DB_PATH) -> None:
    """Store OAuth tokens (keeps the old refresh token when not rotated)."""
    conn = get_conn(path)
    try:
        conn.execute(
            "INSERT INTO spotify_tokens (id, access_token, refresh_token,"
            " expires_at) VALUES (1, ?, ?, ?)"
            " ON CONFLICT(id) DO UPDATE SET access_token = excluded.access_token,"
            " refresh_token = COALESCE(excluded.refresh_token,"
            " spotify_tokens.refresh_token),"
            " expires_at = excluded.expires_at",
            (access_token, refresh_token, int(expires_at)),
        )
        conn.commit()
    finally:
        conn.close()


def get_spotify_tokens(path: Path | str = DB_PATH) -> dict | None:
    """Stored OAuth tokens, or None when Spotify was never connected."""
    conn = get_conn(path)
    try:
        row = conn.execute(
            "SELECT access_token, refresh_token, expires_at"
            " FROM spotify_tokens WHERE id = 1").fetchone()
        if not row or not row["access_token"]:
            return None
        return {"access_token": row["access_token"],
                "refresh_token": row["refresh_token"],
                "expires_at": row["expires_at"]}
    finally:
        conn.close()


def clear_spotify_tokens(path: Path | str = DB_PATH) -> None:
    """Disconnect Spotify: drop the stored tokens."""
    conn = get_conn(path)
    try:
        conn.execute("DELETE FROM spotify_tokens WHERE id = 1")
        conn.commit()
    finally:
        conn.close()


def save_spotify_state(state: str, path: Path | str = DB_PATH) -> None:
    """Remember the OAuth state for the login round-trip (host-independent,
    unlike a session cookie: localhost vs 127.0.0.1 would lose it)."""
    conn = get_conn(path)
    try:
        conn.execute(
            "INSERT INTO spotify_tokens (id, oauth_state) VALUES (1, ?)"
            " ON CONFLICT(id) DO UPDATE SET oauth_state = excluded.oauth_state",
            (state,),
        )
        conn.commit()
    finally:
        conn.close()


def pop_spotify_state(path: Path | str = DB_PATH) -> str | None:
    """Read + clear the pending OAuth state (one-shot)."""
    conn = get_conn(path)
    try:
        row = conn.execute(
            "SELECT oauth_state FROM spotify_tokens WHERE id = 1").fetchone()
        state = row["oauth_state"] if row else None
        if state:
            conn.execute(
                "UPDATE spotify_tokens SET oauth_state = NULL WHERE id = 1")
            conn.commit()
        return state
    finally:
        conn.close()


def all_song_keys(path: Path | str = DB_PATH) -> set[tuple[str, str]]:
    """(title, artist) pairs for every logged song, lowercased/stripped.

    Used to exclude the user's whole listening history from
    recommendations (not just the top seeds that feed the taste profile).
    """
    conn = get_conn(path)
    try:
        rows = conn.execute("SELECT title, artist FROM songs").fetchall()
        return {(str(r["title"]).strip().lower(),
                 str(r["artist"]).strip().lower()) for r in rows}
    finally:
        conn.close()


def voted_keys(path: Path | str = DB_PATH) -> set[tuple[str, str]]:
    """(title, artist) pairs the user has voted on, lowercased/stripped.

    Songs already judged (liked or disliked) are excluded from future
    recommendation lists — a refresh shows new songs.
    """
    conn = get_conn(path)
    try:
        rows = conn.execute(
            "SELECT song_title, artist FROM feedback").fetchall()
        return {(str(r["song_title"]).strip().lower(),
                 str(r["artist"]).strip().lower()) for r in rows}
    finally:
        conn.close()
