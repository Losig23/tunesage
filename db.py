"""SQLite persistence layer for TuneSage.

Tables:
  songs            - one row per unique (title, artist); play_count is the
                     aggregated listen count ("most listened").
  recommendations  - cache of the last pipeline run (JSON payload + timestamp).
  feedback         - one row per (song_title, artist) with like/dislike counts.
  weights          - single-row table holding the hybrid ranker weights.
  spotify_auth     - single row (id=1) with the connected Spotify account's
                     OAuth tokens (access/refresh, expiry, display name).
  spotify_imported - played_at TEXT PRIMARY KEY -> song_id; dedupe log so a
                     Spotify re-import never double-counts a play.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import spotify as spotify_client

DB_PATH = Path(__file__).with_name("tunesage.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS songs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    artist TEXT NOT NULL,
    genre TEXT DEFAULT '',
    tags TEXT DEFAULT '[]',           -- JSON list of mood tags
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
CREATE TABLE IF NOT EXISTS spotify_auth (
    id INTEGER PRIMARY KEY CHECK (id = 1),  -- single auth row
    access_token TEXT NOT NULL,
    refresh_token TEXT NOT NULL,
    expires_at INTEGER NOT NULL,            -- unix timestamp
    spotify_user_id TEXT NOT NULL DEFAULT '',
    display_name TEXT NOT NULL DEFAULT '',
    connected_at INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS spotify_imported (
    played_at TEXT PRIMARY KEY,             -- Spotify ISO-8601 timestamp; dedupe key
    song_id INTEGER NOT NULL               -- FK -> songs(id)
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
        conn.execute(
            "INSERT OR IGNORE INTO weights (id, w_content, w_forum, w_feedback)"
            " VALUES (1, 0.5, 0.35, 0.15)"
        )
        conn.commit()
    finally:
        conn.close()


# ---- songs / listens -------------------------------------------------------

def log_listen(title: str, artist: str, genre: str = "",
               tags: list | None = None,
               path: Path | str = DB_PATH) -> dict:
    """Upsert a song and increment its play_count. Returns the row as a dict."""
    title, artist = title.strip(), artist.strip()
    genre = (genre or "").strip()
    tags = tags or []
    now = int(time.time())
    conn = get_conn(path)
    try:
        conn.execute(
            """INSERT INTO songs (title, artist, genre, tags, play_count, updated_at)
               VALUES (?, ?, ?, ?, 1, ?)
               ON CONFLICT(title, artist) DO UPDATE SET
                   play_count = play_count + 1,
                   genre = excluded.genre,
                   tags = excluded.tags,
                   updated_at = excluded.updated_at""",
            (title, artist, genre, json.dumps(tags), now),
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


# ---- spotify ---------------------------------------------------------------

def save_spotify_auth(access_token: str, refresh_token: str, expires_at: int,
                      spotify_user_id: str = "", display_name: str = "",
                      path: Path | str = DB_PATH) -> None:
    """Store (or replace) the connected Spotify account's tokens."""
    conn = get_conn(path)
    try:
        conn.execute(
            """INSERT INTO spotify_auth
                   (id, access_token, refresh_token, expires_at,
                    spotify_user_id, display_name, connected_at)
               VALUES (1, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                   access_token = excluded.access_token,
                   refresh_token = excluded.refresh_token,
                   expires_at = excluded.expires_at,
                   spotify_user_id = excluded.spotify_user_id,
                   display_name = excluded.display_name,
                   connected_at = excluded.connected_at""",
            (access_token, refresh_token, expires_at,
             spotify_user_id, display_name, int(time.time())),
        )
        conn.commit()
    finally:
        conn.close()


def get_spotify_auth(path: Path | str = DB_PATH) -> dict | None:
    """The stored Spotify auth row, or None if no account is connected."""
    conn = get_conn(path)
    try:
        row = conn.execute(
            "SELECT * FROM spotify_auth WHERE id = 1").fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def clear_spotify_auth(path: Path | str = DB_PATH) -> None:
    """Disconnect: drop the stored tokens."""
    conn = get_conn(path)
    try:
        conn.execute("DELETE FROM spotify_auth WHERE id = 1")
        conn.commit()
    finally:
        conn.close()


def get_valid_access_token(path: Path | str = DB_PATH) -> str | None:
    """A usable Spotify access token, or None if no account is connected.

    Returns the stored token while it's unexpired (60s skew buffer); when
    expired, refreshes it via the stored refresh token and persists the new
    tokens. Spotify's refresh response may omit a new refresh token — in that
    case the previous one is kept.
    """
    auth = get_spotify_auth(path=path)
    if not auth:
        return None
    if auth["expires_at"] > int(time.time()) + 60:
        return auth["access_token"]
    try:
        tokens = spotify_client.refresh_access_token(auth["refresh_token"])
    except Exception:
        return None
    refresh_token = tokens.get("refresh_token") or auth["refresh_token"]
    expires_at = int(time.time()) + tokens["expires_in"]
    save_spotify_auth(tokens["access_token"], refresh_token, expires_at,
                      auth["spotify_user_id"], auth["display_name"],
                      path=path)
    return tokens["access_token"]


def spotify_already_imported(played_at: str,
                             path: Path | str = DB_PATH) -> bool:
    """True if this played_at timestamp was imported before (dedupe check)."""
    conn = get_conn(path)
    try:
        return conn.execute(
            "SELECT 1 FROM spotify_imported WHERE played_at = ?",
            (played_at,)).fetchone() is not None
    finally:
        conn.close()


def record_spotify_import(played_at: str, song_id: int,
                          path: Path | str = DB_PATH) -> None:
    """Record that a played_at was imported as song_id (INSERT OR IGNORE —
    a second import of the same play is a no-op, never a double count)."""
    conn = get_conn(path)
    try:
        conn.execute(
            "INSERT OR IGNORE INTO spotify_imported (played_at, song_id)"
            " VALUES (?, ?)",
            (played_at, song_id),
        )
        conn.commit()
    finally:
        conn.close()


def last_spotify_import_at(path: Path | str = DB_PATH) -> str | None:
    """Newest imported played_at (ISO-8601 strings sort lexicographically)."""
    conn = get_conn(path)
    try:
        row = conn.execute(
            "SELECT MAX(played_at) AS m FROM spotify_imported").fetchone()
        return row["m"] if row and row["m"] else None
    finally:
        conn.close()
