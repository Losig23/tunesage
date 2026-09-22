"""TuneSage — AI-powered music recommendation platform.

Flask app factory + routes. Thin controllers: HTTP in/out here, ML in
recommender.py, forum mining in forum.py, persistence in db.py.
"""

from __future__ import annotations

import logging
import os
import secrets

from flask import Flask, jsonify, request, send_from_directory
from dotenv import load_dotenv

load_dotenv()  # populate os.environ from .env (if present); real env vars win

import db
import forum as forum_miner
import lastfm as lastfm_client
import recommender as engine

log = logging.getLogger(__name__)


def _secret_key() -> str:
    key = os.environ.get("SECRET_KEY", "").strip()
    if key:
        return key
    log.warning(
        "SECRET_KEY is not set; using an ephemeral dev key. "
        "Set SECRET_KEY in production."
    )
    return secrets.token_hex(32)


def create_app() -> Flask:
    app = Flask(__name__, static_folder="static", static_url_path="/static")
    app.secret_key = _secret_key()
    db.init_db()

    @app.get("/api/health")
    def health():
        return jsonify({"status": "ok", "service": "tunesage"})

    # ---- listens ------------------------------------------------------
    @app.post("/api/listens")
    def log_listen():
        data = request.get_json(force=True, silent=True) or {}
        title = (data.get("title") or "").strip()
        artist = (data.get("artist") or "").strip()
        if not title or not artist:
            return jsonify({"error": "title and artist are required"}), 400
        song = db.log_listen(title, artist,
                             genre=data.get("genre", ""),
                             tags=data.get("tags", []))
        return jsonify(song), 201

    @app.get("/api/listens/top")
    def top_listens():
        try:
            limit = max(1, min(50, int(request.args.get("limit", 10))))
        except ValueError:
            limit = 10
        return jsonify({"songs": db.top_songs(limit)})

    # ---- recommendations ----------------------------------------------
    @app.get("/api/recommendations")
    def recommendations():
        try:
            limit = max(1, min(50, int(request.args.get("limit", 10))))
        except ValueError:
            limit = 10
        refresh = request.args.get("refresh", "0") == "1"

        if not refresh:
            cached = db.get_cached_recommendations()
            if cached is not None:
                cached["songs"] = cached["songs"][:limit]
                cached["cached"] = True
                return jsonify(cached)

        seeds = db.top_songs(10)
        if not seeds:
            return jsonify({"songs": [], "threads": [],
                            "message": "Log some listens first!"})

        mined = forum_miner.mine_recommendations(seeds)

        # Deduplicate: don't recommend songs the user already listens to.
        known = {(s["title"].lower(), s["artist"].lower()) for s in seeds}
        candidates = [c for c in mined["candidates"]
                      if (c["title"].lower(), c["artist"].lower()) not in known]

        vectorizer, profile = engine.build_taste_profile(seeds)
        content_scores = engine.score_candidates(candidates, vectorizer,
                                                 profile)
        weights = db.get_weights()
        fb_counts = db.feedback_counts()
        ranked = engine.hybrid_rank(candidates, content_scores, weights,
                                    fb_counts)
        profile_terms = engine.top_profile_terms(vectorizer, profile)
        for c in ranked:
            c["why"] = engine.explain(c, profile_terms)

        payload = {"songs": ranked[:limit], "threads": mined["threads"][:10],
                   "source": mined["source"], "weights": weights,
                   "cached": False}
        db.save_recommendations(payload)
        return jsonify(payload)

    # ---- feedback (online learning) ------------------------------------
    @app.post("/api/feedback")
    def feedback():
        data = request.get_json(force=True, silent=True) or {}
        title = (data.get("song_title") or "").strip()
        artist = (data.get("artist") or "").strip()
        if not title or not artist or "liked" not in data:
            return jsonify({"error": "song_title, artist and liked required"}), 400
        liked = bool(data["liked"])
        db.record_feedback(title, artist, liked)
        weights = db.nudge_weights(liked)
        return jsonify({"ok": True, "liked": liked, "weights": weights})

    # ---- frontend -------------------------------------------------------
    @app.get("/")
    def index():
        return send_from_directory(app.static_folder, "index.html")

    # ---- last.fm --------------------------------------------------------
    @app.get("/api/lastfm/status")
    def lastfm_status():
        api_key, username = _lastfm_config()
        if not api_key or not username:
            return jsonify({
                "configured": False,
                "setup": "Last.fm is not configured on this server. Set the "
                         "LASTFM_API_KEY and LASTFM_USERNAME environment "
                         "variables (see README 'Last.fm integration').",
            })
        return jsonify({
            "configured": True,
            "username": username,
            "last_import_at": db.last_import_at("lastfm"),
        })

    @app.post("/api/lastfm/import")
    def lastfm_import():
        """Import recent Last.fm scrobbles into the listening history.

        Paginates up to ~200 scrobbles (newest first). Each new play is
        logged through the same log_listen() path as manual logging (one
        play each); already-imported timestamps are skipped, so re-imports
        never double-count.
        """
        api_key, username = _lastfm_config()
        if not api_key or not username:
            return jsonify({
                "error": "Last.fm is not configured on this server.",
                "setup": "Set the LASTFM_API_KEY and LASTFM_USERNAME "
                         "environment variables "
                         "(see README 'Last.fm integration').",
            }), 400
        try:
            tracks = lastfm_client.get_all_recent_tracks(
                api_key, username, total_limit=200)
        except lastfm_client.LastfmError as e:
            return jsonify({"error": str(e)}), 502
        imported, skipped, titles = 0, 0, []
        for t in tracks:
            if db.already_imported("lastfm", t["played_at"]):
                skipped += 1
                continue
            song = db.log_listen(t["title"], t["artist"])
            db.record_import("lastfm", t["played_at"], song["id"])
            imported += 1
            titles.append(f"{t['title']} — {t['artist']}")
        return jsonify({"imported": imported, "skipped": skipped,
                        "songs": titles})

    return app


def _lastfm_config() -> tuple[str, str]:
    """(api_key, username) from the environment; empty strings when unset."""
    return (os.environ.get("LASTFM_API_KEY", "").strip(),
            os.environ.get("LASTFM_USERNAME", "").strip())


app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
