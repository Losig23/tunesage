"""TuneSage — AI-powered music recommendation platform.

Flask app factory + routes. Thin controllers: HTTP in/out here, ML in
recommender.py, forum mining in forum.py, persistence in db.py.
"""

from __future__ import annotations

import os

from flask import Flask, jsonify, request, send_from_directory

import db
import forum as forum_miner
import recommender as engine


def create_app() -> Flask:
    app = Flask(__name__, static_folder="static", static_url_path="/static")
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

    return app


app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
