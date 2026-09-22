"""TuneSage — AI-powered music recommendation platform.

Flask app factory + routes. Thin controllers: HTTP in/out here, ML in
recommender.py, forum mining in forum.py, persistence in db.py.
"""

from __future__ import annotations

import logging
import os
import secrets
import time

from flask import Flask, jsonify, redirect, request, send_from_directory, session

import db
import forum as forum_miner
import recommender as engine
import spotify as spotify_client

log = logging.getLogger(__name__)


def _secret_key() -> str:
    key = os.environ.get("SECRET_KEY", "").strip()
    if key:
        return key
    log.warning(
        "SECRET_KEY is not set; using an ephemeral dev key. Flask sessions "
        "(including Spotify OAuth state) will not survive restarts. "
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

    # ---- spotify --------------------------------------------------------
    @app.get("/api/spotify/connect")
    def spotify_connect():
        """Start OAuth: stash a random state in the session, redirect the
        user to Spotify's authorization page."""
        if not os.environ.get("SPOTIFY_CLIENT_ID", "").strip():
            return jsonify({
                "error": "Spotify is not configured on this server.",
                "setup": "Set the SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET "
                         "and SPOTIFY_REDIRECT_URI environment variables "
                         "(see README 'Spotify integration').",
            }), 400
        state = secrets.token_urlsafe(24)
        session["spotify_oauth_state"] = state
        try:
            url = spotify_client.build_authorize_url(state)
        except spotify_client.SpotifyError as e:
            return jsonify({"error": str(e)}), 400
        return redirect(url)

    @app.get("/api/spotify/callback")
    def spotify_callback():
        """Spotify redirects here after the user approves. Verify state
        (CSRF check), exchange the code for tokens, store them."""
        expected = session.pop("spotify_oauth_state", None)
        state = request.args.get("state", "")
        if not expected or not state or state != expected:
            return jsonify({
                "error": "OAuth state mismatch — please try connecting again."
            }), 400
        if request.args.get("error"):
            return jsonify({
                "error": "Spotify authorization failed: "
                         f"{request.args.get('error')}"
            }), 400
        code = request.args.get("code", "")
        if not code:
            return jsonify(
                {"error": "Missing authorization code from Spotify."}), 400
        try:
            tokens = spotify_client.exchange_code(code)
            profile = spotify_client.get_profile(tokens["access_token"])
        except spotify_client.SpotifyError as e:
            return jsonify({"error": str(e)}), 502
        db.save_spotify_auth(
            tokens["access_token"],
            tokens["refresh_token"] or "",
            int(time.time()) + tokens["expires_in"],
            profile["spotify_user_id"],
            profile["display_name"],
        )
        return redirect("/?spotify=connected")

    @app.get("/api/spotify/status")
    def spotify_status():
        auth = db.get_spotify_auth()
        if not auth:
            return jsonify({"connected": False})
        return jsonify({
            "connected": True,
            "display_name": auth["display_name"],
            "last_import_at": db.last_spotify_import_at(),
        })

    @app.post("/api/spotify/disconnect")
    def spotify_disconnect():
        db.clear_spotify_auth()
        return jsonify({"ok": True, "connected": False})

    @app.post("/api/spotify/import")
    def spotify_import():
        """Pull up to 50 recently-played tracks into the listening history.

        Each new played_at is logged through the same log_listen() path as
        manual logging (one play each); already-imported timestamps are
        skipped, so re-imports never double-count.
        """
        token = db.get_valid_access_token()
        if not token:
            return jsonify({
                "error": "Spotify is not connected. "
                         "Connect your account first."
            }), 401
        try:
            tracks = spotify_client.get_recently_played(token, limit=50)
        except spotify_client.SpotifyError as e:
            return jsonify({"error": str(e)}), 502
        imported, skipped, titles = 0, 0, []
        for t in tracks:
            if not t["played_at"] or db.spotify_already_imported(t["played_at"]):
                skipped += 1
                continue
            song = db.log_listen(t["title"], t["artist"])
            db.record_spotify_import(t["played_at"], song["id"])
            imported += 1
            titles.append(f"{t['title']} — {t['artist']}")
        return jsonify({"imported": imported, "skipped": skipped,
                        "songs": titles})

    return app


app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
