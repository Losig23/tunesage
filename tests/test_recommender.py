"""Unit tests for the TuneSage recommendation engine.

All engine functions are pure (no DB, no network), so they're trivially
testable — a design choice worth mentioning in interviews.
"""

import math

import pytest

import recommender as engine


@pytest.fixture
def listens():
    return [
        {"title": "Blinding Lights", "artist": "The Weeknd",
         "genre": "pop", "tags": ["energetic", "night-drive"], "play_count": 12},
        {"title": "Midnight City", "artist": "M83",
         "genre": "electronic", "tags": ["dreamy", "night-drive"], "play_count": 9},
        {"title": "Take Five", "artist": "Dave Brubeck Quartet",
         "genre": "jazz", "tags": ["chill"], "play_count": 2},
    ]


def test_build_taste_profile_shape(listens):
    vec, profile = engine.build_taste_profile(listens)
    assert vec is not None and profile is not None
    # unit-length centroid
    assert math.isclose(sum(x * x for x in profile) ** 0.5, 1.0, rel_tol=1e-6)
    # play-count weighting: pop/electronic vocab should dominate jazz
    terms = vec.get_feature_names_out().tolist()
    pop_idx = terms.index("pop")
    jazz_idx = terms.index("jazz")
    assert profile[pop_idx] > profile[jazz_idx]


def test_build_taste_profile_empty():
    vec, profile = engine.build_taste_profile([])
    assert vec is None and profile is None


def test_cosine_scoring_ordering(listens):
    """A clearly similar song must outrank a clearly dissimilar one."""
    vec, profile = engine.build_taste_profile(listens)
    similar = {"title": "Starboy", "artist": "The Weeknd",
               "genre": "pop", "tags": ["energetic", "night-drive"]}
    dissimilar = {"title": "Blue in Green", "artist": "Miles Davis",
                  "genre": "jazz", "tags": ["melancholy", "piano"]}
    scores = engine.score_candidates([similar, dissimilar], vec, profile)
    assert scores[0] > scores[1]
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_forum_score_more_mentions_and_positive_sentiment():
    loved = {"mentions": 8,
             "snippets": ["I love this masterpiece, amazing and beautiful!"]}
    hated = {"mentions": 8,
             "snippets": ["terrible boring track, awful and overrated, skip it"]}
    assert engine.forum_score(loved) > engine.forum_score(hated)
    # diminishing returns: 100 mentions shouldn't be 10x better than 10
    many = {"mentions": 100, "snippets": ["great track"]}
    assert engine.forum_score(many) < 2 * engine.forum_score(
        {"mentions": 10, "snippets": ["great track"]})


def test_hybrid_rank_weighting_and_sort():
    cands = [
        {"title": "A", "artist": "X", "mentions": 1, "snippets": ["meh"]},
        {"title": "B", "artist": "Y", "mentions": 9,
         "snippets": ["love this amazing masterpiece"]},
    ]
    content = [0.9, 0.1]  # A wins on content, B wins on forum
    w_content_heavy = {"w_content": 0.9, "w_forum": 0.05, "w_feedback": 0.05}
    w_forum_heavy = {"w_content": 0.05, "w_forum": 0.9, "w_feedback": 0.05}
    ranked_content = engine.hybrid_rank(cands, content, w_content_heavy)
    ranked_forum = engine.hybrid_rank(cands, content, w_forum_heavy)
    assert ranked_content[0]["title"] == "A"
    assert ranked_forum[0]["title"] == "B"
    # sorted desc, all keys present
    assert ranked_content[0]["score"] >= ranked_content[1]["score"]
    for c in ranked_content:
        assert {"content_score", "forum_score", "feedback_boost",
                "score"} <= set(c)


def test_feedback_boost_affects_rank():
    cands = [{"title": "A", "artist": "X", "mentions": 2, "snippets": ["great"]},
             {"title": "B", "artist": "Y", "mentions": 2, "snippets": ["great"]}]
    content = [0.5, 0.5]
    weights = {"w_content": 0.4, "w_forum": 0.3, "w_feedback": 0.3}
    fb = {("a", "x"): {"likes": 3, "dislikes": 0, "total": 3}}
    ranked = engine.hybrid_rank(cands, content, weights, fb)
    assert ranked[0]["title"] == "A"  # liked song floats up
    assert ranked[0]["feedback_boost"] == 1.0


def test_explain_mentions_terms_and_sources(listens):
    vec, profile = engine.build_taste_profile(listens)
    terms = engine.top_profile_terms(vec, profile)
    cand = {"title": "Starboy", "artist": "The Weeknd", "genre": "pop",
            "tags": ["energetic"], "mentions": 4,
            "sources": ["https://reddit.com/r/x"],
            "snippets": ["love this amazing track"]}
    why = engine.explain(cand, terms)
    assert isinstance(why, str) and len(why) > 20
    assert any(t in why for t in terms)  # cites a taste term
    assert "popular with other listeners" in why


def test_snippet_sentiment_bounds():
    assert engine.snippet_sentiment("love amazing masterpiece") == 1.0
    assert engine.snippet_sentiment("terrible awful boring") == -1.0
    assert engine.snippet_sentiment("the cat sat on the mat") == 0.0


def _ranked(n):
    # ranked by score desc already; listeners spread so tertiles are clear
    return [{"title": f"T{i}", "artist": "A", "score": 1.0 - i / 100,
             "listeners": (n - i) * 1000} for i in range(n)]


def test_split_streams_tertiles_by_listeners():
    streams = engine.split_streams(_ranked(9))
    assert [len(v) for v in streams.values()] == [3, 3, 3]
    assert set(streams) == {"mainstream", "niche", "underground"}
    # mainstream got the most-heard tracks
    assert streams["mainstream"][0]["listeners"] == 9000
    assert streams["underground"][-1]["listeners"] == 1000


def test_split_streams_keeps_score_order_within_tiers():
    ranked = _ranked(9)
    ranked[8]["score"] = 0.99  # least-heard track, but best content fit
    streams = engine.split_streams(ranked)
    # underground tier re-sorted by score: the great fit leads its tier
    assert streams["underground"][0]["title"] == "T8"
    for songs in streams.values():
        scores = [s["score"] for s in songs]
        assert scores == sorted(scores, reverse=True)


def test_split_streams_small_inputs_dont_crash():
    assert engine.split_streams([]) == {"mainstream": [], "niche": [],
                                        "underground": []}
    s = engine.split_streams(_ranked(2))
    assert len(s["mainstream"]) == 1 and len(s["underground"]) == 1


def test_split_tempo_streams_thresholds():
    ranked = [
        {"title": "Fast", "score": 0.9, "tempo": 128},
        {"title": "Mid", "score": 0.8, "tempo": 110},
        {"title": "Slow", "score": 0.7, "tempo": 85},
        {"title": "Unknown", "score": 0.6},  # no tempo: excluded, not guessed
        {"title": "Edge", "score": 0.5, "tempo": 120},
    ]
    streams = engine.split_tempo_streams(ranked)
    assert [s["title"] for s in streams["upbeat"]] == ["Fast", "Edge"]
    assert [s["title"] for s in streams["slowburn"]] == ["Slow"]
