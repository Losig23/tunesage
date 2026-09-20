"""AI recommendation engine for TuneSage.

Pipeline math (interview-friendly summary):
  1. build_taste_profile(): each listened song becomes a text document
     "title artist genre tag1 tag2". TF-IDF turns documents into sparse vectors
     where rare, distinctive words (e.g. "synthwave") weigh more than common
     ones ("the"). The profile is the play-count-weighted centroid (mean) of
     the user's song vectors — i.e. "the average direction of their taste".
  2. score_candidates(): cosine similarity between the profile vector and each
     candidate's TF-IDF vector. Cosine sim = (A·B)/(||A||·||B||), i.e. the
     cosine of the angle between vectors; 1 = same direction, 0 = orthogonal.
     Crucially it ignores magnitude, so long vs. short descriptions compare
     fairly.
  3. forum_score(): social proof — log-scaled mention frequency × lexicon
     sentiment of the forum snippets mentioning the candidate.
  4. hybrid_rank(): weighted fusion of the signals; weights adapt via user
     feedback (online learning, see db.nudge_weights).
"""

from __future__ import annotations

import math
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Small positive/negative lexicon for snippet sentiment. Real systems train a
# classifier; a lexicon is transparent and dependency-free — fine for a demo.
POSITIVE_WORDS = {
    "love", "loved", "amazing", "incredible", "masterpiece", "perfect",
    "beautiful", "gorgeous", "fantastic", "brilliant", "awesome", "great",
    "excellent", "obsessed", "classic", "underrated", "gem", "chills",
    "powerful", "haunting", "catchy", "soulful", "recommend", "favorite",
    "favourite", "best", "stunning", "goosebumps",
}
NEGATIVE_WORDS = {
    "hate", "hated", "terrible", "awful", "boring", "worst", "bad",
    "overrated", "annoying", "skip", "disappointing", "meh", "bland",
    "repetitive", "cringe",
}


def song_document(song: dict) -> str:
    """Flatten a song dict into one text document for TF-IDF."""
    tags = song.get("tags") or []
    if isinstance(tags, str):
        tags = [tags]
    parts = [song.get("title", ""), song.get("artist", ""),
             song.get("genre", ""), " ".join(tags)]
    return " ".join(p for p in parts if p).lower()


def build_taste_profile(listens: list[dict]):
    """Return (vectorizer, profile_vector): play-count-weighted TF-IDF centroid.

    Weighting by play_count means a song you played 50x pulls the centroid
    toward its vocabulary more than a song you played once — "most listened"
    literally shapes taste.
    """
    if not listens:
        return None, None
    docs = [song_document(s) for s in listens]
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(docs)              # (n_songs, n_terms)
    counts = np.array([max(1, int(s.get("play_count", 1))) for s in listens],
                      dtype=float)
    weights = counts / counts.sum()
    centroid = weights @ matrix.toarray()                # weighted mean vector
    norm = np.linalg.norm(centroid)
    if norm > 0:
        centroid = centroid / norm                        # unit length
    return vectorizer, centroid


def score_candidates(candidates: list[dict], vectorizer,
                     profile_vector) -> list[float]:
    """Cosine similarity of each candidate doc against the taste profile."""
    if vectorizer is None or profile_vector is None or not candidates:
        return [0.0] * len(candidates)
    docs = [song_document(c) for c in candidates]
    cand_matrix = vectorizer.transform(docs)
    sims = cosine_similarity(cand_matrix,
                             profile_vector.reshape(1, -1)).ravel()
    # Clip tiny negative float noise to 0 for clean display.
    return [float(max(0.0, s)) for s in sims]


def snippet_sentiment(snippet: str) -> float:
    """Lexicon sentiment in [-1, 1]: (pos - neg) / total sentiment words."""
    words = re.findall(r"[a-z']+", snippet.lower())
    pos = sum(1 for w in words if w in POSITIVE_WORDS)
    neg = sum(1 for w in words if w in NEGATIVE_WORDS)
    total = pos + neg
    if total == 0:
        return 0.0
    return (pos - neg) / total


def forum_score(candidate: dict) -> float:
    """Social-proof score in [0, 1].

    frequency component: log1p(mentions) — log scale so the 50th mention
      matters less than the 2nd (diminishing returns, avoids mega-threads
      dominating).
    sentiment component: mean lexicon sentiment of snippets, mapped [-1,1]
      -> [0,1].
    Combined 50/50.
    """
    mentions = max(1, int(candidate.get("mentions", 1)))
    freq = math.log1p(mentions) / math.log1p(10)   # normalize ~[0,1] for <=10
    freq = min(1.0, freq)
    snippets = candidate.get("snippets") or []
    if snippets:
        avg_sent = sum(snippet_sentiment(s) for s in snippets) / len(snippets)
    else:
        avg_sent = 0.0
    sentiment = (avg_sent + 1.0) / 2.0             # [-1,1] -> [0,1]
    return round(0.5 * freq + 0.5 * sentiment, 4)


def feedback_boost(candidate: dict, fb_counts: dict) -> float:
    """Feedback signal in [-1, 1]: net likes normalized by total feedback.

    Positive if the user liked this (or a same-titled) song before, negative
    if they disliked it. Zero when there's no feedback — the cold-start-safe
    default.
    """
    key = (candidate.get("title", "").lower(),
           candidate.get("artist", "").lower())
    fb = fb_counts.get(key)
    if not fb or fb["total"] == 0:
        return 0.0
    return (fb["likes"] - fb["dislikes"]) / fb["total"]


def hybrid_rank(candidates: list[dict], content_scores: list[float],
                weights: dict, fb_counts: dict | None = None) -> list[dict]:
    """Weighted fusion + sort. Returns candidates annotated with scores.

    final = w_content * cos_sim
          + w_forum   * forum_score
          + w_feedback * (0.5 + 0.5 * feedback_boost)   # mapped to [0,1]

    Mapping feedback_boost from [-1,1] to [0,1] keeps every term on the same
    scale so the weights stay interpretable.
    """
    fb_counts = fb_counts or {}
    ranked = []
    for cand, cos_sim in zip(candidates, content_scores):
        f_score = forum_score(cand)
        fb = feedback_boost(cand, fb_counts)
        fb_term = 0.5 + 0.5 * fb
        final = (weights["w_content"] * cos_sim
                 + weights["w_forum"] * f_score
                 + weights["w_feedback"] * fb_term)
        ranked.append({**cand,
                       "content_score": round(float(cos_sim), 4),
                       "forum_score": round(f_score, 4),
                       "feedback_boost": round(float(fb), 4),
                       "score": round(float(final), 4)})
    ranked.sort(key=lambda c: c["score"], reverse=True)
    return ranked


def explain(candidate: dict, profile_terms: list[str]) -> str:
    """Human-readable 'why this recommendation'.

    Cites: (a) which of the user's taste-profile terms overlap the candidate,
    (b) how many forum threads mentioned it and with what sentiment vibe,
    (c) any past feedback. Keep it to 1–2 sentences for the UI.
    """
    doc_words = set(re.findall(r"[a-z']+", song_document(candidate)))
    overlap = [t for t in profile_terms if t in doc_words][:4]
    mentions = candidate.get("mentions", 1)
    sources = candidate.get("sources") or []
    src = f" (e.g. {sources[0]})" if sources else ""

    reasons = []
    if overlap:
        reasons.append(f"matches your taste for {', '.join(overlap)}")
    if mentions > 1:
        reasons.append(f"mentioned in {mentions} forum threads{src}")
    elif mentions == 1:
        reasons.append(f"suggested in a forum thread{src}")
    avg_sent = 0.0
    snippets = candidate.get("snippets") or []
    if snippets:
        avg_sent = sum(snippet_sentiment(s) for s in snippets) / len(snippets)
    if avg_sent > 0.3:
        reasons.append("forum sentiment is strongly positive")
    if candidate.get("feedback_boost", 0) > 0:
        reasons.append("you've liked similar picks before")
    if not reasons:
        reasons.append("broadly popular in recommendation threads")
    why = "; ".join(reasons)
    return f"Recommended because it {why}."


def top_profile_terms(vectorizer, profile_vector, n: int = 12) -> list[str]:
    """The n highest-weighted terms in the taste centroid — used by explain()."""
    if vectorizer is None or profile_vector is None:
        return []
    terms = np.array(vectorizer.get_feature_names_out())
    idx = np.argsort(profile_vector)[::-1][:n]
    return [t for t in terms[idx] if profile_vector[terms.tolist().index(t)] > 0]
