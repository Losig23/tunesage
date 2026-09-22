# 🎧 TuneSage — AI-Powered Music Recommendations

Log the songs you listen to. TuneSage figures out your taste, mines Reddit
music forums (`r/MusicSuggestions`, `r/ifyoulikeblank`, `r/LetsTalkMusic`) for
"songs like X" threads, and ranks candidate songs with a hybrid AI engine
(content similarity + forum social proof + your feedback). Vote 👍/👎 and the
ranking weights adapt — simple online learning.

## Architecture

```
┌────────────┐   log listen    ┌──────────┐
│  Frontend  │ ─────────────▶ │  Flask   │
│ (vanilla   │ ◀───────────── │  app.py  │
│  JS SPA)   │   JSON API     └──┬───┬───┘
└────────────┘                   │   │
              ┌──────────────────┘   └──────────────────┐
              ▼                                        ▼
     ┌────────────────┐                     ┌────────────────────┐
     │ recommender.py │                     │ forum.py           │
     │ TF-IDF taste   │                     │ Reddit JSON mining │
     │ profile, cosine│◀──── candidates ────│ regex extraction,  │
     │ sim, hybrid    │                     │ 6h cache, fallback │
     │ rank, explain  │                     │ dataset on failure │
     └───────┬────────┘                     └────────────────────┘
             ▼
     ┌────────────────┐
     │ db.py (SQLite) │
     │ songs, cache,  │
     │ feedback,      │
     │ weights        │
     └────────────────┘
```

## Run locally

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python seed.py          # 15 sample listens so the demo works instantly
pytest -q               # engine unit tests
python app.py           # → http://localhost:5000
```

## API

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | health check |
| `POST` | `/api/listens` | `{title, artist, genre?, tags?[]}` → upserts, `play_count++` |
| `GET` | `/api/listens/top?limit=10` | most-listened, `play_count` desc |
| `GET` | `/api/recommendations?limit=10&refresh=0` | ranked recs (6h cache; `refresh=1` re-mines) |
| `POST` | `/api/feedback` | `{song_title, artist, liked}` → stores vote, nudges weights |
| `GET` | `/api/lastfm/status` | `{configured, username?, last_import_at?}` |
| `POST` | `/api/lastfm/import` | import up to ~200 recent scrobbles → `{imported, skipped, songs}` |
| `GET` | `/` | the dashboard |

## How the AI works

**Taste profile (content-based filtering).** Each listened song becomes a text
document (`title artist genre tags`). `TfidfVectorizer` turns these into sparse
vectors where rare, distinctive terms ("synthwave") outweigh common ones
("the"). The profile is the *play-count-weighted centroid* — the average
direction of your taste, pulled hardest by songs you play most.

**Candidate scoring.** Cosine similarity between the profile and each candidate
song's TF-IDF vector: `cos(A,B) = A·B / (‖A‖‖B‖)`. Ignores vector magnitude,
so short vs. long descriptions compare fairly.

**Forum signal.** `forum_score = 0.5·log-scaled mentions + 0.5·lexicon
sentiment` of the snippets mentioning the candidate. Log scaling gives
diminishing returns so mega-threads don't dominate.

**Hybrid rank.** `score = w_content·cos_sim + w_forum·forum_score +
w_feedback·feedback_boost`. Voting 👍/👎 nudges the weights (±0.02,
renormalized) — online learning: the system learns which signals *you* trust.

**Explainability.** `explain()` cites the overlapping taste terms, mention
count, source threads, and sentiment — every recommendation ships with a "why".

## Last.fm integration

Import your recent Last.fm scrobbles instead of logging listens by hand.
Manual logging (`POST /api/listens`) still works — both write to the same
listening history. Imports are deduplicated by play timestamp, so
re-importing never double-counts.

**Why Last.fm instead of Spotify?** Spotify's Developer Policy prohibits
using its platform or content to train machine-learning/AI models — which
is exactly what TuneSage does with listening history. Last.fm's API has no
such restriction (it's free for non-commercial use with attribution, which
covers this portfolio project). The official Last.fm scrobbler can even
scrobble from Spotify, so your Spotify plays show up as scrobbles.

**Setup (one time):**

1. Create an API account at [last.fm/api/account/create](https://www.last.fm/api/account/create)
   to get an **API key**.
2. Set environment variables (locally via `.env`/shell, on Render under the
   service's **Environment** tab):
   - `LASTFM_API_KEY` — from your Last.fm API account page
   - `LASTFM_USERNAME` — your Last.fm username
3. Open the dashboard → **Last.fm** card → **Import recent scrobbles**.

How it works: `POST /api/lastfm/import` calls Last.fm's
`user.getrecenttracks` (`lastfm.py`), paginating up to ~200 scrobbles
(newest first, staying under the ~5 req/sec guidance). Each new scrobble is
logged through the same `log_listen()` path as manual logging and recorded
in the `external_imports` table, so a later import of the same play is a
no-op.

## Deploy to Render (free)

1. Push this repo to GitHub.
2. Render Dashboard → **New +** → **Web Service** → select the repo.
   (Or: `render.yaml` enables one-click Blueprint deploy.)
3. Build command: `pip install -r requirements.txt` · Start: `gunicorn app:app`.
4. Open the service URL. Run `python seed.py` locally first if you want demo
   data — or just log listens in the UI.

> Note: SQLite lives on the container's ephemeral disk; on Render's free tier
> data resets on redeploy. For persistence, attach a Render Disk or swap
> `db.py`'s path for Postgres — the SQL is standard and portable.
