# 🎧 TuneSage — AI-Powered Music Recommendations

Log the songs you listen to. TuneSage figures out your taste, pulls candidate
songs from the Last.fm API (top tracks of artists similar to your most-played
artists), and ranks them with a hybrid AI engine (content similarity + crowd
social proof + your feedback). Vote 👍/👎 and the ranking weights adapt —
simple online learning.

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
     │ TF-IDF taste   │                     │ Last.fm candidate  │
     │ profile, cosine│◀──── candidates ────│ mining: similar    │
     │ sim, hybrid    │                     │ artists → top      │
     │ rank, explain  │                     │ tracks, 6h cache,  │
     └───────┬────────┘                     │ fallback dataset   │
             ▼                              │ on failure         │
     ┌────────────────┐                     └────────────────────┘
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

**Crowd signal.** `forum_score = 0.5·log-scaled mentions + 0.5·lexicon
sentiment`. Candidates come from Last.fm (`artist.getSimilar` →
`artist.getTopTracks`); raw listener counts are log10-bucketed onto the old
mention-count scale so mega-hits don't pin the score at max. (This module
originally mined Reddit's public JSON API, but Reddit now 403s
unauthenticated script clients — the offline fallback dataset remains for
when the API is unreachable.)

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
2. Set environment variables:
   - **Locally:** copy `.env.example` to `.env` and fill in your values
     (`cp .env.example .env`):
     - `LASTFM_API_KEY` — from your Last.fm API account page
     - `LASTFM_USERNAME` — your Last.fm username

     `.env` is gitignored, so your key never gets committed. The app loads
     it automatically via `python-dotenv`; real environment variables
     (shell exports, CI, Render) still take precedence over `.env` values.
   - **On Render:** set them under the service's **Environment** tab.
   - **Alternative:** export them in your shell profile instead
     (`~/.bashrc` / `~/.zshrc`):
     ```bash
     export LASTFM_API_KEY="your-key-here"
     export LASTFM_USERNAME="your-lastfm-username"
     ```
3. Open the dashboard → **Last.fm** card → **Import recent scrobbles**.

How it works: `POST /api/lastfm/import` calls Last.fm's
`user.getrecenttracks` (`lastfm.py`), paginating up to ~200 scrobbles
(newest first, staying under the ~5 req/sec guidance). Each new scrobble is
logged through the same `log_listen()` path as manual logging and recorded
in the `external_imports` table, so a later import of the same play is a
no-op.

### Tempo data (GetSongBPM)

The ⚡ Upbeat / 🌙 Slow burn streams need per-track BPM, which Last.fm
doesn't provide. Tempo comes from [GetSongBPM](https://getsongbpm.com)
(`bpm.py`; set `GETSONGBPM_API_KEY` in `.env` — free key at
[getsongbpm.com/api](https://getsongbpm.com/api)). Lookups are cached in
SQLite forever (BPM never changes), so each unique track costs exactly one
API call. Tempo data by [GetSongBPM](https://getsongbpm.com), whose free
tier requires this backlink — it's also in the dashboard footer.

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
