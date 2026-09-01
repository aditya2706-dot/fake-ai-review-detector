# Moderation console

A triage queue for the fake-review detector. It consumes the FastAPI service in
`src/frd/api/` — it does not reimplement any scoring, feature engineering, or
decision policy of its own.

From the repository root:

```bash
make up            # sets up whatever is missing, then runs everything
```

Then open <http://localhost:5173>. If the project is already set up,
`make console` starts just the two servers, and `make web-setup` / `make web`
drive the console alone against an API you are running yourself.

The first request features and scores the whole corpus (~20s for 40,000
reviews) and caches it; everything after that is served from the cache. The
page waits for the API rather than failing if it is still loading its model,
which is the normal case for a second or two after `make console`.

Needs a trained model (`make train`) and a corpus (`make dataset`) — `make up`
does both. Without either, the console says which one is missing rather than
rendering an empty queue.

## What it shows

**The queue.** Reviews ranked by calibrated P(fake), coloured by decision tier,
with the moderator-facing reason codes on each row. Filter by tier, category,
or free text; sort by score or date.

**The operating point.** A threshold slider that re-tiers the queue live and
states the cost of the choice in ops terms: precision, recall, how many reviews
land in front of a human, and how many of those are innocent. That last number
is the one this project treats as expensive, so it is the one in red.

**The evidence.** Clicking a row opens the full case: the review with stage A's
per-token contributions painted onto the text, curated reason codes, tree-SHAP
feature contributions, and — the part that makes a flag auditable — the
author's earlier reviews and the product's earlier reviews, drawn strictly from
before the review under inspection.

## Two things it is deliberately careful about

**It defaults to the held-out period.** The model was fitted on the first 70% of
the corpus by time. Scores there are partly memorised, so a console that opened
on them would be a demo of overfitting. `all reviews` is available and carries a
banner saying so.

**It does not keep its own copy of the decision policy.** The tier cut-points
come from the server (`/queue/stats` → `decision_policy`) and the client's
re-tiering is checked against the server's own answer on every page; a
disagreement raises a banner instead of silently miscolouring the queue. The
Python side goes to some trouble to avoid a second implementation of anything on
the serving path, and a TypeScript copy of the thresholds would be that same
mistake in a new language.

## Layout

```
src/
  api.ts                    fetch wrapper; surfaces FastAPI's `detail` on errors
  types.ts                  mirrors the server payloads
  lib.ts                    tiering, formatting, localStorage verdicts
  App.tsx                   state, data fetching, race-safe pagination
  components/
    Sidebar.tsx             tier counts, corpus facts, score histogram
    ThresholdPanel.tsx      the slider and its cost readout
    QueueTable.tsx          the ranked list
    DetailDrawer.tsx        evidence: reasons, SHAP, tokens, neighbours
```

## Configuration

| Variable | Where | Default |
|---|---|---|
| `FRD_API_URL` | dev server proxy target | `http://127.0.0.1:8000` |
| `VITE_API_BASE` | API base for a built bundle | `/api` |
| `FRD_CORS_ORIGINS` | allowed origins, on the API | `http://localhost:5173,http://127.0.0.1:5173` |
| `FRD_DATA_PATH` | corpus, on the API | `data/processed/reviews.{parquet,csv}` |

## Not built, on purpose

Moderator verdicts live in `localStorage`. A real console records decisions
durably with the reason codes attached and exposes them to an appeals process
(ROADMAP phase 9); faking that with a browser-local JSON blob would be worse
than obviously not doing it. There is no auth, and the queue is read-only
against the corpus — nothing here writes back.
