# The web app

Two surfaces over one API, switched in the top left:

- **Shop** — what a customer sees. Every rating shown as listed and again with
  suspected fakes withheld.
- **Console** — what a moderator sees. The ranked queue and the evidence behind
  each flag.

Neither reimplements any scoring, feature engineering, or decision policy of its
own. Both read the FastAPI service in `src/frd/api/`.

From the repository root:

```bash
make up            # sets up whatever is missing, then runs everything
```

Then open <http://localhost:5173>. If the project is already set up,
`make console` starts just the two servers, and `make web-setup` / `make web`
drive the app alone against an API you are running yourself.

The first request features and scores the whole corpus (~20s for 40,000
reviews) and caches it; everything after that is served from the cache. Both
surfaces read that one cache, so whichever you open first pays for it. The page
waits for the API rather than failing if it is still loading its model, which is
the normal case for a second or two after `make console`.

Needs a trained model (`make train`) and a corpus (`make dataset`) — `make up`
does both. Without either, the app says which one is missing rather than
rendering an empty page.

## Routing

A hash router in `router.ts`, thirty lines and no dependency:

| route | surface |
|---|---|
| `#/shop` | the catalogue, plus the products whose ratings move most |
| `#/shop/P00207` | one product page |
| `#/console` | the moderation queue |

Hash rather than the History API because the built bundle is served as static
files from whatever path the API or a CDN puts it under, and a hash needs no
server-side rewrite rule to survive a refresh.

## What the shop shows

**The catalogue.** Product listings — names, brands, prices, generated art —
derived from the product id by `frd.data.catalog`, because the review schema
carries an id and a category and nothing else. A card shows the cleaned rating
and carries a trust chip *only* when cleaning actually moved it.

**The product page.** The listing, the rating both ways, the rating breakdown as
listed against after withholding, and the reviews that survived. "Show what was
withheld" lists the reviews that did not, each with its score and the same
reason codes the console shows a moderator.

**Reviews the model finds suspicious but a human has not confirmed** stay
visible, marked. Only the `auto_remove` tier is withheld.

## What the console shows

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

## Things both surfaces are deliberately careful about

**Neither keeps its own copy of the decision policy.** Tier cut-points come from
the server (`/queue/stats` → `decision_policy`, `/shop/stats` →
`material_delta`) and the console's re-tiering is checked against the server's
own answer on every page; a disagreement raises a banner instead of silently
miscolouring the queue. The Python side goes to some trouble to avoid a second
implementation of anything on the serving path, and a TypeScript copy of the
thresholds would be that same mistake in a new language.

**The console defaults to the held-out period.** The model was fitted on the
first 70% of the corpus by time. Scores there are partly memorised, so a console
that opened on them would be a demo of overfitting. `all reviews` is available
and carries a banner saying so.

**The shop cannot do the same, and says so instead.** A store shows a product's
whole history; it has the reviews it has. So every review carries
`in_train_period` and fitted-period ones are tagged wherever they appear.

## Styling

One stylesheet, two palettes. The console is dark and looks like an operator
tool; the storefront is light and looks like a store. Every storefront rule is
scoped under `.shop` and its tokens (`--s-*`) are declared there, so the two
cannot bleed into each other. This is not hypothetical tidiness: the shop's
rating histogram originally reused `.hist`, which the console sidebar's score
sparkline already owned as `display: flex`, and the shop's panel rendered
sideways. It is `.rhist` now, and the scoping is what stops the next one.

## Layout

```
src/
  api.ts                    fetch wrapper; surfaces FastAPI's `detail` on errors
  types.ts                  mirrors the server payloads
  lib.ts                    tiering, formatting, localStorage verdicts
  router.ts                 hash router
  App.tsx                   the shell: route switch, health/wait, footer
  shop/
    Storefront.tsx          the finding, the movers lane, the catalogue grid
    ProductPage.tsx         listing, reviews, rating breakdown
    TrustPanel.tsx          the rating both ways + the withheld audit trail
    bits.tsx                stars, money, generated product art
  console/
    Console.tsx             state, data fetching, race-safe pagination
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
than obviously not doing it. There is no auth, and both surfaces are read-only
against the corpus — nothing here writes back.

Nothing in the shop is for sale. `Add to basket` is inert, there is no cart, and
no listing describes a real product: names and prices are generated from the
product id, and no real brand is named. Attaching a fabricated fraud score to a
real company's product is what `docs/ROADMAP.md` phase 10 is about.
