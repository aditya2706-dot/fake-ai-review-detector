# The web app

Two surfaces over one API:

- **Shop** (`#/`) — what a customer sees. A browsable catalogue with accounts,
  a basket and a checkout, where every rating is shown as listed and again with
  suspected fakes withheld.
- **Console** (`#/console`) — what a moderator sees. The ranked queue and the
  evidence behind each flag.

Neither reimplements any scoring, feature engineering, or decision policy of its
own. Both read the FastAPI service in `src/frd/api/`.

From the repository root:

```bash
make up            # sets up whatever is missing, then runs everything
```

Then open <http://localhost:5173>. If the project is already set up,
`make app` starts just the two servers, and `make web-setup` / `make web`
drive the app alone against an API you are running yourself.

The first request features and scores the whole corpus (~20s for 40,000
reviews) and caches it; everything after that is served from the cache. Both
surfaces read that one cache, so whichever you open first pays for it. The page
waits for the API rather than failing if it is still loading its model, which is
the normal case for a second or two after `make app`.

Needs a trained model (`make train`) and a corpus (`make dataset`) — `make up`
does both. Without either, the app says which one is missing rather than
rendering an empty page.

## Routing

A hash router in `router.ts`, thirty lines and no dependency:

| route | page |
|---|---|
| `#/` | the shop front: departments, new in, the ratings we corrected |
| `#/c/home_kitchen` | a department listing, with facets and sorting |
| `#/search?q=kettle` | search results, same component as a department |
| `#/p/ferro-dutch-oven` | a product page |
| `#/cart` | the basket |
| `#/checkout` | address and place order |
| `#/order/ORD-1A2B3C4D` | one order |
| `#/orders`, `#/account` | order history, account |
| `#/signin` | sign in / create an account |
| `#/console` | the moderation queue |

Products are addressed by **slug**, not by corpus id: `/p/ferro-dutch-oven` is
a URL a shop can publish, `/p/P00782` is an internal identifier that happens to
be routable.

Hash rather than the History API because the built bundle is served as static
files from whatever path the API or a CDN puts it under, and a hash needs no
server-side rewrite rule to survive a refresh.

## What the shop shows

**The catalogue.** 61 hand-authored listings from `src/frd/data/products.json`
— copy, specifications, variants, stock and photography — attached to corpus
products by `frd.data.catalog.assign`. A tile shows the cleaned rating and
carries a trust note *only* when cleaning actually moved it.

**The product page.** The listing and its variants, the buy box, the rating
both ways, the rating breakdown after withholding, and the reviews that
survived. The disclosure under them lists the reviews that did not, struck
through, each with its score and the same reason codes the console shows a
moderator.

**The basket and checkout.** A real basket that survives signing in, and a
checkout that records an order against the account. Every price and total comes
from the server; the client sends a slug, a variant and a quantity, never a
price. No card details are asked for — see the root README.

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

Two stylesheets, two palettes. `styles.css` is the console: dark, dense,
monospaced numbers, and it looks like an operator tool. `shop.css` is the shop:
light, roomy, quiet. Every shop rule is scoped under `.shop` and its tokens are
declared there, so the two cannot bleed into each other. This is not
hypothetical tidiness — when they shared one file, the shop's rating histogram
reused a class the console sidebar's sparkline already owned as
`display: flex`, and the shop's panel rendered sideways.

The quiet is load-bearing. The shop's one unusual claim is that a rating can be
wrong, and a page that shouts everywhere has nothing left for the products where
that matters.

## Layout

```
src/
  api.ts                    fetch wrapper; credentials: include for the session cookie
  types.ts                  mirrors the server payloads
  lib.ts                    tiering, formatting, localStorage verdicts
  router.ts                 hash router, ten routes
  App.tsx                   the shell: route switch, health/wait
  styles.css                console theme      shop.css  shop theme
  state/
    shop.tsx                one provider: who is signed in + what is in the basket
  shop/
    Layout.tsx              header, department nav, search, footer
    Home.tsx                departments, new in, the ratings we corrected
    Listing.tsx             a department or a search, one component
    Product.tsx             listing, variants, buy box, reviews, withheld
    TrustPanel.tsx          the rating both ways + the withheld audit trail
    Cart.tsx                basket and the shared order summary
    Checkout.tsx            address, place order
    Orders.tsx              confirmation and order history
    Account.tsx             sign in, register, account
    bits.tsx                stars, money, photos, product tiles
  console/
    Console.tsx             state, data fetching, race-safe pagination
    Sidebar.tsx             tier counts, corpus facts, score histogram
    ThresholdPanel.tsx      the slider and its cost readout
    QueueTable.tsx          the ranked list
    DetailDrawer.tsx        evidence: reasons, SHAP, tokens, neighbours
public/
  products/                 the catalogue's photography + CREDITS.md
```

## Configuration

| Variable | Where | Default |
|---|---|---|
| `FRD_API_URL` | dev server proxy target | `http://127.0.0.1:8000` |
| `VITE_API_BASE` | API base for a built bundle | `/api` |
| `FRD_CORS_ORIGINS` | allowed origins, on the API | `http://localhost:5173,http://127.0.0.1:5173` |
| `FRD_DATA_PATH` | corpus, on the API | `data/processed/reviews.{parquet,csv}` |
| `FRD_STORE_DB` | shop database, on the API | `data/store.db` |
| `FRD_AUTH` | identity provider | `local` (or `firebase`) |
| `FRD_SECURE_COOKIES` | mark cookies Secure (needs https) | off |

## Not built, on purpose

**Moderator verdicts live in `localStorage`.** A real console records decisions
durably with the reason codes attached and exposes them to an appeals process
(ROADMAP phase 9); faking that with a browser-local JSON blob would be worse
than obviously not doing it. The console has no auth of its own, and it is
read-only against the corpus — nothing there writes back.

**Nothing in the shop is really for sale.** Orders are recorded and no payment
is taken; the checkout deliberately has no card field. No listing describes a
real product: the brands are invented, and every photograph was chosen so that
no real brand is legible in it. Attaching a fabricated fraud score to a real
company's product is what `docs/ROADMAP.md` phase 10 is about.

**Shoppers cannot write reviews yet.** Every review in the shop comes from the
corpus. Letting a signed-in customer post one — and watching the detector score
it live — is the obvious next step, and it is the reason accounts and orders
exist at all.
