"""Shopper-facing storefront over the scored corpus.

WHAT THIS SURFACE IS FOR
------------------------
The moderation console (`queue.py`) answers "which reviews should a human look
at". This answers the question a *shopper* has: given that some of these
reviews are paid, what is this product actually rated?

THE THREE THINGS IT IS CAREFUL ABOUT
------------------------------------
1. **It withholds at the auto-remove tier, not the queue threshold.** The
   decision policy already draws two lines (`frd.explain._decision`): above
   the queue threshold a human should look, above the auto-remove line the
   platform is willing to act unattended. A storefront that hid everything
   above the *queue* line would be removing reviews at 87.7% precision with no
   moderator involved -- one wrongly suppressed review in eight. The
   auto-remove line runs at 94.2%, one in seventeen. Withholding at the
   stricter line roughly halves that rate and, more to the point, is the bar
   the platform had already accepted for acting unattended; anything looser is
   the cost asymmetry `docs/METHODOLOGY.md` §2 says not to accept silently. So
   `auto_remove` is withheld; `queue_for_review` stays visible and is merely
   counted. (Both figures are held-out precision at those thresholds;
   `/queue/stats` recomputes them for whatever corpus is loaded.)

2. **It does not own a second copy of the rating maths or the policy.** Tiers
   come from `frd.explain._decision`, reached through the console's
   `_row_payload` rather than reimplemented here, and scores come from the
   corpus cache the console already built. `MATERIAL_DELTA` is served to the client rather than duplicated in
   TypeScript, for the same reason the console is handed `decision_policy`.

3. **It reports the whole product history, and says so.** A store cannot show
   only the held-out period -- a product has the reviews it has. But the model
   was fitted on the first 70% of the corpus by time, so scores on older
   reviews are partly memorised and the withheld set there is flattered.
   Every review carries `in_train_period` and every product carries
   `train_period_ends` so the UI can mark it instead of implying the numbers
   are all equally honest.

WHAT THE NUMBERS ACTUALLY DO ON THIS CORPUS
-------------------------------------------
Cleaning moves the median product's rating by -0.004 stars. That is not a bug
and not worth hiding: simulated fakes average 3.84 stars against genuine 3.70,
because 22% of campaigns are one-star competitor attacks and sophisticated
farms mix in 3s and 4s deliberately. The effect is concentrated -- ~4% of
products move half a star or more -- and it runs in *both* directions. The
biggest movers include products rescued from an attack campaign, not just
products deflated from an inflated one. `direction` says which, and
`/shop/stats` ranks them, because that concentration is the finding.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from ..data.catalog import catalog_for
from ..explain import auto_remove_at
from ..explain import explain as explain_reviews
from ..schema import LABEL, PRODUCT_ID, TIMESTAMP
from .queue import _row_payload, corpus

router = APIRouter(prefix="/shop", tags=["storefront"])

# A rating delta smaller than this is not shown to a shopper. Printing
# "4.6 -> 4.6" on the 58% of products where nothing moved trains the eye to
# ignore the badge, which costs credibility on the 4% where it matters.
MATERIAL_DELTA = 0.1

# Products with almost no reviews have a meaningless mean rating; a storefront
# still lists them, but they are kept out of the "most manipulated" ranking
# where a two-review product would otherwise top the table on noise.
MIN_REVIEWS_FOR_RANKING = 20

# A cleaned rating computed from a handful of survivors is arithmetically fine
# and epistemically weak: P00207 in this corpus loses 46 of 49 reviews, so its
# "real" rating is the mean of three. The number is still worth showing -- it
# is the best estimate available -- but it must not be printed in the same
# weight as one standing on three hundred reviews, so it is flagged.
MIN_SHOWN_FOR_CONFIDENCE = 10

CURRENCY = "USD"


def _aggregate(store: dict[str, Any], threshold: float) -> pd.DataFrame:
    """One row per product: raw rating, cleaned rating, and the tier counts.

    Cached per threshold, inside the corpus store rather than in a module
    global. The cache therefore lives and dies with the frame it describes: a
    rebuilt corpus gets a new store dict and cannot be served a stale
    aggregate. An earlier version keyed a module-level dict on `id(frame)` and
    was wrong, because CPython reuses an id once the object behind it is
    collected -- a fresh frame could hit the previous frame's entry.
    """
    key = round(float(threshold), 4)
    cache = store.setdefault("agg_cache", {})
    hit = cache.get(key)
    if hit is not None:
        return hit

    df = store["frame"]
    remove_at = auto_remove_at(threshold)
    withheld_mask = df["score"].to_numpy(float) >= remove_at
    queued_mask = (~withheld_mask) & (df["score"].to_numpy(float) >= threshold)

    work = pd.DataFrame({
        PRODUCT_ID: df[PRODUCT_ID].astype(str).to_numpy(),
        "category": df["category"].astype(str).to_numpy(),
        "rating": df["rating"].to_numpy(float),
        "withheld": withheld_mask,
        "queued": queued_mask,
    })

    g = work.groupby(PRODUCT_ID, sort=True)
    agg = pd.DataFrame({
        "n_reviews": g.size(),
        "raw_rating": g["rating"].mean(),
        "n_withheld": g["withheld"].sum().astype(int),
        "n_under_review": g["queued"].sum().astype(int),
        "category": g["category"].first(),
    })

    shown = work[~work["withheld"]]
    shown_g = shown.groupby(PRODUCT_ID, sort=True)
    agg["n_shown"] = shown_g.size().reindex(agg.index).fillna(0).astype(int)
    # A product whose every review was withheld has no cleaned rating to show.
    # NaN, not the raw rating: claiming "unchanged" there would be a lie.
    agg["adjusted_rating"] = shown_g["rating"].mean().reindex(agg.index)

    agg["delta"] = agg["adjusted_rating"] - agg["raw_rating"]
    agg["adjusted_thin"] = agg["n_shown"] < MIN_SHOWN_FOR_CONFIDENCE
    agg["withheld_share"] = agg["n_withheld"] / agg["n_reviews"]
    material = agg["delta"].abs() >= MATERIAL_DELTA
    agg["material"] = material.fillna(False)
    agg["direction"] = np.where(
        ~agg["material"], "unchanged",
        np.where(agg["delta"] < 0, "inflated", "attacked"),
    )

    if LABEL in df.columns:
        lab = pd.DataFrame({
            PRODUCT_ID: df[PRODUCT_ID].astype(str).to_numpy(),
            LABEL: pd.to_numeric(df[LABEL], errors="coerce").to_numpy(float),
        })
        agg["true_fake_share"] = lab.groupby(PRODUCT_ID)[LABEL].mean()

    cache.clear()               # one threshold's worth is all that is worth holding
    cache[key] = agg
    return agg


def _catalog(store: dict[str, Any]) -> dict:
    if "catalog" not in store:
        df = store["frame"]
        pairs = df[[PRODUCT_ID, "category"]].drop_duplicates(PRODUCT_ID)
        store["catalog"] = catalog_for(
            pairs[PRODUCT_ID].astype(str), pairs["category"].astype(str),
            currency=CURRENCY,
        )
    return store["catalog"]


def _f(v: Any, digits: int = 2) -> float | None:
    """Round for display, preserving 'no answer' as null rather than as 0.0."""
    if v is None or (isinstance(v, float) and np.isnan(v)) or pd.isna(v):
        return None
    return round(float(v), digits)


def _card(pid: str, row: pd.Series, catalog: dict) -> dict[str, Any]:
    prod = catalog[pid]
    out = prod.to_dict()
    out.update({
        "n_reviews": int(row["n_reviews"]),
        "n_shown": int(row["n_shown"]),
        "n_withheld": int(row["n_withheld"]),
        "n_under_review": int(row["n_under_review"]),
        "raw_rating": _f(row["raw_rating"]),
        "adjusted_rating": _f(row["adjusted_rating"]),
        "delta": _f(row["delta"]),
        "withheld_share": _f(row["withheld_share"], 4),
        "material": bool(row["material"]),
        "direction": str(row["direction"]),
        "adjusted_thin": bool(row["adjusted_thin"]),
    })
    if "true_fake_share" in row.index:
        out["true_fake_share"] = _f(row["true_fake_share"], 4)
    return out


def _threshold(det, threshold: float | None) -> float:
    return det.threshold_ if threshold is None else float(threshold)


@router.get("/stats")
def stats(threshold: float | None = Query(None, ge=0.0, le=1.0)) -> dict[str, Any]:
    """Catalogue-wide facts, plus the products where cleaning changes the answer.

    `movers` is the demo: ranked by absolute rating change, both directions,
    restricted to products with enough reviews for the mean to mean anything.
    """
    from .main import _detector  # noqa: PLC0415  (avoids a circular import)

    det = _detector()
    store = corpus(det)
    thr = _threshold(det, threshold)
    agg = _aggregate(store, thr)
    catalog = _catalog(store)

    rankable = agg[agg["n_reviews"] >= MIN_REVIEWS_FOR_RANKING]
    order = rankable["delta"].abs().sort_values(ascending=False, kind="mergesort")
    movers = [_card(pid, rankable.loc[pid], catalog) for pid in order.index[:12]]

    counts = {
        "n_products": len(agg),
        "n_rankable": len(rankable),
        "n_material": int(rankable["material"].sum()),
        "n_inflated": int((rankable["direction"] == "inflated").sum()),
        "n_attacked": int((rankable["direction"] == "attacked").sum()),
    }

    return {
        "counts": counts,
        "categories": sorted(agg["category"].dropna().unique().tolist()),
        "currency": CURRENCY,
        "threshold": thr,
        "auto_remove_at": auto_remove_at(thr),
        # Served, not duplicated client-side -- same discipline as the
        # console's `decision_policy`.
        "material_delta": MATERIAL_DELTA,
        "min_reviews_for_ranking": MIN_REVIEWS_FOR_RANKING,
        "min_shown_for_confidence": MIN_SHOWN_FOR_CONFIDENCE,
        "n_reviews": int(agg["n_reviews"].sum()),
        "n_withheld": int(agg["n_withheld"].sum()),
        "train_period_ends": pd.Timestamp(store["train_end"]).isoformat(),
        "median_delta": _f(rankable["delta"].median(), 4),
        "movers": movers,
    }


@router.get("/products")
def products(
    limit: int = Query(24, ge=1, le=120),
    offset: int = Query(0, ge=0),
    category: str | None = None,
    search: str | None = None,
    min_reviews: int = Query(0, ge=0),
    sort: str = Query(
        "reviews",
        pattern="^(reviews|manipulation|rating|rating_asc|price|price_desc|name)$",
    ),
    threshold: float | None = Query(None, ge=0.0, le=1.0),
) -> dict[str, Any]:
    """One page of the catalogue."""
    from .main import _detector  # noqa: PLC0415

    det = _detector()
    store = corpus(det)
    thr = _threshold(det, threshold)
    agg = _aggregate(store, thr)
    catalog = _catalog(store)

    view = agg
    if category:
        view = view[view["category"] == category]
    if min_reviews:
        view = view[view["n_reviews"] >= min_reviews]
    if search:
        needle = search.lower()
        keep = [
            pid for pid in view.index
            if needle in catalog[pid].name.lower()
            or needle in catalog[pid].brand.lower()
            or needle in pid.lower()
        ]
        view = view.loc[keep]

    prices = pd.Series(
        {pid: catalog[pid].price for pid in view.index}, dtype=float
    ) if len(view) else pd.Series(dtype=float)

    if sort == "reviews":
        order = view["n_reviews"].sort_values(ascending=False, kind="mergesort").index
    elif sort == "manipulation":
        rankable = view[view["n_reviews"] >= MIN_REVIEWS_FOR_RANKING]
        order = rankable["delta"].abs().sort_values(
            ascending=False, kind="mergesort").index
    elif sort == "rating":
        order = view["adjusted_rating"].sort_values(
            ascending=False, kind="mergesort", na_position="last").index
    elif sort == "rating_asc":
        order = view["adjusted_rating"].sort_values(
            ascending=True, kind="mergesort", na_position="last").index
    elif sort == "price":
        order = prices.sort_values(kind="mergesort").index
    elif sort == "price_desc":
        order = prices.sort_values(ascending=False, kind="mergesort").index
    else:
        order = pd.Series(
            {pid: catalog[pid].name for pid in view.index}, dtype=object
        ).sort_values(kind="mergesort").index

    total = len(order)
    page = order[offset:offset + limit]
    items = [_card(pid, view.loc[pid], catalog) for pid in page]

    return {
        "items": items,
        "total": total,
        "offset": offset,
        "limit": limit,
        "sort": sort,
        "threshold": thr,
        "material_delta": MATERIAL_DELTA,
        "currency": CURRENCY,
    }


def _reviews_payload(
    rows: pd.DataFrame, det, threshold: float, train_end, with_reasons: bool,
) -> list[dict[str, Any]]:
    items = [_row_payload(r, threshold) for _, r in rows.iterrows()]
    for item, (_, row) in zip(items, rows.iterrows()):
        item["in_train_period"] = bool(pd.Timestamp(row[TIMESTAMP]) <= train_end)
    # Tree-SHAP over a page is cheap; over a 10,000-review product it is not.
    # So reasons are computed for the page being returned, never precomputed.
    if with_reasons and len(rows):
        for item, exp in zip(items, explain_reviews(det, rows)):
            item["reasons"] = exp.reasons
    return items


@router.get("/products/{product_id}")
def product_detail(
    product_id: str,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    withheld_limit: int = Query(12, ge=0, le=50),
    sort: str = Query("helpful", pattern="^(helpful|newest|oldest|rating|rating_asc)$"),
    threshold: float | None = Query(None, ge=0.0, le=1.0),
) -> dict[str, Any]:
    """A product page: catalogue metadata, the rating both ways, and the reviews.

    `reviews` is what a shopper sees. `withheld` is the audit trail for the
    "why?" panel -- the reviews this page is not showing, with the moderator
    reason codes attached, so the claim is inspectable rather than asserted.
    """
    from .main import _detector  # noqa: PLC0415

    det = _detector()
    store = corpus(det)
    thr = _threshold(det, threshold)
    agg = _aggregate(store, thr)
    catalog = _catalog(store)

    pid = str(product_id)
    if pid not in agg.index:
        raise HTTPException(status_code=404, detail=f"no product {pid} in corpus")

    df = store["frame"]
    rows = df[df[PRODUCT_ID].astype(str) == pid]
    remove_at = auto_remove_at(thr)
    is_withheld = rows["score"].to_numpy(float) >= remove_at

    shown, hidden = rows[~is_withheld], rows[is_withheld]

    if sort == "helpful":
        shown = shown.sort_values(
            ["helpful_votes", TIMESTAMP], ascending=[False, False], kind="mergesort")
    elif sort == "newest":
        shown = shown.sort_values(TIMESTAMP, ascending=False, kind="mergesort")
    elif sort == "oldest":
        shown = shown.sort_values(TIMESTAMP, ascending=True, kind="mergesort")
    elif sort == "rating":
        shown = shown.sort_values("rating", ascending=False, kind="mergesort")
    else:
        shown = shown.sort_values("rating", ascending=True, kind="mergesort")

    hidden = hidden.sort_values("score", ascending=False, kind="mergesort")
    train_end = pd.Timestamp(store["train_end"])

    def histogram(frame: pd.DataFrame) -> dict[str, int]:
        counts = frame["rating"].round().astype(int).value_counts()
        return {str(star): int(counts.get(star, 0)) for star in range(5, 0, -1)}

    return {
        "product": _card(pid, agg.loc[pid], catalog),
        "rating_histogram": {"raw": histogram(rows), "shown": histogram(shown)},
        "reviews": _reviews_payload(
            shown.iloc[offset:offset + limit], det, thr, train_end,
            with_reasons=False),
        "withheld": _reviews_payload(
            hidden.iloc[:withheld_limit], det, thr, train_end,
            with_reasons=True),
        "total_shown": len(shown),
        "total_withheld": len(hidden),
        "offset": offset,
        "limit": limit,
        "sort": sort,
        "threshold": thr,
        "auto_remove_at": remove_at,
        "material_delta": MATERIAL_DELTA,
        "currency": CURRENCY,
        "train_period_ends": train_end.isoformat(),
    }
