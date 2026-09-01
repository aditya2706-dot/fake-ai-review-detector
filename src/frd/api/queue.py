"""The moderation queue: a scored, ranked feed of reviews awaiting a decision.

WHY THIS IS A SEPARATE ENDPOINT AND NOT JUST /score/batch
---------------------------------------------------------
`/score/batch` is the integration surface: a caller hands over reviews plus
whatever context they can supply, and gets scores back. It is stateless and it
is honest about what it was given -- if you send no context, the behavioural
and graph blocks report their "no history" sentinels and the response says so.

A moderation console needs something different: a queue drawn from the corpus
the platform already holds, where every review is scored with its FULL prior
history because that history is sitting right there in the reviews table. Asking
the browser to POST 40,000 context reviews so the server can score fifty of them
would be absurd. So this module keeps the corpus server-side, features it once,
scores it once, and serves pages out of the result.

WHY THE QUEUE DEFAULTS TO THE HELD-OUT PERIOD
---------------------------------------------
The shipped model was fitted on the first 70% of this corpus by time. Reviews
from that period are partly memorised, so scores on them look far better than
the model deserves and would make the console a demo of overfitting. The queue
therefore serves the temporal TEST window by default -- the same rows the
reported 0.914 PR-AUC was measured on. `window=all` is available and clearly
labelled, because being able to see the difference is more instructive than
being protected from it.

A NOTE ON THE LABEL COLUMN
--------------------------
This corpus is synthetic and carries ground truth. That is exposed as
`label`, nullable, so the console can show whether the model was right. Real
moderation queues have no such column -- everything here degrades to
`label: null` and the console hides the comparison. Treating the label as
optional from the start is what keeps this from being a demo that only works
on simulated data.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from ..config import CONFIG, PROCESSED_DIR
from ..data.splits import temporal_split
from ..evaluate import threshold_table
from ..explain import (
    AUTO_REMOVE_CAP, AUTO_REMOVE_MARGIN, _decision, auto_remove_at,
)
from ..explain import explain as explain_reviews
from ..schema import LABEL, REVIEW_ID, TIMESTAMP, normalise
from ..train import prepare_dataset

router = APIRouter(prefix="/queue", tags=["queue"])

# The corpus is loaded, featured and scored exactly once, behind a lock so two
# concurrent first requests cannot both pay the ~20s cost. Everything after
# that is a slice of a cached frame.
_CACHE: dict[str, Any] = {}
_LOCK = threading.Lock()

DISPLAY_COLUMNS = (
    "review_id", "product_id", "reviewer_id", "rating", "title", "text",
    "timestamp", "category", "verified_purchase", "helpful_votes",
)


def corpus_path() -> Path | None:
    """Locate the corpus, mirroring the CLI's parquet-then-csv preference."""
    override = os.environ.get("FRD_DATA_PATH")
    if override:
        p = Path(override)
        return p if p.exists() else None
    for name in ("reviews.parquet", "reviews.csv"):
        p = PROCESSED_DIR / name
        if p.exists():
            return p
    return None


def _read(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _build(detector) -> dict[str, Any]:
    """Load, feature and score the whole corpus. Called once."""
    path = corpus_path()
    if path is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "no corpus found; run `make dataset` or set FRD_DATA_PATH. "
                f"Looked in {PROCESSED_DIR} for reviews.parquet / reviews.csv"
            ),
        )

    t0 = time.time()
    df = normalise(_read(path))
    prepared = prepare_dataset(df, verbose=False)

    # The split is computed on the prepared frame so the window boundaries are
    # identical to the ones training used -- not recomputed from row counts
    # here, where an off-by-one would quietly leak training rows into the queue.
    split = temporal_split(prepared, CONFIG.split)
    train_end = split.train[TIMESTAMP].max()
    test_start = split.test[TIMESTAMP].min()

    scores = detector.predict_proba(prepared)
    prepared = prepared.assign(score=scores)

    return {
        "frame": prepared,
        "test_ids": set(split.test[REVIEW_ID].astype(str)),
        "path": str(path),
        "train_end": train_end,
        "test_start": test_start,
        "build_seconds": round(time.time() - t0, 1),
    }


def corpus(detector) -> dict[str, Any]:
    if "frame" not in _CACHE:
        with _LOCK:
            if "frame" not in _CACHE:
                _CACHE.update(_build(detector))
    return _CACHE


def _window(store: dict[str, Any], window: str) -> pd.DataFrame:
    df = store["frame"]
    if window == "all":
        return df
    return df[df[REVIEW_ID].astype(str).isin(store["test_ids"])]


def _row_payload(row: pd.Series, threshold: float) -> dict[str, Any]:
    score = float(row["score"])
    out: dict[str, Any] = {c: row[c] for c in DISPLAY_COLUMNS}
    out["timestamp"] = pd.Timestamp(row["timestamp"]).isoformat()
    out["rating"] = float(row["rating"])
    out["helpful_votes"] = int(row["helpful_votes"])
    out["verified_purchase"] = bool(row["verified_purchase"])
    for key in ("review_id", "product_id", "reviewer_id", "title", "text",
                "category"):
        out[key] = str(out[key])
    out["score"] = round(score, 4)
    out["decision"] = _decision(score, threshold)
    out["label"] = None if LABEL not in row.index or pd.isna(row[LABEL]) \
        else int(row[LABEL])
    return out


@router.get("/stats")
def stats(
    window: str = Query("test", pattern="^(test|all)$"),
) -> dict[str, Any]:
    """Counts, operating points, and the policy constants the UI re-tiers with.

    The threshold table is computed on the rows actually being displayed, not
    copied from the training report. If a moderator is looking at the test
    window, the precision the slider promises is the precision measured on the
    reviews in front of them.
    """
    from .main import _detector  # noqa: PLC0415  (avoids a circular import)

    det = _detector()
    store = corpus(det)
    df = _window(store, window)
    s = df["score"].to_numpy(float)
    thr = det.threshold_

    has_labels = LABEL in df.columns and df[LABEL].notna().all()
    table: list[dict] = []
    if has_labels:
        y = df[LABEL].to_numpy(int)
        table = threshold_table(y, s, daily_volume=len(df)).to_dict("records")

    return {
        "window": window,
        "n_reviews": len(df),
        "corpus_path": store["path"],
        "build_seconds": store["build_seconds"],
        "train_period_ends": pd.Timestamp(store["train_end"]).isoformat(),
        "test_period_starts": pd.Timestamp(store["test_start"]).isoformat(),
        "threshold": thr,
        "auto_remove_at": auto_remove_at(thr),
        # Handed to the client so its slider re-tiers rows with the same policy
        # the server applies, rather than a hardcoded TypeScript copy of it.
        "decision_policy": {
            "auto_remove_margin": AUTO_REMOVE_MARGIN,
            "auto_remove_cap": AUTO_REMOVE_CAP,
        },
        "tiers": {
            "auto_remove": int((s >= auto_remove_at(thr)).sum()),
            "queue_for_review": int(((s >= thr) & (s < auto_remove_at(thr))).sum()),
            "allow": int((s < thr).sum()),
        },
        "has_labels": bool(has_labels),
        "positive_rate": round(float(df[LABEL].mean()), 4) if has_labels else None,
        "score_histogram": _histogram(s),
        "threshold_table": table,
    }


def _histogram(s: np.ndarray, bins: int = 20) -> list[dict]:
    counts, edges = np.histogram(s, bins=bins, range=(0.0, 1.0))
    return [
        {"lo": round(float(edges[i]), 3), "hi": round(float(edges[i + 1]), 3),
         "count": int(counts[i])}
        for i in range(len(counts))
    ]


@router.get("")
def queue(
    window: str = Query("test", pattern="^(test|all)$"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    min_score: float = Query(0.0, ge=0.0, le=1.0),
    decision: str | None = Query(None, pattern="^(allow|queue_for_review|auto_remove)$"),
    category: str | None = None,
    search: str | None = None,
    sort: str = Query("score_desc", pattern="^(score_desc|score_asc|newest|oldest)$"),
    threshold: float | None = Query(None, ge=0.0, le=1.0),
    explain: bool = False,
) -> dict[str, Any]:
    """One page of the queue, ranked.

    `threshold` overrides the model's tuned operating point for this request
    only -- it re-tiers and re-filters, it does not rescore. Scores are a
    property of the review; the threshold is a policy applied to them.
    """
    from .main import _detector  # noqa: PLC0415

    det = _detector()
    store = corpus(det)
    df = _window(store, window)
    thr = det.threshold_ if threshold is None else float(threshold)

    if min_score > 0:
        df = df[df["score"] >= min_score]
    if category:
        df = df[df["category"].astype(str) == category]
    if search:
        needle = search.lower()
        hay = (df["text"].astype(str) + " " + df["title"].astype(str)).str.lower()
        df = df[hay.str.contains(needle, regex=False)
                | df[REVIEW_ID].astype(str).str.lower().str.contains(needle, regex=False)
                | df["reviewer_id"].astype(str).str.lower().str.contains(needle, regex=False)
                | df["product_id"].astype(str).str.lower().str.contains(needle, regex=False)]
    if decision:
        lo = thr if decision == "queue_for_review" else (
            auto_remove_at(thr) if decision == "auto_remove" else 0.0)
        hi = auto_remove_at(thr) if decision == "queue_for_review" else (
            1.01 if decision == "auto_remove" else thr)
        df = df[(df["score"] >= lo) & (df["score"] < hi)]

    total = len(df)
    if sort == "score_desc":
        df = df.sort_values("score", ascending=False, kind="mergesort")
    elif sort == "score_asc":
        df = df.sort_values("score", ascending=True, kind="mergesort")
    elif sort == "newest":
        df = df.sort_values(TIMESTAMP, ascending=False, kind="mergesort")
    else:
        df = df.sort_values(TIMESTAMP, ascending=True, kind="mergesort")

    page = df.iloc[offset:offset + limit]
    items = [_row_payload(r, thr) for _, r in page.iterrows()]

    # Reason codes for a page are cheap (tree SHAP over ~50 rows); reason codes
    # for 40,000 are not. So they are opt-in per page rather than precomputed.
    if explain and len(page):
        for item, exp in zip(items, explain_reviews(det, page)):
            item["reasons"] = exp.reasons

    return {
        "items": items,
        "total": total,
        "offset": offset,
        "limit": limit,
        "threshold": thr,
        "auto_remove_at": auto_remove_at(thr),
        "window": window,
    }


@router.get("/categories")
def categories() -> list[str]:
    from .main import _detector  # noqa: PLC0415

    det = _detector()
    store = corpus(det)
    return sorted(store["frame"]["category"].astype(str).unique().tolist())


@router.get("/{review_id}")
def detail(review_id: str, threshold: float | None = None) -> dict[str, Any]:
    """Full evidence for one review: reasons, SHAP, tokens, and its neighbours.

    The neighbour lists are what make a coordination reason auditable. A
    moderator told "this author substantially overlaps 2 other accounts" can
    only act on it if they can see the author's other reviews and the product's
    other recent reviews -- so both are returned, drawn strictly from BEFORE
    the review under inspection, which is all the model itself was allowed.
    """
    from .main import _detector  # noqa: PLC0415

    det = _detector()
    store = corpus(det)
    df = store["frame"]

    hit = df[df[REVIEW_ID].astype(str) == review_id]
    if hit.empty:
        raise HTTPException(status_code=404, detail=f"no review {review_id!r}")

    row = hit.iloc[0]
    thr = det.threshold_ if threshold is None else float(threshold)
    exp = explain_reviews(det, hit)[0]

    ts = pd.Timestamp(row[TIMESTAMP])
    prior = df[df[TIMESTAMP] < ts]
    by_author = prior[prior["reviewer_id"].astype(str) == str(row["reviewer_id"])]
    on_product = prior[prior["product_id"].astype(str) == str(row["product_id"])]

    def brief(sub: pd.DataFrame, n: int = 8) -> list[dict]:
        sub = sub.sort_values(TIMESTAMP, ascending=False).head(n)
        return [{
            "review_id": str(r["review_id"]),
            "product_id": str(r["product_id"]),
            "reviewer_id": str(r["reviewer_id"]),
            "rating": float(r["rating"]),
            "text": str(r["text"])[:280],
            "timestamp": pd.Timestamp(r[TIMESTAMP]).isoformat(),
            "score": round(float(r["score"]), 4),
            "label": None if LABEL not in r.index or pd.isna(r[LABEL])
                     else int(r[LABEL]),
        } for _, r in sub.iterrows()]

    payload = _row_payload(row, thr)
    payload.update({
        "reasons": exp.reasons,
        "feature_contributions": exp.feature_contributions,
        "token_contributions": exp.token_contributions,
        "explain_method": exp.method,
        "threshold": thr,
        "auto_remove_at": auto_remove_at(thr),
        "author_prior_reviews": brief(by_author),
        "author_prior_count": len(by_author),
        "product_prior_reviews": brief(on_product),
        "product_prior_count": len(on_product),
    })
    return payload
