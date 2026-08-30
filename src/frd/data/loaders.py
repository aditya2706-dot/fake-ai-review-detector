"""Adapters that map real-world review dumps into the canonical schema.

Swap the synthetic generator for any of these and the rest of the pipeline is
unchanged -- that is the whole point of `frd.schema.normalise`.

A caution that matters more than the code: several popular public corpora
(notably the LLM-generated "fake reviews" sets) contain NO reviewer id and NO
timestamp. On those, every behavioural feature degenerates to a constant and
you are training a pure text classifier. `describe_coverage` tells you that
explicitly instead of letting you quietly report inflated metrics.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..schema import (
    CATEGORY, HELPFUL, LABEL, PRODUCT_ID, RATING, REVIEW_ID, REVIEWER_ID,
    TEXT, TIMESTAMP, TITLE, VERIFIED, normalise,
)


def load_generic(
    path: str | Path,
    column_map: dict[str, str],
    label_map: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Load a CSV/JSONL and rename columns into the canonical schema.

    Args:
        path: file to read (.csv, .tsv, .json, .jsonl -- inferred).
        column_map: {source_column: canonical_column}.
        label_map: optional {source_label_value: 0|1}, e.g. {"CG": 1, "OR": 0}.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".csv"}:
        df = pd.read_csv(path)
    elif suffix in {".tsv", ".tab"}:
        df = pd.read_csv(path, sep="\t")
    elif suffix in {".jsonl", ".ndjson"}:
        df = pd.read_json(path, lines=True)
    elif suffix == ".json":
        df = pd.read_json(path)
    elif suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"unsupported extension: {suffix}")

    df = df.rename(columns=column_map)

    if label_map and LABEL in df.columns:
        df[LABEL] = df[LABEL].map(label_map)
        if df[LABEL].isna().any():
            bad = df.loc[df[LABEL].isna()].index[:5].tolist()
            raise ValueError(f"label_map did not cover all values; rows {bad}")

    df = _fill_surrogates(df)
    return normalise(df)


def load_llm_fake_reviews(path: str | Path) -> pd.DataFrame:
    """Salminen et al. 'Fake Reviews Dataset' (category, rating, label, text_).

    label: CG = computer-generated (positive class), OR = original human.
    No reviewer/timestamp columns -> behavioural features are unavailable.
    """
    return load_generic(
        path,
        column_map={"text_": TEXT, "rating": RATING, "category": CATEGORY,
                    "label": LABEL},
        label_map={"CG": 1, "OR": 0},
    )


def load_amazon_jsonl(path: str | Path, limit: int | None = None) -> pd.DataFrame:
    """McAuley-style Amazon review JSONL (reviewerID, asin, overall, ...).

    Unlabelled -- use it for unsupervised/weak-label work or to pre-train the
    text encoder, not for supervised evaluation.
    """
    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if limit is not None and i >= limit:
                break
            r = json.loads(line)
            records.append({
                REVIEW_ID: f"A{i:08d}",
                PRODUCT_ID: r.get("asin", "unknown"),
                REVIEWER_ID: r.get("reviewerID", f"anon{i}"),
                RATING: r.get("overall", 3.0),
                TITLE: r.get("summary", "") or "",
                TEXT: r.get("reviewText", "") or "",
                TIMESTAMP: pd.to_datetime(
                    r.get("unixReviewTime", 0), unit="s", utc=True
                ),
                VERIFIED: bool(r.get("verified", False)),
                HELPFUL: int((r.get("helpful") or [0, 0])[0])
                if isinstance(r.get("helpful"), list) else 0,
                CATEGORY: r.get("category", "unknown"),
            })
    return normalise(pd.DataFrame(records))


def _fill_surrogates(df: pd.DataFrame) -> pd.DataFrame:
    """Invent the identity/time columns a corpus is missing, loudly.

    Surrogates keep the pipeline runnable; `describe_coverage` records that the
    behavioural block is inert so nobody misreads the results.
    """
    out = df.copy()
    n = len(out)
    if REVIEW_ID not in out.columns:
        out[REVIEW_ID] = [f"S{i:08d}" for i in range(n)]
    if REVIEWER_ID not in out.columns:
        out[REVIEWER_ID] = [f"unknown_{i}" for i in range(n)]  # each review its own author
    if PRODUCT_ID not in out.columns:
        out[PRODUCT_ID] = "unknown_product"
    if TIMESTAMP not in out.columns:
        # Arbitrary but ordered, so time-based splitting still runs.
        out[TIMESTAMP] = pd.date_range("2020-01-01", periods=n, freq="min", tz="UTC")
    return out


def describe_coverage(df: pd.DataFrame) -> dict[str, object]:
    """Report which feature families this corpus can actually support."""
    n = len(df)
    n_reviewers = df[REVIEWER_ID].nunique()
    n_products = df[PRODUCT_ID].nunique()
    surrogate_reviewers = df[REVIEWER_ID].astype(str).str.startswith("unknown_").mean()
    time_span = (df[TIMESTAMP].max() - df[TIMESTAMP].min())

    behavioural_usable = (
        n_reviewers < n * 0.95 and surrogate_reviewers < 0.5
    )
    return {
        "n_reviews": n,
        "n_reviewers": n_reviewers,
        "n_products": n_products,
        "reviews_per_reviewer": round(n / max(n_reviewers, 1), 2),
        "time_span_days": round(time_span.total_seconds() / 86400, 1),
        "has_labels": LABEL in df.columns,
        "positive_rate": round(float(df[LABEL].mean()), 4)
        if LABEL in df.columns else None,
        "behavioural_features_usable": bool(behavioural_usable),
        "note": (
            "Behavioural block is INERT: reviewer ids are surrogate or unique "
            "per review. Expect text-only performance."
            if not behavioural_usable else
            "Behavioural block is usable."
        ),
    }
