"""Canonical review schema.

Every data source -- synthetic generator, Kaggle dump, or a live production
table -- is normalised into this shape before it reaches feature engineering.
Keeping one schema is what lets the same model serve a research notebook and
the FastAPI endpoint without a second code path.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any

import pandas as pd

# --- Column contract -------------------------------------------------------

REVIEW_ID = "review_id"
PRODUCT_ID = "product_id"
REVIEWER_ID = "reviewer_id"
RATING = "rating"
TITLE = "title"
TEXT = "text"
TIMESTAMP = "timestamp"
VERIFIED = "verified_purchase"
HELPFUL = "helpful_votes"
CATEGORY = "category"
LABEL = "label"  # 1 = deceptive / paid, 0 = genuine

REQUIRED_COLUMNS: tuple[str, ...] = (
    REVIEW_ID,
    PRODUCT_ID,
    REVIEWER_ID,
    RATING,
    TEXT,
    TIMESTAMP,
)

OPTIONAL_COLUMNS: dict[str, Any] = {
    TITLE: "",
    VERIFIED: False,
    HELPFUL: 0,
    CATEGORY: "unknown",
}

DTYPES: dict[str, str] = {
    REVIEW_ID: "string",
    PRODUCT_ID: "string",
    REVIEWER_ID: "string",
    RATING: "float32",
    TITLE: "string",
    TEXT: "string",
    VERIFIED: "boolean",
    HELPFUL: "int32",
    CATEGORY: "string",
}


@dataclass(slots=True)
class Review:
    """A single review, as accepted by the scoring API."""

    review_id: str
    product_id: str
    reviewer_id: str
    rating: float
    text: str
    timestamp: datetime
    title: str = ""
    verified_purchase: bool = False
    helpful_votes: int = 0
    category: str = "unknown"
    label: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ReviewerHistory:
    """Prior activity for the review's author, supplied at scoring time.

    In production this is fetched from the reviews table; the point of passing
    it explicitly is that the API can score a brand-new review without the
    model needing database access, and unit tests can construct any history
    they like.
    """

    reviewer_id: str
    prior_ratings: list[float] = field(default_factory=list)
    prior_timestamps: list[datetime] = field(default_factory=list)
    prior_texts: list[str] = field(default_factory=list)
    account_age_days: float | None = None


def normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and coerce an arbitrary frame into the canonical schema.

    Raises on missing required columns, fills documented defaults for the
    optional ones, and sorts by time -- every downstream feature assumes
    chronological order.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"missing required column(s): {missing}. "
            f"Required: {list(REQUIRED_COLUMNS)}"
        )

    out = df.copy()
    for col, default in OPTIONAL_COLUMNS.items():
        if col not in out.columns:
            out[col] = default

    out[TIMESTAMP] = pd.to_datetime(out[TIMESTAMP], utc=True, errors="coerce")
    if out[TIMESTAMP].isna().any():
        n = int(out[TIMESTAMP].isna().sum())
        raise ValueError(f"{n} row(s) have an unparseable {TIMESTAMP}")

    for col, dtype in DTYPES.items():
        if col in out.columns:
            out[col] = out[col].astype(dtype)

    out[TEXT] = out[TEXT].fillna("")
    out[TITLE] = out[TITLE].fillna("")
    out[RATING] = out[RATING].clip(1, 5)

    if LABEL in out.columns:
        out[LABEL] = out[LABEL].astype("int8")

    return out.sort_values(TIMESTAMP, kind="mergesort").reset_index(drop=True)


def utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)
