"""Request/response models for the scoring API."""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field, field_validator


class ReviewIn(BaseModel):
    review_id: str = Field(..., examples=["R0001"])
    product_id: str = Field(..., examples=["P00042"])
    reviewer_id: str = Field(..., examples=["U000123"])
    rating: float = Field(..., ge=1, le=5)
    text: str = Field(..., examples=["Highly recommend! Amazing quality."])
    timestamp: datetime | None = None
    title: str = ""
    verified_purchase: bool = False
    helpful_votes: int = 0
    category: str = "unknown"

    @field_validator("timestamp")
    @classmethod
    def _default_now(cls, v: datetime | None) -> datetime:
        return v or datetime.now(tz=timezone.utc)


class ContextReview(BaseModel):
    """A prior review supplying behavioural context.

    Send the author's recent reviews AND the product's recent reviews. Without
    them the behavioural features fall back to their "no history" sentinels and
    the score is driven by text alone -- the response says so via
    `behavioural_context_used`, so a caller is never silently downgraded.
    """

    review_id: str
    product_id: str
    reviewer_id: str
    rating: float = Field(..., ge=1, le=5)
    text: str = ""
    timestamp: datetime
    title: str = ""
    verified_purchase: bool = False
    helpful_votes: int = 0
    category: str = "unknown"


class ScoreRequest(BaseModel):
    review: ReviewIn
    context: list[ContextReview] = Field(
        default_factory=list,
        description="Prior reviews by this author and on this product. "
                    "Anything at or after the target review's timestamp is "
                    "discarded server-side.",
    )
    explain: bool = True


class BatchScoreRequest(BaseModel):
    reviews: list[ReviewIn] = Field(..., max_length=1000)
    context: list[ContextReview] = Field(default_factory=list)
    explain: bool = False


class TokenContribution(BaseModel):
    token: str
    weight: float
    direction: str


class FeatureContribution(BaseModel):
    feature: str
    contribution: float
    value: float
    direction: str


class ScoreResponse(BaseModel):
    review_id: str
    score: float = Field(..., description="Calibrated P(fake), 0-1")
    decision: str = Field(..., description="allow | queue_for_review | auto_remove")
    threshold: float
    behavioural_context_used: bool
    n_context_reviews: int
    reasons: list[str] = []
    feature_contributions: list[FeatureContribution] = []
    token_contributions: list[TokenContribution] = []
    model_version: str
    explain_method: str = ""


class BatchScoreResponse(BaseModel):
    results: list[ScoreResponse]
    count: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    blocks: list[str] = []
    backend: str = ""
    threshold: float | None = None
    calibration: str = ""
