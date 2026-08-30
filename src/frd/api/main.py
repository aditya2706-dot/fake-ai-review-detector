"""FastAPI scoring service.

THE ONE THING THAT MATTERS HERE
-------------------------------
Training/serving skew. The classic way a fake-review model dies in production
is a second, subtly different feature implementation on the serving path: a
window computed in local time instead of UTC, a similarity metric normalised
differently, a sentinel of 0 instead of -1. Offline PR-AUC stays beautiful and
the live model quietly degrades.

So this service does NOT reimplement anything. It assembles the supplied
context reviews plus the target review into a small DataFrame in canonical
schema and calls the SAME `build_behaviour_features` used in training, then
takes the last row. Slightly wasteful, completely skew-free.

Run:  uvicorn frd.api.main:app --reload
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException

from ..config import MODEL_DIR
from ..explain import explain as explain_reviews
from ..features.behavior import BEHAVIOUR_COLUMNS, build_behaviour_features
from ..features.text import LINGUISTIC_COLUMNS, LinguisticFeatures
from ..models.pipeline import FakeReviewDetector
from ..schema import normalise
from .schemas import (
    BatchScoreRequest, BatchScoreResponse, HealthResponse, ScoreRequest,
    ScoreResponse,
)

MODEL_PATH = Path(os.environ.get("FRD_MODEL_PATH", MODEL_DIR / "detector.joblib"))
MODEL_VERSION = os.environ.get("FRD_MODEL_VERSION", "dev")

STATE: dict[str, Any] = {"detector": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup, not per request."""
    if MODEL_PATH.exists():
        STATE["detector"] = FakeReviewDetector.load(MODEL_PATH)
        print(f"[api] loaded model from {MODEL_PATH}")
    else:
        print(f"[api] WARNING no model at {MODEL_PATH}; "
              f"run `make train` first. /score will return 503.")
    yield
    STATE.clear()


app = FastAPI(
    title="Fake Review Detector",
    description="Scores e-commerce reviews for the likelihood of being "
                "paid, incentivised, or machine-generated.",
    version=MODEL_VERSION,
    lifespan=lifespan,
)


def _detector() -> FakeReviewDetector:
    det = STATE.get("detector")
    if det is None:
        raise HTTPException(
            status_code=503,
            detail=f"no model loaded (expected {MODEL_PATH}); run training first",
        )
    return det


def _assemble(
    targets: list[dict], context: list[dict]
) -> tuple[pd.DataFrame, int]:
    """Build a canonical frame of [context..., targets...] and feature it.

    Context reviews at or after a target's timestamp are dropped: the model was
    trained on strictly-backward-looking features, so letting future context
    through would produce a feature distribution the model never saw.
    """
    tgt = pd.DataFrame(targets)
    ctx = pd.DataFrame(context) if context else None

    if ctx is not None and len(ctx):
        earliest_target = pd.to_datetime(tgt["timestamp"], utc=True).min()
        ctx = ctx.assign(timestamp=pd.to_datetime(ctx["timestamp"], utc=True))
        ctx = ctx[ctx["timestamp"] < earliest_target]

    combined = (
        pd.concat([ctx, tgt], ignore_index=True)
        if ctx is not None and len(ctx) else tgt
    )
    combined = combined.drop(columns=["label"], errors="ignore")

    df = normalise(combined)
    # normalise() sorts by timestamp, so identify the target rows by id rather
    # than by position.
    target_ids = set(tgt["review_id"].astype(str))

    beh = build_behaviour_features(df)
    ling = pd.DataFrame(
        LinguisticFeatures().transform(df), columns=LINGUISTIC_COLUMNS,
        index=df.index,
    )
    full = pd.concat([df, beh, ling], axis=1)
    mask = full["review_id"].astype(str).isin(target_ids)
    return full[mask].reset_index(drop=True), int(len(df) - mask.sum())


def _to_response(
    exp_dict: dict, det: FakeReviewDetector, n_context: int, used: bool
) -> ScoreResponse:
    return ScoreResponse(
        review_id=exp_dict["review_id"],
        score=exp_dict["score"],
        decision=exp_dict["decision"],
        threshold=det.threshold_,
        behavioural_context_used=used,
        n_context_reviews=n_context,
        reasons=exp_dict.get("reasons", []),
        feature_contributions=exp_dict.get("feature_contributions", []),
        token_contributions=exp_dict.get("token_contributions", []),
        model_version=MODEL_VERSION,
        explain_method=exp_dict.get("method", ""),
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    det = STATE.get("detector")
    return HealthResponse(
        status="ok" if det is not None else "degraded",
        model_loaded=det is not None,
        model_version=MODEL_VERSION,
        blocks=list(det.blocks) if det else [],
        backend=det.backend_ if det else "",
        threshold=det.threshold_ if det else None,
        calibration=det.calibration_kind_ if det else "",
    )


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest) -> ScoreResponse:
    det = _detector()
    target = req.review.model_dump()
    context = [c.model_dump() for c in req.context]

    features, n_ctx = _assemble([target], context)
    if req.explain:
        exps = explain_reviews(det, features)
        payload = exps[0].as_dict()
    else:
        s = float(det.predict_proba(features)[0])
        from ..explain import _decision  # noqa: PLC0415
        payload = {
            "review_id": target["review_id"], "score": s,
            "decision": _decision(s, det.threshold_), "method": "",
        }
    return _to_response(payload, det, n_ctx, used=n_ctx > 0)


@app.post("/score/batch", response_model=BatchScoreResponse)
def score_batch(req: BatchScoreRequest) -> BatchScoreResponse:
    det = _detector()
    if not req.reviews:
        return BatchScoreResponse(results=[], count=0)

    targets = [r.model_dump() for r in req.reviews]
    context = [c.model_dump() for c in req.context]
    features, n_ctx = _assemble(targets, context)

    if req.explain:
        payloads = [e.as_dict() for e in explain_reviews(det, features)]
    else:
        from ..explain import _decision  # noqa: PLC0415
        scores = det.predict_proba(features)
        ids = features["review_id"].astype(str).tolist()
        payloads = [
            {"review_id": ids[i], "score": float(scores[i]),
             "decision": _decision(float(scores[i]), det.threshold_), "method": ""}
            for i in range(len(ids))
        ]

    results = [_to_response(p, det, n_ctx, used=n_ctx > 0) for p in payloads]
    return BatchScoreResponse(results=results, count=len(results))


@app.get("/model")
def model_info() -> dict:
    det = _detector()
    imp = det.feature_importance(top=20)
    return {
        "version": MODEL_VERSION,
        "blocks": list(det.blocks),
        "backend": det.backend_,
        "calibration": det.calibration_kind_,
        "threshold": det.threshold_,
        "n_stage_b_features": len(det.stage_b_columns_),
        "behaviour_features": len(BEHAVIOUR_COLUMNS),
        "linguistic_features": len(LINGUISTIC_COLUMNS),
        "top_features": imp.to_dict(orient="records"),
        "fit_report": det.report_.__dict__ if det.report_ else None,
    }
