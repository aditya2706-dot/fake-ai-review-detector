"""Explanations for a flagged review.

A score alone is not actionable. A moderator about to delete a customer's
review needs the reason, an appeals process needs a record of it, and the ML
team needs it to notice when the model has latched onto something silly. This
module produces three layers:

  1. Exact per-feature contributions for stage B. LightGBM computes true tree
     SHAP values natively via `pred_contrib=True`, so no `shap` dependency is
     needed. Under the sklearn fallback booster that API does not exist, and
     the module says so rather than inventing numbers.
  2. Exact per-token contributions for stage A -- the text model is linear, so
     a token's contribution is simply coefficient x TF-IDF weight.
  3. Reason codes: short English sentences generated from the top
     contributions, which is the layer a human actually reads.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .models.pipeline import FakeReviewDetector

# Feature name -> template for a human-readable reason. `{v}` is the feature's
# value for this review. Only features with a template produce a reason code,
# which keeps the moderator-facing output curated rather than a raw dump.
#
# The list is deliberately incomplete. Features like exclamation count, raw
# character length, and hour-of-day genuinely contribute to the score, but
# "contains 2 exclamation marks" is not a reason a moderator can act on and it
# crowds out the substantive evidence. Those features stay visible in
# `feature_contributions` for the ML team; they just do not become prose.
REASON_TEMPLATES: dict[str, str] = {
    "rv_count_24h": "Author posted {v:.0f} reviews in a 24-hour window",
    "rv_count_7d": "Author posted {v:.0f} reviews in the past 7 days",
    "rv_max_burst_24h": "Author's peak activity is {v:.0f} reviews in 24 hours",
    "rv_rate_per_day": "Author averages {v:.1f} reviews per day",
    "rv_self_similarity_max": "Text is {v:.0%} similar to another review by the same author",
    "rv_prior_extreme_share": "{v:.0%} of the author's past ratings are 1 or 5 stars",
    "rv_verified_share": "Only {v:.0%} of the author's past reviews were verified purchases",
    "rv_tenure_days": "Account has been active for {v:.0f} days",
    "rv_product_repeat_share": "{v:.0%} of the author's reviews repeat the same product",
    "rv_night_share": "{v:.0%} of the author's reviews were posted overnight",
    "rv_category_entropy": "Author reviews across unusually many categories (entropy {v:.2f})",
    "rv_hours_since_prev": "Posted {v:.1f} hours after the author's previous review",
    "pr_peer_similarity_max": "Text is {v:.0%} similar to another review on this product",
    "pr_count_24h": "Product received {v:.0f} reviews in 24 hours",
    "pr_count_7d": "Product received {v:.0f} reviews in the past 7 days",
    "pr_burst_ratio": "Product's review volume is {v:.1f}x its weekly average",
    "pr_rating_deviation": "Rating differs from the product's average by {v:.1f} stars",
    "pr_new_reviewer_share_7d": "{v:.0%} of this product's recent reviews come from new accounts",
    "pr_prior_five_share": "{v:.0%} of this product's prior reviews are 5 stars",
    "lx_superlative_density": "Unusually high superlative density ({v:.1%} of words)",
    "lx_intensifier_density": "Unusually high intensifier density ({v:.1%} of words)",
    "lx_specificity_score": "Low specificity: little concrete detail (score {v:.2f})",
    "lx_promo_phrase_count": "Contains {v:.0f} stock promotional phrase(s)",
    "lx_hedge_density": "No hedging or qualification, unlike typical genuine reviews",
    "lx_verified": "Not a verified purchase",
    "lx_rating_is_extreme": "Extreme (1 or 5 star) rating",
    "text_score": "Wording resembles known paid-review text (text model score {v:.2f})",
    # Graph evidence is about the company an account keeps, so these sentences
    # name another account or a group. That makes them the most consequential
    # reasons in the list for an appeal -- and the ones most in need of the
    # guards below, since "linked to 2 other accounts" on a marketplace where
    # everyone buys the same kettle is not evidence of anything.
    "gp_strong_partner_count": "Author's review history substantially overlaps {v:.0f} other account(s)",
    "gp_max_partner_jaccard": "{v:.0%} of this author's products were also reviewed by one single other account",
    "gp_partner_concentration": "Author's overlap with others is concentrated in one account ({v:.0%} of it)",
    "gp_component_size": "Author belongs to a cluster of {v:.0f} tightly linked accounts",
    "gp_tie_to_product_share": "Author has previously co-reviewed with {v:.0%} of this product's recent reviewers",
    "gp_cohort_strong_pair_density": "{v:.0%} of this product's recent reviewer pairs have reviewed together before",
    "gp_tight_cooccur_max": "Author has posted within 48 hours of the same account {v:.0f} times",
}


# A positive SHAP contribution means the feature pushed the score up, but it
# does not mean the feature's VALUE reads as incriminating to a human. A review
# 6% similar to its author's previous one can still contribute positively
# relative to the model's baseline, while "Text is 6% similar to another review"
# is actively misleading in a moderation UI. These guards suppress a reason
# unless the value itself supports the claim the sentence makes.
REASON_GUARDS: dict[str, tuple[str, float]] = {
    "rv_self_similarity_max": ("min", 0.55),
    "pr_peer_similarity_max": ("min", 0.55),
    "rv_count_24h": ("min", 3),
    "rv_count_7d": ("min", 5),
    "rv_max_burst_24h": ("min", 3),
    "rv_rate_per_day": ("min", 1.5),
    "pr_count_24h": ("min", 4),
    "pr_count_7d": ("min", 6),
    "pr_burst_ratio": ("min", 2.0),
    "pr_rating_deviation": ("min", 1.0),
    "rv_prior_extreme_share": ("min", 0.6),
    "rv_verified_share": ("max", 0.4),
    "rv_product_repeat_share": ("min", 0.3),
    "rv_night_share": ("min", 0.4),
    "pr_new_reviewer_share_7d": ("min", 0.5),
    "pr_prior_five_share": ("min", 0.7),
    "lx_superlative_density": ("min", 0.05),
    "lx_intensifier_density": ("min", 0.08),
    "lx_specificity_score": ("max", 0.5),
    "lx_promo_phrase_count": ("min", 1),
    "lx_hedge_density": ("max", 0.001),
    "lx_verified": ("max", 0.5),
    "rv_tenure_days": ("max", 30),
    "rv_hours_since_prev": ("max", 6),
    "rv_category_entropy": ("min", 1.5),
    # Deliberately strict. One shared product is the base rate of any
    # marketplace, and an accusation of coordination is the hardest kind for a
    # customer to rebut, so these only speak when the structure is unambiguous.
    "gp_strong_partner_count": ("min", 2),
    "gp_max_partner_jaccard": ("min", 0.35),
    "gp_partner_concentration": ("min", 0.5),
    "gp_component_size": ("min", 4),
    "gp_tie_to_product_share": ("min", 0.3),
    "gp_cohort_strong_pair_density": ("min", 0.2),
    "gp_tight_cooccur_max": ("min", 3),
}


def _reason_allowed(feature: str, value: float) -> bool:
    """Does the feature's value support the sentence the template asserts?"""
    guard = REASON_GUARDS.get(feature)
    if guard is None:
        return True
    kind, bound = guard
    if kind == "min":
        return value >= bound
    return 0.0 <= value <= bound      # negative values are "unknown" sentinels


@dataclass
class Explanation:
    review_id: str
    score: float
    decision: str
    reasons: list[str] = field(default_factory=list)
    feature_contributions: list[dict] = field(default_factory=list)
    token_contributions: list[dict] = field(default_factory=list)
    method: str = ""

    def as_dict(self) -> dict:
        return {
            "review_id": self.review_id,
            "score": round(self.score, 4),
            "decision": self.decision,
            "reasons": self.reasons,
            "feature_contributions": self.feature_contributions,
            "token_contributions": self.token_contributions,
            "method": self.method,
        }


def _stage_b_contributions(
    det: FakeReviewDetector, X: pd.DataFrame
) -> tuple[np.ndarray | None, str]:
    """(n_rows, n_features) SHAP contributions, or None if unsupported."""
    booster = det.stage_b_
    if booster is None:
        raise RuntimeError("model is not fitted")

    text_score = (
        det.text_model_.predict_proba(X)[:, 1].astype(np.float32)
        if det.text_model_ is not None else None
    )
    Xb, _ = det._stage_b_matrix(X, text_score)

    if hasattr(booster, "booster_"):  # LightGBM: exact tree SHAP, no extra dep
        contrib = booster.booster_.predict(Xb, pred_contrib=True)
        return np.asarray(contrib)[:, :-1], "lightgbm_tree_shap"

    return None, "unavailable_sklearn_histgb"


def _token_contributions(
    det: FakeReviewDetector, X: pd.DataFrame, top: int = 8
) -> list[list[dict]]:
    """Exact linear contributions (coef x tfidf) for the text model."""
    if det.text_model_ is None:
        return [[] for _ in range(len(X))]

    union = det.text_model_.named_steps["features"]
    clf = det.text_model_.named_steps["clf"]
    M = union.transform(X)                      # sparse, (n, n_text_features)
    coef = clf.coef_.ravel()
    names = union.get_feature_names_out()

    out: list[list[dict]] = []
    M = M.tocsr()
    for i in range(M.shape[0]):
        row = M[[i]]           # .getrow() was removed in scipy 1.14
        if row.nnz == 0:
            out.append([])
            continue
        idx = row.indices
        vals = row.data * coef[idx]
        order = np.argsort(-np.abs(vals))[:top]
        out.append([
            {
                "token": str(names[idx[j]]).split("__", 1)[-1],
                "weight": round(float(vals[j]), 4),
                "direction": "fake" if vals[j] > 0 else "genuine",
            }
            for j in order
        ])
    return out


# The auto-remove tier sits a fixed margin above the queue threshold, capped so
# that no threshold setting can make removal automatic below 0.95. Named
# constants rather than literals because the moderation console re-tiers rows
# client-side as its threshold slider moves: it reads these two numbers from
# the API instead of keeping a second copy of the policy in TypeScript, which
# is the same anti-skew argument the feature code makes.
AUTO_REMOVE_MARGIN = 0.25
AUTO_REMOVE_CAP = 0.95


def auto_remove_at(threshold: float) -> float:
    """Score at or above which removal is automatic, for a given threshold."""
    return min(AUTO_REMOVE_CAP, threshold + AUTO_REMOVE_MARGIN)


def _decision(score: float, threshold: float) -> str:
    """Three-tier policy. See docs/METHODOLOGY.md for why not a binary call."""
    if score >= auto_remove_at(threshold):
        return "auto_remove"
    if score >= threshold:
        return "queue_for_review"
    return "allow"


def explain(
    det: FakeReviewDetector,
    X: pd.DataFrame,
    top_features: int = 6,
    top_tokens: int = 8,
) -> list[Explanation]:
    """Explain every row of `X`."""
    scores = det.predict_proba(X)
    contrib, method = _stage_b_contributions(det, X)
    tokens = _token_contributions(det, X, top=top_tokens)
    names = det.stage_b_columns_

    # Feature values, for filling reason templates.
    text_score = (
        det.text_model_.predict_proba(X)[:, 1].astype(np.float32)
        if det.text_model_ is not None else None
    )
    Xb, _ = det._stage_b_matrix(X, text_score)

    ids = (
        X["review_id"].astype(str).tolist()
        if "review_id" in X.columns else [str(i) for i in range(len(X))]
    )

    results: list[Explanation] = []
    for i in range(len(X)):
        feats: list[dict] = []
        reasons: list[str] = []

        if contrib is not None:
            c = contrib[i]
            order = np.argsort(-np.abs(c))[: top_features * 3]
            for j in order:
                name = names[j]
                feats.append({
                    "feature": name,
                    "contribution": round(float(c[j]), 4),
                    "value": round(float(Xb[i, j]), 4),
                    "direction": "fake" if c[j] > 0 else "genuine",
                })
            # Reason codes come only from features pushing TOWARDS fake, and
            # only from the curated template list.
            # Require a materially positive contribution, not merely a
            # non-negative one, so marginal features do not pad the list.
            strongest = max((abs(f["contribution"]) for f in feats), default=0.0)
            floor = max(0.02, strongest * 0.05)
            for f in feats:
                if len(reasons) >= top_features:
                    break
                if f["contribution"] < floor:
                    continue
                tpl = REASON_TEMPLATES.get(f["feature"])
                if tpl and _reason_allowed(f["feature"], f["value"]):
                    try:
                        reasons.append(tpl.format(v=f["value"]))
                    except (ValueError, KeyError):
                        continue
            feats = feats[:top_features]

        results.append(Explanation(
            review_id=ids[i],
            score=float(scores[i]),
            decision=_decision(float(scores[i]), det.threshold_),
            reasons=reasons,
            feature_contributions=feats,
            token_contributions=tokens[i],
            method=method,
        ))
    return results


def global_summary(det: FakeReviewDetector, top: int = 20) -> pd.DataFrame:
    """Model-level feature importance, for a model card."""
    return det.feature_importance(top=top)
