/**
 * The rating block: what the product is rated, and what it is rated once the
 * reviews the detector will not stand behind are taken out.
 *
 * Three deliberate refusals here:
 *
 * 1. **No cleaned rating unless it moved.** Printing "4.6 -> 4.6" on the 58%
 *    of products where nothing changed teaches shoppers to ignore the badge,
 *    which costs credibility on the products where it matters. Below the
 *    server's `material_delta`, this collapses to one quiet line.
 * 2. **The withheld set is inspectable, not asserted.** The "why" panel lists
 *    the actual reviews with their moderator reason codes. A trust badge that
 *    cannot be audited is marketing.
 * 3. **A thin cleaned rating says so.** Some products lose 46 of 49 reviews;
 *    the mean of the three survivors is the best estimate available and still
 *    not worth much, and the panel says which of those two it is.
 */

import { useState } from "react";
import type { ProductDetail, ShopReview } from "../types";
import { DIRECTION_COPY, Stars } from "./bits";

export function TrustPanel({ detail }: { detail: ProductDetail }) {
  const [open, setOpen] = useState(false);
  const p = detail.product;
  const copy = DIRECTION_COPY[p.direction] ?? DIRECTION_COPY.unchanged;
  const moved = p.material && p.delta !== null && p.adjusted_rating !== null;

  return (
    <section className="trust">
      <div className="trust-raw">
        <div className="trust-label">Listed rating</div>
        <div className="trust-row">
          <Stars value={p.raw_rating} size={19} />
          <strong>{p.raw_rating?.toFixed(2) ?? "—"}</strong>
          <span className="muted">{p.n_reviews.toLocaleString()} reviews</span>
        </div>
      </div>

      {moved ? (
        <div className={`trust-clean ${p.delta! > 0 ? "up" : "down"}`}>
          <div className="trust-label">
            Rating with suspected fakes removed
            <span className="badge">{copy.label}</span>
          </div>
          <div className="trust-row">
            <Stars value={p.adjusted_rating} size={19} />
            <strong>{p.adjusted_rating!.toFixed(2)}</strong>
            <span className={`delta ${p.delta! > 0 ? "up" : "down"}`}>
              {p.delta! > 0 ? "▲" : "▼"} {Math.abs(p.delta!).toFixed(2)} stars
            </span>
          </div>
          <p className="trust-note">
            {p.n_withheld.toLocaleString()} of {p.n_reviews.toLocaleString()} reviews{" "}
            {copy.blurb}.
            {p.adjusted_thin ? (
              <>
                {" "}
                <span className="warn">
                  This figure rests on the {p.n_shown} review
                  {p.n_shown === 1 ? "" : "s"} that remain, so it is an estimate
                  rather than a settled rating.
                </span>
              </>
            ) : null}
          </p>
          <button className="link" onClick={() => setOpen(!open)}>
            {open ? "Hide what was withheld" : `Show what was withheld (${detail.total_withheld})`}
          </button>
        </div>
      ) : (
        <div className="trust-clean flat">
          <div className="trust-label">Checked for review manipulation</div>
          <p className="trust-note">
            {p.n_withheld > 0
              ? `${p.n_withheld.toLocaleString()} review${p.n_withheld === 1 ? "" : "s"} withheld, but the rating moves by less than ${detail.material_delta} of a star — not enough to change what this listing claims.`
              : "Nothing on this listing scored high enough to withhold."}
          </p>
          {detail.total_withheld > 0 ? (
            <button className="link" onClick={() => setOpen(!open)}>
              {open ? "Hide what was withheld" : `Show what was withheld (${detail.total_withheld})`}
            </button>
          ) : null}
        </div>
      )}

      {p.n_under_review > 0 ? (
        <p className="trust-under">
          {p.n_under_review.toLocaleString()} further review
          {p.n_under_review === 1 ? " is" : "s are"} flagged for a human to check
          and {p.n_under_review === 1 ? "is" : "are"} still shown below. Only
          reviews the policy will act on unattended (score ≥{" "}
          {detail.auto_remove_at.toFixed(2)}) are withheld.
        </p>
      ) : null}

      {open ? <WithheldList detail={detail} /> : null}
    </section>
  );
}

function WithheldList({ detail }: { detail: ProductDetail }) {
  return (
    <div className="withheld">
      <div className="withheld-head">
        Withheld reviews — top {detail.withheld.length} of{" "}
        {detail.total_withheld.toLocaleString()} by score
      </div>
      {detail.withheld.map((r) => <WithheldRow key={r.review_id} review={r} />)}
      <p className="withheld-foot">
        Scores are calibrated probabilities from the model in{" "}
        <code>artifacts/models/</code>; reason codes are the same ones a
        moderator sees in the console. Nothing here is deleted — it is withheld
        from the rating on this page and remains auditable and appealable.
      </p>
    </div>
  );
}

function WithheldRow({ review }: { review: ShopReview }) {
  return (
    <div className="withheld-row">
      <div className="withheld-meta">
        <span className="score">{review.score.toFixed(3)}</span>
        <span className="muted">{review.rating.toFixed(0)}★</span>
        <span className="muted">{review.reviewer_id}</span>
        <span className="muted">{new Date(review.timestamp).toLocaleDateString()}</span>
        {review.in_train_period ? (
          <span className="tag-train" title="This review falls inside the period the model was fitted on, so its score is partly memorised.">
            fitted period
          </span>
        ) : null}
        {review.label !== null ? (
          <span className={`tag-truth ${review.label ? "fake" : "genuine"}`}>
            {review.label ? "truly fake" : "actually genuine"}
          </span>
        ) : null}
      </div>
      {review.title ? <div className="withheld-title">{review.title}</div> : null}
      <p className="withheld-text">{review.text}</p>
      {review.reasons?.length ? (
        <ul className="reasons">
          {review.reasons.map((r) => <li key={r}>{r}</li>)}
        </ul>
      ) : null}
    </div>
  );
}
