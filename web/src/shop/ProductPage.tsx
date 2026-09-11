/**
 * A product page: the listing, the rating both ways, and the reviews that
 * survived.
 *
 * The review list shows the `queue_for_review` tier alongside the clean ones,
 * marked. That is the policy this repo argues for rather than a UI compromise:
 * a review a model finds suspicious but a human has not confirmed is not
 * evidence of anything, and suppressing everything above the queue threshold
 * would run at 87.7% precision -- one wrongly hidden review in eight.
 */

import { useEffect, useState } from "react";
import { ApiError, api } from "../api";
import { navigate } from "../router";
import type { ProductDetail, ShopReview } from "../types";
import { ProductArt, Stars, money } from "./bits";
import { TrustPanel } from "./TrustPanel";

const PAGE = 20;

export function ProductPage({ id }: { id: string }) {
  const [detail, setDetail] = useState<ProductDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [sort, setSort] = useState("helpful");
  const [offset, setOffset] = useState(0);

  useEffect(() => { setOffset(0); setDetail(null); }, [id]);

  useEffect(() => {
    let live = true;
    setLoading(true);
    api
      .product(id, { limit: PAGE, offset, sort })
      .then((d) => { if (live) { setDetail(d); setError(null); } })
      .catch((e) => { if (live) setError(describe(e)); })
      .finally(() => { if (live) setLoading(false); });
    return () => { live = false; };
  }, [id, offset, sort]);

  if (error) {
    return (
      <div className="shop-page">
        <button className="link" onClick={() => navigate({ name: "shop" })}>← Back to the store</button>
        <div className="banner err" style={{ marginTop: 12 }}>{error}</div>
      </div>
    );
  }
  if (!detail) return <div className="spinner">Loading&hellip;</div>;

  const p = detail.product;
  return (
    <div className="shop-page">
      <button className="link crumb" onClick={() => navigate({ name: "shop" })}>
        ← {p.category.replace(/_/g, " ")}
      </button>

      <div className="pdp">
        <div className="pdp-art"><ProductArt product={p} size={340} /></div>

        <div className="pdp-buy">
          <div className="pdp-brand">{p.brand}</div>
          <h1>{p.name}</h1>
          <div className="pdp-rating">
            <Stars value={p.adjusted_rating ?? p.raw_rating} size={17} />
            <span className="muted">
              {(p.adjusted_rating ?? p.raw_rating)?.toFixed(2) ?? "—"} ·{" "}
              {p.n_reviews.toLocaleString()} reviews
            </span>
          </div>
          <div className="pdp-price">{money(p.price, p.currency)}</div>
          <ul className="pdp-features">
            {p.features.map((f) => <li key={f}>{f}</li>)}
          </ul>
          <button className="buy" disabled title="This is a detector demo — there is nothing to buy.">
            Add to basket
          </button>
          <p className="pdp-inert">
            Nothing here is for sale. The listing is generated from the product
            id; the reviews and their scores are real output from the model in{" "}
            <code>artifacts/models/</code>.
          </p>
        </div>

        <div className="pdp-trust">
          <TrustPanel detail={detail} />
          <Histogram detail={detail} />
        </div>
      </div>

      <section className="reviews">
        <div className="reviews-head">
          <h2>{detail.total_shown.toLocaleString()} reviews shown</h2>
          <select className="field" value={sort} onChange={(e) => setSort(e.target.value)}>
            <option value="helpful">Most helpful</option>
            <option value="newest">Newest first</option>
            <option value="oldest">Oldest first</option>
            <option value="rating">Highest rated</option>
            <option value="rating_asc">Lowest rated</option>
          </select>
          <span style={{ flex: 1 }} />
          <span className="pager-count">
            {detail.total_shown ? offset + 1 : 0}–
            {Math.min(offset + PAGE, detail.total_shown)} of{" "}
            {detail.total_shown.toLocaleString()}
          </span>
          <button className="ghost" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - PAGE))}>
            Prev
          </button>
          <button
            className="ghost"
            disabled={offset + PAGE >= detail.total_shown}
            onClick={() => setOffset(offset + PAGE)}
          >
            Next
          </button>
        </div>

        <div className={loading ? "stale" : ""}>
          {detail.reviews.map((r) => <Review key={r.review_id} review={r} />)}
          {detail.reviews.length === 0 ? (
            <div className="empty">
              Every review on this listing was withheld. There is nothing left
              to show, which is itself the finding.
            </div>
          ) : null}
        </div>
      </section>
    </div>
  );
}

function Histogram({ detail }: { detail: ProductDetail }) {
  const raw = detail.rating_histogram.raw;
  const shown = detail.rating_histogram.shown;
  const max = Math.max(1, ...Object.values(raw));
  return (
    <div className="rhist">
      <div className="rhist-head">
        Rating breakdown <span className="muted">listed vs. cleaned</span>
      </div>
      {["5", "4", "3", "2", "1"].map((star) => (
        <div className="rhist-row" key={star}>
          <span className="rhist-star">{star}★</span>
          <span className="rhist-track">
            <span className="rhist-raw" style={{ width: `${(100 * (raw[star] ?? 0)) / max}%` }} />
            <span className="rhist-shown" style={{ width: `${(100 * (shown[star] ?? 0)) / max}%` }} />
          </span>
          <span className="rhist-n">{(raw[star] ?? 0).toLocaleString()}</span>
        </div>
      ))}
      <div className="rhist-key">
        <span><i className="k-raw" /> as listed</span>
        <span><i className="k-shown" /> after withholding</span>
      </div>
    </div>
  );
}

function Review({ review }: { review: ShopReview }) {
  const flagged = review.decision === "queue_for_review";
  return (
    <article className={`review ${flagged ? "flagged" : ""}`}>
      <div className="review-top">
        <Stars value={review.rating} />
        {review.title ? <strong className="review-title">{review.title}</strong> : null}
      </div>
      <div className="review-meta">
        <span>{review.reviewer_id}</span>
        <span>{new Date(review.timestamp).toLocaleDateString()}</span>
        {review.verified_purchase ? <span className="verified">Verified purchase</span> : null}
        {review.helpful_votes > 0 ? (
          <span>{review.helpful_votes} found this helpful</span>
        ) : null}
        {review.in_train_period ? (
          <span className="tag-train" title="Inside the period the model was fitted on, so its score is partly memorised.">
            fitted period
          </span>
        ) : null}
      </div>
      <p className="review-text">{review.text}</p>
      {flagged ? (
        <div className="review-flag">
          Flagged for a moderator to check (score {review.score.toFixed(3)}) and
          shown anyway — a model's suspicion is not a finding.
        </div>
      ) : null}
    </article>
  );
}

function describe(e: unknown): string {
  if (e instanceof ApiError) return `${e.status}: ${e.message}`;
  if (e instanceof Error) return e.message;
  return String(e);
}
