/**
 * The panel that tells a shopper what the detector did to this product's rating.
 *
 * THREE THINGS IT IS CAREFUL ABOUT
 *
 * 1. It is quiet when nothing happened. On most products cleaning moves the
 *    rating by hundredths of a star, and a panel that announces "verified!"
 *    on all of them teaches the eye to skip it -- which costs exactly the
 *    credibility needed on the one product in six where it matters.
 *
 * 2. It shows the arithmetic, not a verdict. Both ratings, the number of
 *    reviews behind each, and a link to the ones that were removed. A badge
 *    saying "trusted" would be an assertion; this is an argument someone can
 *    check.
 *
 * 3. It says when its own number is weak. `adjusted_thin` means the cleaned
 *    rating rests on fewer survivors than the server is willing to stand
 *    behind, and that is printed rather than quietly rounded away.
 */

import type { ProductDetail } from "../types";
import { Stars } from "./bits";

export function TrustPanel({ detail }: { detail: ProductDetail }) {
  const p = detail.product;
  const moved = p.material && p.delta !== null;
  const up = (p.delta ?? 0) > 0;
  const hist = detail.rating_histogram.shown;
  const maxBar = Math.max(1, ...Object.values(hist));

  return (
    <aside className={`trust ${moved ? "moved" : ""}`}>
      <h3>{moved ? "We corrected this rating" : "This rating checks out"}</h3>
      <p className="lede">
        {moved
          ? up
            ? "Reviews on this product look like a competitor attack campaign. With them removed, it rates higher than the shop was showing."
            : "Reviews on this product look paid for. With them removed, it rates lower than the shop was showing."
          : detail.total_withheld > 0
            ? "Some reviews here look inauthentic, but removing them does not meaningfully change the score."
            : "Nothing on this product looks inauthentic."}
      </p>

      {moved ? (
        <div className="trust-nums">
          <div className="trust-num was">
            <span className="k">As listed</span>
            <span className="v">{p.raw_rating?.toFixed(2) ?? "—"}</span>
          </div>
          <span className="trust-arw">→</span>
          <div className={`trust-num now ${up ? "up" : "down"}`}>
            <span className="k">Corrected</span>
            <span className="v">{p.adjusted_rating?.toFixed(2) ?? "—"}</span>
          </div>
        </div>
      ) : (
        <div className="trust-nums">
          <div className="trust-num">
            <span className="k">Rating</span>
            <span className="v">{(p.adjusted_rating ?? p.raw_rating)?.toFixed(2) ?? "—"}</span>
          </div>
          <Stars value={p.adjusted_rating ?? p.raw_rating} size={16} />
        </div>
      )}

      <div className="bars">
        {["5", "4", "3", "2", "1"].map((star) => (
          <div className="bar" key={star}>
            <span className="bar-k">{star}</span>
            <span className="bar-t">
              <span className="bar-f"
                    style={{ width: `${(100 * (hist[star] ?? 0)) / maxBar}%` }} />
            </span>
            <span className="bar-n">{hist[star] ?? 0}</span>
          </div>
        ))}
      </div>

      <div className="trust-line">
        Rated by <b>{p.n_shown.toLocaleString()}</b> reviews.{" "}
        {detail.total_withheld > 0 ? (
          <>
            <b>{detail.total_withheld.toLocaleString()}</b> more scored at or
            above {detail.auto_remove_at.toFixed(2)} and were withheld.
          </>
        ) : (
          <>None were withheld.</>
        )}
        {p.n_under_review > 0 ? (
          <>
            {" "}Another <b>{p.n_under_review}</b>{" "}
            {p.n_under_review === 1 ? "is" : "are"} flagged for a human to look
            at and are still counted above — a model's suspicion is not grounds
            for removing a review.
          </>
        ) : null}
        {p.adjusted_thin ? (
          <>
            {" "}Treat the corrected figure with care: it rests on fewer than{" "}
            {detail.product.n_shown + 1} surviving reviews.
          </>
        ) : null}
      </div>
    </aside>
  );
}
