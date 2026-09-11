/**
 * A product page: the listing, the buy box, and the reviews with the removed
 * ones shown alongside rather than silently gone.
 *
 * WHY THE WITHHELD REVIEWS ARE ON THE PAGE AT ALL
 * A shop that quietly deletes reviews is asking to be trusted. Putting them
 * behind a disclosure, struck through, with the moderator reason codes
 * attached, turns the claim into something a shopper can check -- and makes
 * the failure mode visible too, because some of them plainly do not look fake
 * and that is the honest cost of running a detector at 94% precision.
 */

import { useEffect, useMemo, useState } from "react";
import { api, describe } from "../api";
import { href } from "../router";
import { useShop } from "../state/shop";
import type { ProductDetail, ShopReview, Variant } from "../types";
import { Photo, Stars, Tile, money, when } from "./bits";
import { TrustPanel } from "./TrustPanel";

export function Product({ slug }: { slug: string }) {
  const { addToCart, busy } = useShop();
  const [detail, setDetail] = useState<ProductDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [choices, setChoices] = useState<Record<string, string>>({});
  const [qty, setQty] = useState(1);
  const [added, setAdded] = useState(false);
  const [sort, setSort] = useState("helpful");

  useEffect(() => {
    let live = true;
    setDetail(null);
    setError(null);
    api.product(slug, { limit: 8, sort })
      .then((d) => {
        if (!live) return;
        setDetail(d);
        // Default every option group to its first value, so the buy button is
        // live on arrival rather than after three clicks the shopper did not
        // know were required.
        setChoices((prev) =>
          Object.keys(prev).length ? prev
            : Object.fromEntries(d.product.options.map((o) => [o.name, o.values[0]])));
      })
      .catch((e) => { if (live) setError(describe(e)); });
    return () => { live = false; };
  }, [slug, sort]);

  const product = detail?.product;

  const variant: Variant | null = useMemo(() => {
    if (!product) return null;
    if (product.options.length === 0) return product.variants[0] ?? null;
    return product.variants.find((v) =>
      product.options.every((o) => v.choices[o.name] === choices[o.name])) ?? null;
  }, [product, choices]);

  useEffect(() => { setAdded(false); setQty(1); }, [variant?.sku]);

  if (error) {
    return (
      <div className="wrap narrow">
        <div className="err">{error}</div>
        <p style={{ marginTop: 16 }}>
          <a className="btn ghost sm" href={href({ name: "home" })}>Back to the shop</a>
        </p>
      </div>
    );
  }
  if (!detail || !product) return <div className="spin">Loading…</div>;

  const stock = variant?.stock ?? 0;
  const canBuy = variant !== null && stock > 0;
  const rating = product.adjusted_rating ?? product.raw_rating;

  const add = async () => {
    if (!variant) return;
    try {
      await addToCart(product.slug, variant.sku, qty);
      setAdded(true);
    } catch {
      /* the provider already surfaced it */
    }
  };

  return (
    <>
      <div className="wrap">
        <div className="pdp">
          <div className="pdp-media">
            <div className="pdp-img"><Photo src={product.image} alt={product.name} /></div>
          </div>

          <div>
            <div className="crumbs">
              <a href={href({ name: "home" })}>Shop</a>
              <span>/</span>
              <a href={href({ name: "department", key: product.category })}>
                {product.department}
              </a>
            </div>

            <div className="pdp-brand">{product.brand}</div>
            <h1>{product.name}</h1>
            <p className="pdp-sum">{product.summary}</p>

            <div className="pdp-rate">
              <Stars value={rating} size={15} />
              <b>{rating === null ? "—" : rating.toFixed(2)}</b>
              <a href="#reviews">
                {product.n_shown.toLocaleString()} review
                {product.n_shown === 1 ? "" : "s"}
              </a>
              {product.material ? (
                <span className={`tag ${product.delta! > 0 ? "ver" : "warn"}`}>
                  {product.delta! > 0 ? "rating corrected up" : "rating corrected down"}
                </span>
              ) : null}
            </div>

            <div className="pdp-price">
              <span className="now">{money(product.price, product.currency)}</span>
              {product.on_sale && product.compare_at !== null ? (
                <>
                  <span className="was">{money(product.compare_at, product.currency)}</span>
                  <span className="off">
                    Save {money(product.compare_at - product.price, product.currency)}
                  </span>
                </>
              ) : null}
            </div>

            {product.options.length > 0 ? (
              <div className="opts">
                {product.options.map((o) => (
                  <div className="opt" key={o.name}>
                    <div className="opt-h">
                      <b>{o.name}</b>
                      <span>{choices[o.name]}</span>
                    </div>
                    <div className="pills">
                      {o.values.map((v) => (
                        <button key={v}
                                className={`pill ${choices[o.name] === v ? "on" : ""}`}
                                onClick={() => setChoices({ ...choices, [o.name]: v })}>
                          {v}
                        </button>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            ) : null}

            <div className="buy">
              <div className="qty">
                <button onClick={() => setQty(Math.max(1, qty - 1))} disabled={qty <= 1}
                        aria-label="Decrease quantity">−</button>
                <span>{qty}</span>
                <button onClick={() => setQty(Math.min(10, stock, qty + 1))}
                        disabled={qty >= Math.min(10, stock)}
                        aria-label="Increase quantity">+</button>
              </div>
              <button className="btn wide" onClick={add} disabled={!canBuy || busy}>
                {!canBuy ? "Out of stock" : added ? "Added to basket ✓" : "Add to basket"}
              </button>
            </div>

            {added ? (
              <p className="stock">
                <a className="btn ghost sm" href={href({ name: "cart" })}>
                  View basket
                </a>
              </p>
            ) : (
              <p className={`stock ${stock > 0 && stock <= 5 ? "low" : ""}`}>
                {stock === 0 ? "Currently unavailable"
                  : stock <= 5 ? `Only ${stock} left`
                  : "In stock"}
                {" · "}Free delivery over $75{" · "}SKU {variant?.sku ?? product.sku}
              </p>
            )}

            <div className="pdp-sec">
              <h3>Description</h3>
              <p>{product.description}</p>
              <ul>{product.bullets.map((b) => <li key={b}>{b}</li>)}</ul>
            </div>

            {Object.keys(product.specs).length > 0 ? (
              <div className="pdp-sec">
                <h3>Specification</h3>
                <dl className="specs">
                  {Object.entries(product.specs).map(([k, v]) => (
                    <div key={k}><dt>{k}</dt><dd>{v}</dd></div>
                  ))}
                </dl>
              </div>
            ) : null}
          </div>
        </div>
      </div>

      <div className="wrap" id="reviews">
        <div className="rev-wrap">
          <div className="rev-grid">
            <TrustPanel detail={detail} />

            <div>
              <div className="listing-head" style={{ marginBottom: 4 }}>
                <div><h2>Reviews</h2></div>
                <select className="field listing-tools" value={sort}
                        aria-label="Sort reviews"
                        style={{ width: "auto", height: 36, fontSize: 13.5 }}
                        onChange={(e) => setSort(e.target.value)}>
                  <option value="helpful">Most helpful</option>
                  <option value="newest">Newest</option>
                  <option value="rating">Highest rated</option>
                  <option value="rating_asc">Lowest rated</option>
                </select>
              </div>

              {detail.reviews.map((r) => <Review key={r.review_id} review={r} />)}

              {detail.total_shown > detail.reviews.length ? (
                <p className="note" style={{ paddingTop: 14 }}>
                  Showing {detail.reviews.length} of{" "}
                  {detail.total_shown.toLocaleString()} reviews.
                </p>
              ) : null}

              {detail.withheld.length > 0 ? (
                <details className="withheld">
                  <summary>
                    <span>
                      {detail.total_withheld.toLocaleString()} review
                      {detail.total_withheld === 1 ? "" : "s"} withheld from this rating
                    </span>
                    <span className="muted">show</span>
                  </summary>
                  <div className="withheld-b">
                    <p className="note" style={{ padding: "10px 0 2px" }}>
                      These scored at or above {detail.auto_remove_at.toFixed(2)},
                      the point at which this shop removes a review without
                      asking a moderator. They are shown because a shop that
                      silently deletes reviews is asking to be taken on
                      trust — and because at this threshold roughly one in
                      seventeen is removed wrongly, which you can only judge by
                      reading them.
                    </p>
                    {detail.withheld.map((r) => (
                      <Review key={r.review_id} review={r} withheld />
                    ))}
                  </div>
                </details>
              ) : null}
            </div>
          </div>
        </div>
      </div>

      {detail.related.length > 0 ? (
        <div className="wrap">
          <section className="sec">
            <div className="sec-head"><div><h2>More in {product.department}</h2></div></div>
            <div className="tiles">
              {detail.related.map((p) => <Tile key={p.slug} product={p} />)}
            </div>
          </section>
        </div>
      ) : null}
    </>
  );
}

function Review({ review, withheld = false }:
                { review: ShopReview; withheld?: boolean }) {
  return (
    <article className="rev">
      <div className="rev-h">
        <Stars value={review.rating} size={13} />
        <span className="who">{review.reviewer_id}</span>
        <span className="when">{when(review.timestamp)}</span>
        {review.verified_purchase ? <span className="tag ver">Verified purchase</span> : null}
        {withheld ? <span className="tag warn">Withheld · {review.score.toFixed(2)}</span> : null}
        {/* The model was fitted on this period, so its score here is flattered.
            Marked rather than hidden -- see storefront.py. */}
        {review.in_train_period ? <span className="tag fit">fitted period</span> : null}
      </div>
      {review.title ? <div className="rev-t">{review.title}</div> : null}
      <p className="rev-b">{review.text}</p>
      <div className="rev-f">
        {review.helpful_votes > 0 ? (
          <span>{review.helpful_votes} found this helpful</span>
        ) : null}
        {withheld && review.reasons?.length ? (
          <div className="reasons">
            {review.reasons.map((r) => <span className="reason" key={r}>{r}</span>)}
          </div>
        ) : null}
      </div>
    </article>
  );
}
