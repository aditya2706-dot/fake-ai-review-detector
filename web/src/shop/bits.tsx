/** Small shared pieces of the storefront: money, stars, photos, product tiles. */

import { href, navigate } from "../router";
import type { ProductCard } from "../types";

export function money(value: number, currency = "USD"): string {
  try {
    return new Intl.NumberFormat(undefined, {
      style: "currency", currency, maximumFractionDigits: 2,
      // Without this a browser in a non-US locale renders USD as "US$21.99",
      // which reads as a currency conversion on what is meant to be a price tag.
      currencyDisplay: "narrowSymbol",
    }).format(value);
  } catch {
    return `${currency} ${value.toFixed(2)}`;
  }
}

export function when(iso: string): string {
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? iso
    : d.toLocaleDateString(undefined, { year: "numeric", month: "long", day: "numeric" });
}

/**
 * Five stars with a fractional fill.
 *
 * Two stacked copies of the same glyphs, the filled one clipped to a
 * percentage width. Rounding to the nearest half-star instead would hide the
 * thing this whole shop is about: a rating moving by 0.2 of a star is exactly
 * the size of change the detector produces on most products.
 */
export function Stars({ value, size = 14 }: { value: number | null; size?: number }) {
  if (value === null || Number.isNaN(value)) {
    return <span className="stars empty" style={{ fontSize: size }}>—</span>;
  }
  const pct = Math.max(0, Math.min(100, (value / 5) * 100));
  return (
    <span className="stars" style={{ fontSize: size }}
          aria-label={`${value.toFixed(2)} out of 5`} role="img">
      <span className="stars-bg" aria-hidden="true">★★★★★</span>
      <span className="stars-fg" style={{ width: `${pct}%` }} aria-hidden="true">★★★★★</span>
    </span>
  );
}

/**
 * A product photograph.
 *
 * `loading="lazy"` matters here: a listing page is 24 square images and the
 * grid below the fold should not compete with the ones above it. The alt text
 * is the product name because that is what the image is of -- "product photo"
 * would be noise in a screen reader running down a grid.
 */
export function Photo({ src, alt }: { src: string; alt: string }) {
  return <img src={src} alt={alt} loading="lazy" decoding="async" />;
}

/** The rating a shopper is shown: cleaned where there is one, raw otherwise. */
export function shownRating(p: ProductCard): number | null {
  return p.adjusted_rating ?? p.raw_rating;
}

/** The one-line trust note under a tile. Absent when nothing moved. */
export function TrustNote({ product }: { product: ProductCard }) {
  if (!product.material || product.delta === null) return null;
  const up = product.delta > 0;
  return (
    <div className={`tile-trust ${up ? "up" : "down"}`}>
      <span>{up ? "↑" : "↓"}</span>
      <span>{Math.abs(product.delta).toFixed(2)} after removing fake reviews</span>
    </div>
  );
}

export function Tile({ product }: { product: ProductCard }) {
  const rating = shownRating(product);
  const flag = product.material && product.delta !== null
    ? (product.delta > 0 ? "up" : "down") : product.on_sale ? "sale" : null;

  return (
    <a className="tile" href={href({ name: "product", slug: product.slug })}>
      <div className="tile-img">
        {flag ? (
          <span className={`flag ${flag}`}>
            {flag === "sale" ? "Sale"
              : flag === "up" ? "Rating was suppressed"
              : "Rating was inflated"}
          </span>
        ) : null}
        <Photo src={product.image} alt={product.name} />
      </div>
      <div className="tile-brand">{product.brand}</div>
      <div className="tile-name">{product.name}</div>
      <div className="tile-rate">
        <Stars value={rating} size={12} />
        <span>
          {rating === null ? "no rating" : rating.toFixed(1)}
          {" · "}
          {product.n_reviews.toLocaleString()}
        </span>
      </div>
      <div className="tile-price">
        {money(product.price, product.currency)}
        {product.on_sale && product.compare_at !== null ? (
          <span className="tile-was">{money(product.compare_at, product.currency)}</span>
        ) : null}
      </div>
      <TrustNote product={product} />
    </a>
  );
}

export function MoverCard({ product }: { product: ProductCard }) {
  const up = (product.delta ?? 0) > 0;
  return (
    <button className="mover"
            onClick={() => navigate({ name: "product", slug: product.slug })}>
      <div className="mover-img"><Photo src={product.image} alt={product.name} /></div>
      <div className="mover-b">
        <div className="mover-n">{product.name}</div>
        <div className="mover-m">
          <span className="was">{product.raw_rating?.toFixed(2)}</span>
          <span className="arw">→</span>
          <span className={`now ${up ? "up" : "down"}`}>
            {product.adjusted_rating?.toFixed(2)}
          </span>
          <Stars value={product.adjusted_rating} size={11} />
        </div>
        <div className="mover-t">
          {up ? "Pushed down by an attack campaign" : "Inflated by paid reviews"}
          {" · "}
          {Math.round(100 * (product.withheld_share ?? 0))}% of reviews withheld
        </div>
      </div>
    </button>
  );
}

export const ICON = {
  search: (
    <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor"
         strokeWidth="1.6" aria-hidden="true">
      <circle cx="7" cy="7" r="4.5" /><path d="M10.5 10.5L14 14" strokeLinecap="round" />
    </svg>
  ),
  bag: (
    <svg width="19" height="19" viewBox="0 0 20 20" fill="none" stroke="currentColor"
         strokeWidth="1.5" aria-hidden="true">
      <path d="M4 6h12l-1 10H5L4 6z" strokeLinejoin="round" />
      <path d="M7.5 6V4.5a2.5 2.5 0 015 0V6" strokeLinecap="round" />
    </svg>
  ),
  user: (
    <svg width="19" height="19" viewBox="0 0 20 20" fill="none" stroke="currentColor"
         strokeWidth="1.5" aria-hidden="true">
      <circle cx="10" cy="7" r="3" /><path d="M4 17c0-3.3 2.7-5 6-5s6 1.7 6 5" strokeLinecap="round" />
    </svg>
  ),
};
