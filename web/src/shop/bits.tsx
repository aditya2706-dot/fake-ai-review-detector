/** Small shared pieces of the storefront: stars, money, generated product art. */

import type { ProductCard } from "../types";

export function money(value: number, currency: string): string {
  try {
    return new Intl.NumberFormat(undefined, {
      style: "currency", currency, maximumFractionDigits: 2,
      // Without this a browser in a non-US locale renders USD as "US$21.99",
      // which reads as a currency conversion on what is meant to be a plain
      // price tag.
      currencyDisplay: "narrowSymbol",
    }).format(value);
  } catch {
    return `${currency} ${value.toFixed(2)}`;
  }
}

/**
 * Five stars with a fractional fill.
 *
 * Two stacked copies of the same glyphs, the filled one clipped to a
 * percentage width. Rounding to the nearest half-star instead would hide the
 * thing this whole page is about: a rating moving by 0.2 of a star is exactly
 * the size of change the detector produces on most products.
 */
export function Stars({ value, size = 15 }: { value: number | null; size?: number }) {
  if (value === null || Number.isNaN(value)) {
    return <span className="stars empty" style={{ fontSize: size }}>—</span>;
  }
  const pct = Math.max(0, Math.min(100, (value / 5) * 100));
  return (
    <span className="stars" style={{ fontSize: size }} aria-label={`${value.toFixed(2)} out of 5`}>
      <span className="stars-bg">★★★★★</span>
      <span className="stars-fg" style={{ width: `${pct}%` }}>★★★★★</span>
    </span>
  );
}

/**
 * Generated product art.
 *
 * The corpus has no images and this repo will not fetch any: an offline demo
 * that needs a CDN is not offline, and putting a real photograph next to a
 * fabricated review score attaches invented fraud to someone's actual product.
 * So each listing gets a deterministic gradient plus its category glyph, keyed
 * by the hue the server derived from the product id -- see
 * `frd.data.catalog.product_for` for why the hue comes from the server rather
 * than being re-hashed here.
 */
export function ProductArt(
  { product, size = 168 }: { product: ProductCard; size?: number },
) {
  const { hue, hue_shift, glyph, brand } = product;
  const initials = brand.split(/\s+/).map((w) => w[0]).join("").slice(0, 2).toUpperCase();
  return (
    <div
      className="art"
      style={{
        height: size,
        background:
          `linear-gradient(145deg, hsl(${hue} 62% 88%), hsl(${hue + hue_shift} 54% 78%))`,
      }}
      aria-hidden="true"
    >
      <span className="art-glyph" style={{ color: `hsl(${hue} 42% 38%)`, fontSize: size * 0.44 }}>
        {glyph}
      </span>
      <span className="art-mark" style={{ color: `hsl(${hue} 38% 32%)` }}>{initials}</span>
    </div>
  );
}

/** The one-line trust chip a grid card carries. Absent when nothing moved. */
export function TrustChip({ product }: { product: ProductCard }) {
  if (!product.material || product.delta === null) return null;
  const up = product.delta > 0;
  return (
    <span className={`chip ${up ? "up" : "down"}`}>
      {up ? "↑" : "↓"} {Math.abs(product.delta).toFixed(2)} after cleaning
    </span>
  );
}

export const DIRECTION_COPY: Record<string, { label: string; blurb: string }> = {
  inflated: {
    label: "Inflated by paid reviews",
    blurb: "withheld as suspected paid or incentivised reviews",
  },
  attacked: {
    label: "Targeted by an attack campaign",
    blurb: "withheld as a suspected competitor attack campaign",
  },
  unchanged: {
    label: "No material change",
    blurb: "withheld as suspected inauthentic",
  },
};
