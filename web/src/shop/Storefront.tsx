/**
 * The storefront home: the catalogue, plus the finding that justifies the page.
 *
 * The "where cleaning changes the answer" lane exists because the headline
 * number on this corpus is boring and the distribution is not. The median
 * product's rating moves by hundredths of a star; a hundred-odd products move
 * enough to change a buying decision, in both directions. A grid sorted by
 * popularity would never show you one.
 */

import { useEffect, useRef, useState } from "react";
import { ApiError, api } from "../api";
import { navigate } from "../router";
import type { ProductCard, ProductPage, ShopStats } from "../types";
import { ProductArt, Stars, TrustChip, money } from "./bits";

const PAGE = 24;

export function Storefront({ stats }: { stats: ShopStats }) {
  const [page, setPage] = useState<ProductPage | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const [category, setCategory] = useState("");
  const [search, setSearch] = useState("");
  const [sort, setSort] = useState("reviews");
  const [offset, setOffset] = useState(0);

  const debounced = useDebounced(search, 250);
  const reqId = useRef(0);

  useEffect(() => { setOffset(0); }, [category, debounced, sort]);

  useEffect(() => {
    const mine = ++reqId.current;
    setLoading(true);
    api
      .products({
        limit: PAGE, offset, sort,
        category: category || undefined,
        search: debounced || undefined,
        // The manipulation ranking is meaningless on two-review products, and
        // the server already refuses to rank them; asking for the same floor
        // keeps the count in the pager honest about what is being listed.
        min_reviews: sort === "manipulation" ? stats.min_reviews_for_ranking : undefined,
      })
      .then((p) => {
        if (mine !== reqId.current) return;   // a newer query has been issued
        setPage(p);
        setError(null);
      })
      .catch((e) => { if (mine === reqId.current) setError(describe(e)); })
      .finally(() => { if (mine === reqId.current) setLoading(false); });
  }, [offset, sort, category, debounced, stats.min_reviews_for_ranking]);

  const c = stats.counts;
  return (
    <div className="shop-page">
      <Finding stats={stats} />

      <section className="lane">
        <div className="lane-head">
          <h2>Where the reviews are lying</h2>
          <p>
            The {c.n_material} products of {c.n_rankable.toLocaleString()} whose
            rating moves by at least {stats.material_delta} of a star once
            suspected fakes are withheld — {c.n_inflated} inflated by paid
            reviews, {c.n_attacked} pushed down by attack campaigns.
          </p>
        </div>
        <div className="lane-strip">
          {stats.movers.map((p) => <MoverCard key={p.product_id} product={p} />)}
        </div>
      </section>

      <section className="catalogue">
        <div className="toolbar shop-toolbar">
          <input
            className="field search"
            placeholder="Search the catalogue"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
          <select className="field" value={category} onChange={(e) => setCategory(e.target.value)}>
            <option value="">All departments</option>
            {stats.categories.map((k) => (
              <option key={k} value={k}>{k.replace(/_/g, " ")}</option>
            ))}
          </select>
          <select className="field" value={sort} onChange={(e) => setSort(e.target.value)}>
            <option value="reviews">Most reviewed</option>
            <option value="manipulation">Most manipulated</option>
            <option value="rating">Highest cleaned rating</option>
            <option value="rating_asc">Lowest cleaned rating</option>
            <option value="price">Price: low to high</option>
            <option value="price_desc">Price: high to low</option>
            <option value="name">Name</option>
          </select>
          <span style={{ flex: 1 }} />
          <span className="pager-count">
            {page
              ? `${page.total ? offset + 1 : 0}–${Math.min(offset + PAGE, page.total)} of ${page.total.toLocaleString()}`
              : "—"}
          </span>
          <button className="ghost" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - PAGE))}>
            Prev
          </button>
          <button
            className="ghost"
            disabled={!page || offset + PAGE >= page.total}
            onClick={() => setOffset(offset + PAGE)}
          >
            Next
          </button>
        </div>

        {error ? <div className="banner err">{error}</div> : null}

        {loading && !page ? (
          <div className="spinner">Loading the catalogue&hellip;</div>
        ) : page && page.items.length === 0 ? (
          <div className="empty">
            Nothing matches that. {sort === "manipulation"
              ? `The manipulation ranking only covers products with at least ${stats.min_reviews_for_ranking} reviews.`
              : ""}
          </div>
        ) : (
          <div className={`grid ${loading ? "stale" : ""}`}>
            {page?.items.map((p) => <GridCard key={p.product_id} product={p} />)}
          </div>
        )}
      </section>
    </div>
  );
}

function Finding({ stats }: { stats: ShopStats }) {
  const pct = stats.n_reviews
    ? (100 * stats.n_withheld) / stats.n_reviews
    : 0;
  return (
    <section className="finding">
      <div>
        <h1>Every rating on this store is shown twice</h1>
        <p>
          Once as the marketplace lists it, and once with the reviews the
          detector will not stand behind taken out. Across{" "}
          {stats.counts.n_products.toLocaleString()} products and{" "}
          {stats.n_reviews.toLocaleString()} reviews it withholds{" "}
          {stats.n_withheld.toLocaleString()} ({pct.toFixed(1)}%).
        </p>
        <p className="finding-caveat">
          On the typical product that changes almost nothing — the median rating
          moves {stats.median_delta === null ? "—" : Math.abs(stats.median_delta).toFixed(3)} of
          a star. Manipulation is concentrated, not spread, so the honest
          version of this page shows you where rather than claiming everywhere.
        </p>
      </div>
      <dl className="finding-stats">
        <div><dt>Products checked</dt><dd>{stats.counts.n_products.toLocaleString()}</dd></div>
        <div><dt>Ratings materially wrong</dt><dd>{stats.counts.n_material}</dd></div>
        <div><dt>Inflated / attacked</dt><dd>{stats.counts.n_inflated} / {stats.counts.n_attacked}</dd></div>
        <div><dt>Withheld at score</dt><dd>≥ {stats.auto_remove_at.toFixed(2)}</dd></div>
      </dl>
    </section>
  );
}

function MoverCard({ product }: { product: ProductCard }) {
  const up = (product.delta ?? 0) > 0;
  return (
    <button className="mover" onClick={() => navigate({ name: "product", id: product.product_id })}>
      <ProductArt product={product} size={96} />
      <div className="mover-body">
        <div className="mover-name">{product.name}</div>
        <div className="mover-move">
          <span className="was">
            <Stars value={product.raw_rating} size={12} /> {product.raw_rating?.toFixed(2)}
          </span>
          <span className="arrow">→</span>
          <span className={`now ${up ? "up" : "down"}`}>
            <Stars value={product.adjusted_rating} size={12} /> {product.adjusted_rating?.toFixed(2)}
          </span>
        </div>
        <div className={`mover-tag ${up ? "up" : "down"}`}>
          {up ? "attacked" : "inflated"} · {Math.round(100 * (product.withheld_share ?? 0))}% withheld
        </div>
      </div>
    </button>
  );
}

function GridCard({ product }: { product: ProductCard }) {
  const shown = product.adjusted_rating ?? product.raw_rating;
  return (
    <button className="card" onClick={() => navigate({ name: "product", id: product.product_id })}>
      <ProductArt product={product} />
      <div className="card-body">
        <div className="card-brand">{product.brand}</div>
        <div className="card-name">{product.name}</div>
        <div className="card-rating">
          <Stars value={shown} />
          <span className="muted">
            {shown === null ? "no rating" : shown.toFixed(2)} · {product.n_reviews.toLocaleString()}
          </span>
        </div>
        <div className="card-price">{money(product.price, product.currency)}</div>
        <TrustChip product={product} />
      </div>
    </button>
  );
}

function describe(e: unknown): string {
  if (e instanceof ApiError) return `${e.status}: ${e.message}`;
  if (e instanceof Error) return e.message;
  return String(e);
}

function useDebounced<T>(value: T, ms: number): T {
  const [v, setV] = useState(value);
  useEffect(() => {
    const t = setTimeout(() => setV(value), ms);
    return () => clearTimeout(t);
  }, [value, ms]);
  return v;
}
