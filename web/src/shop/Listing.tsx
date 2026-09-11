/**
 * A department, or a set of search results.
 *
 * One component for both, because they differ only in which filter is pinned
 * and what the heading says. Paging and sorting are server-side -- the sort
 * options include "most corrected", which the client could not compute without
 * holding the whole catalogue and re-deriving the policy.
 */

import { useEffect, useRef, useState } from "react";
import { api, describe } from "../api";
import { type Route, href, navigate } from "../router";
import type { ProductPage, ShopStats } from "../types";
import { Tile } from "./bits";

const PAGE = 24;

const SORTS: [string, string][] = [
  ["featured", "Featured"],
  ["reviews", "Most reviewed"],
  ["manipulation", "Most corrected"],
  ["rating", "Highest rated"],
  ["price", "Price: low to high"],
  ["price_desc", "Price: high to low"],
  ["name", "Name"],
];

export function Listing({ route, stats }: { route: Route; stats: ShopStats }) {
  const category = route.name === "department" ? route.key : "";
  const search = route.name === "search" ? route.q : "";

  const [page, setPage] = useState<ProductPage | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [sort, setSort] = useState("featured");
  const [offset, setOffset] = useState(0);
  const reqId = useRef(0);

  useEffect(() => { setOffset(0); }, [category, search, sort]);

  useEffect(() => {
    const mine = ++reqId.current;
    setLoading(true);
    api.products({
      limit: PAGE, offset, sort,
      category: category || undefined,
      search: search || undefined,
      // The "most corrected" ranking is meaningless on a two-review product,
      // and the server already refuses to rank them. Asking for the same floor
      // keeps the count in the pager honest about what is being listed.
      min_reviews: sort === "manipulation" ? stats.min_reviews_for_ranking : undefined,
    })
      .then((p) => {
        if (mine !== reqId.current) return;      // a newer query has been issued
        setPage(p);
        setError(null);
      })
      .catch((e) => { if (mine === reqId.current) setError(describe(e)); })
      .finally(() => { if (mine === reqId.current) setLoading(false); });
  }, [offset, sort, category, search, stats.min_reviews_for_ranking]);

  const dept = stats.departments.find((d) => d.key === category);
  const title = search ? `“${search}”` : dept ? dept.name : "Everything";
  const total = page?.total ?? 0;

  return (
    <div className="wrap listing">
      <aside className="facets">
        <div className="facet">
          <h4>Departments</h4>
          <a className={!category && !search ? "on" : ""} href={href({ name: "home" })}>
            All
          </a>
          {stats.departments.map((d) => (
            <a key={d.key} className={category === d.key ? "on" : ""}
               href={href({ name: "department", key: d.key })}>
              {d.name}
            </a>
          ))}
        </div>
      </aside>

      <div>
        <div className="listing-head">
          <div>
            <h1>{title}</h1>
            <p>
              {search ? "Search results" : dept ? dept.tagline : "The whole catalogue"}
              {page ? ` · ${total} product${total === 1 ? "" : "s"}` : ""}
            </p>
          </div>
          <div className="listing-tools">
            <select className="field" value={sort} aria-label="Sort products"
                    onChange={(e) => setSort(e.target.value)}>
              {SORTS.map(([v, label]) => <option key={v} value={v}>{label}</option>)}
            </select>
          </div>
        </div>

        {error ? <div className="err">{error}</div> : null}

        {loading && !page ? (
          <div className="spin">Loading…</div>
        ) : page && page.items.length === 0 ? (
          <div className="empty">
            <p>Nothing matches that.</p>
            {sort === "manipulation" ? (
              <p className="note">
                The “most corrected” ranking only covers products with at least{" "}
                {stats.min_reviews_for_ranking} reviews.
              </p>
            ) : null}
            <button className="btn ghost sm" onClick={() => navigate({ name: "home" })}>
              Back to the shop
            </button>
          </div>
        ) : (
          <>
            <div className={`tiles ${loading ? "stale" : ""}`}>
              {page?.items.map((p) => <Tile key={p.slug} product={p} />)}
            </div>
            {total > PAGE ? (
              <div className="pager">
                <button className="btn ghost sm" disabled={offset === 0}
                        onClick={() => setOffset(Math.max(0, offset - PAGE))}>
                  Previous
                </button>
                <span>
                  {offset + 1}–{Math.min(offset + PAGE, total)} of {total}
                </span>
                <button className="btn ghost sm" disabled={offset + PAGE >= total}
                        onClick={() => setOffset(offset + PAGE)}>
                  Next
                </button>
              </div>
            ) : null}
          </>
        )}
      </div>
    </div>
  );
}
