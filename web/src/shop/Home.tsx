/**
 * The shop front.
 *
 * It leads with the catalogue, not with the detector. That ordering is the
 * whole point: a shop whose home page opens with a lecture about fake reviews
 * is a demo wearing a shop costume, and the interesting claim -- that ratings
 * you would otherwise trust are wrong -- only lands if the shop looks like
 * somewhere you would actually buy a kettle.
 *
 * The "ratings we corrected" rail sits below the fold for the same reason, and
 * it ranks by how far a rating moved rather than by how many reviews were
 * withheld, because the shopper's question is "would this have changed my
 * mind", not "how busy was the moderation queue".
 */

import { useEffect, useState } from "react";
import { api, describe } from "../api";
import { href } from "../router";
import type { ProductCard, ShopStats } from "../types";
import { MoverCard, Tile } from "./bits";

export function Home({ stats }: { stats: ShopStats }) {
  const [featured, setFeatured] = useState<ProductCard[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let live = true;
    api.products({ limit: 8, offset: 0, sort: "featured" })
      .then((p) => { if (live) setFeatured(p.items); })
      .catch((e) => { if (live) setError(describe(e)); });
    return () => { live = false; };
  }, []);

  const c = stats.counts;
  const pct = stats.n_reviews ? (100 * stats.n_withheld) / stats.n_reviews : 0;

  return (
    <>
      <section className="hero">
        <div className="wrap hero-in">
          <div>
            <h1>Ratings you can actually read.</h1>
            <p>
              Everything in this shop is rated twice — once as the reviews came
              in, and once with the paid ones taken out. Where those two numbers
              disagree, we show you both.
            </p>
            <div className="hero-cta">
              <a className="btn" href={href({ name: "department", key: "electronics" })}>
                Shop the catalogue
              </a>
              <a className="btn ghost" href="#corrected">See what we corrected</a>
            </div>
          </div>
          <dl className="hero-facts">
            <div>
              <dt>Products in the shop</dt>
              <dd>{c.n_listed.toLocaleString()}</dd>
            </div>
            <div>
              <dt>Reviews checked</dt>
              <dd>{stats.n_reviews.toLocaleString()}</dd>
            </div>
            <div>
              <dt>Withheld as fake</dt>
              <dd>{stats.n_withheld.toLocaleString()} ({pct.toFixed(1)}%)</dd>
            </div>
            <div>
              <dt>Ratings materially wrong</dt>
              <dd>{c.n_material}</dd>
            </div>
          </dl>
        </div>
      </section>

      <div className="wrap">
        <section className="sec">
          <div className="sec-head">
            <div><h2>Departments</h2></div>
          </div>
          <div className="depts">
            {stats.departments.map((d) => (
              <a key={d.key} className="dept" href={href({ name: "department", key: d.key })}>
                <b>{d.name}</b>
                <span>{d.tagline}</span>
              </a>
            ))}
          </div>
        </section>

        <section className="sec">
          <div className="sec-head">
            <div><h2>New in</h2></div>
            <a href={href({ name: "department", key: "electronics" })}>Browse everything</a>
          </div>
          {error ? <div className="err">{error}</div> : null}
          {featured === null ? (
            <div className="spin">Loading the catalogue…</div>
          ) : (
            <div className="tiles">
              {featured.map((p) => <Tile key={p.slug} product={p} />)}
            </div>
          )}
        </section>

        <section className="sec" id="corrected">
          <div className="sec-head">
            <div>
              <h2>Ratings we corrected</h2>
              <p>
                {c.n_material} of the {c.n_rankable} products with enough
                reviews to judge have a rating that moves by at least{" "}
                {stats.material_delta} of a star once suspected fakes come out —{" "}
                {c.n_inflated} that were inflated by paid reviews and{" "}
                {c.n_attacked} that were pushed down by attack campaigns. On the
                typical product it changes almost nothing: the median rating
                moves{" "}
                {stats.median_delta === null
                  ? "—"
                  : Math.abs(stats.median_delta).toFixed(3)}{" "}
                of a star. Manipulation is concentrated, not spread, so this is
                where it went rather than a claim about everything.
              </p>
            </div>
          </div>
          <div className="rail">
            {stats.movers.slice(0, 4).map((p) => (
              <MoverCard key={p.slug} product={p} />
            ))}
          </div>
        </section>
      </div>
    </>
  );
}
