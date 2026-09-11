/**
 * The shell: two surfaces over one API.
 *
 * `shop/` is what a customer sees -- ratings with suspected fakes withheld.
 * `console/` is what a moderator sees -- the ranked queue and the evidence
 * behind each flag. They are the same scores from the same corpus cache; the
 * difference is entirely in what each audience is allowed to act on.
 *
 * The storefront needs `/shop/stats` before it can render anything (categories,
 * the policy constants, the movers lane), and that first request is the one
 * that pays the ~20s corpus scoring cost. So the wait, the retry and the
 * "no model / no corpus" messages live here, once, rather than in each surface.
 */

import { useEffect, useState } from "react";
import { ApiError, api } from "./api";
import { Console } from "./console/Console";
import { type Route, href, useRoute } from "./router";
import { ProductPage } from "./shop/ProductPage";
import { Storefront } from "./shop/Storefront";
import type { Health, ShopStats } from "./types";  // Health types the probe below

export default function App() {
  const route = useRoute();
  const [shop, setShop] = useState<ShopStats | null>(null);
  const [fatal, setFatal] = useState<string | null>(null);
  const [waiting, setWaiting] = useState(false);

  useEffect(() => {
    let live = true;
    (async () => {
      // `make up` starts the API and this dev server together and the API is
      // the slower of the two -- it loads a model before it binds. Retrying
      // reachability is correct; anything the API actually answers is final.
      const deadline = Date.now() + 45_000;
      let h: Health | null = null;
      while (live && h === null) {
        try {
          h = await api.health();
        } catch (e) {
          if (Date.now() > deadline) {
            if (live) setFatal(describe(e));
            return;
          }
          if (live) setWaiting(true);
          await new Promise((r) => setTimeout(r, 1000));
        }
      }
      if (!live || h === null) return;
      setWaiting(false);
      if (!h.model_loaded) {
        setFatal("No model is loaded. Run `make train`, then start the API again.");
        return;
      }
      try {
        const s = await api.shopStats();
        if (live) setShop(s);
      } catch (e) {
        if (live) setFatal(describe(e));
      }
    })();
    return () => { live = false; };
  }, []);

  const nav = <Nav route={route} />;

  if (route.name === "console") return <Console nav={nav} />;

  return (
    <div className="shop">
      <header className="shop-head">
        {nav}
        <a className="wordmark" href={href({ name: "shop" })}>Marketplace</a>
        <span style={{ flex: 1 }} />
        <span className="shop-head-note">
          a fake-review detector, seen from the customer's side
        </span>
      </header>

      {fatal ? (
        <div className="shop-page"><div className="banner err">{fatal}</div></div>
      ) : !shop ? (
        <div className="spinner">
          {waiting
            ? "Waiting for the API — it loads the model before it accepts requests."
            : "Scoring the corpus. The first request features and scores all 40,000 reviews (about 20 seconds); everything after it is served from that."}
        </div>
      ) : route.name === "product" ? (
        <ProductPage id={route.id} />
      ) : (
        <Storefront stats={shop} />
      )}

      {shop ? <Footer stats={shop} /> : null}
    </div>
  );
}

function Nav({ route }: { route: Route }) {
  const onShop = route.name !== "console";
  return (
    <div className="seg surface-switch">
      <a className={onShop ? "on" : ""} href={href({ name: "shop" })}>Shop</a>
      <a className={onShop ? "" : "on"} href={href({ name: "console" })}>Console</a>
    </div>
  );
}

function Footer({ stats }: { stats: ShopStats }) {
  return (
    <footer className="shop-foot">
      <p>
        Ratings on this store are recomputed with reviews scoring ≥{" "}
        {stats.auto_remove_at.toFixed(2)} withheld — the point at which this
        project's decision policy is willing to act without a moderator.
        Reviews between the queue threshold ({stats.threshold.toFixed(2)}) and
        that line are shown, marked, and sent to a human instead.
      </p>
      <p>
        The corpus is synthetic and every product listing is generated from its
        id, so nothing here describes a real product or a real person. Model
        scores before{" "}
        {new Date(stats.train_period_ends).toLocaleDateString()} come from the
        period the model was fitted on and are flattered; those reviews are
        marked <em>fitted period</em> wherever they appear.
      </p>
    </footer>
  );
}

function describe(e: unknown): string {
  if (e instanceof ApiError) return `${e.status}: ${e.message}`;
  if (e instanceof Error) return e.message;
  return String(e);
}
