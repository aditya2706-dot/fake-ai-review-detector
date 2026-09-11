/**
 * The shell: two surfaces over one API.
 *
 * `shop/` is what a customer sees -- a catalogue, a basket, a checkout, and
 * ratings with suspected fakes withheld. `console/` is what a moderator sees --
 * the ranked queue and the evidence behind each flag. They are the same scores
 * from the same corpus cache; the difference is entirely in what each audience
 * is allowed to act on.
 *
 * The storefront needs `/shop/stats` before it can render its chrome (the
 * department nav, the policy constants, the footer's terms), and that first
 * request is the one that pays the ~20s corpus scoring cost. So the wait, the
 * retry and the "no model / no corpus" messages live here, once, rather than
 * in each page.
 */

import { useEffect, useState } from "react";
import { api, describe } from "./api";
import { Console } from "./console/Console";
import { href, useRoute } from "./router";
import { Account, SignIn } from "./shop/Account";
import { Cart } from "./shop/Cart";
import { Checkout } from "./shop/Checkout";
import { Home } from "./shop/Home";
import { Layout } from "./shop/Layout";
import { Listing } from "./shop/Listing";
import { OrderPage, Orders } from "./shop/Orders";
import { Product } from "./shop/Product";
import { ShopProvider } from "./state/shop";
import type { Health, ShopStats } from "./types";

export default function App() {
  const route = useRoute();

  if (route.name === "console") {
    return (
      <Console
        nav={<a className="to-shop" href={href({ name: "home" })}>← Shop</a>}
      />
    );
  }

  return (
    <ShopProvider>
      <Shop />
    </ShopProvider>
  );
}

function Shop() {
  const route = useRoute();
  const [stats, setStats] = useState<ShopStats | null>(null);
  const [fatal, setFatal] = useState<string | null>(null);
  const [waiting, setWaiting] = useState(false);

  useEffect(() => {
    let live = true;
    (async () => {
      // `make up` starts the API and this dev server together and the API is
      // the slower of the two -- it loads a model before it binds. Retrying
      // reachability is correct; anything the API actually answers is final.
      const deadline = Date.now() + 45_000;
      let health: Health | null = null;
      while (live && health === null) {
        try {
          health = await api.health();
        } catch (e) {
          if (Date.now() > deadline) {
            if (live) setFatal(describe(e));
            return;
          }
          if (live) setWaiting(true);
          await new Promise((r) => setTimeout(r, 1000));
        }
      }
      if (!live || health === null) return;
      setWaiting(false);
      if (!health.model_loaded) {
        setFatal("No model is loaded. Run `make train`, then start the API again.");
        return;
      }
      try {
        const s = await api.shopStats();
        if (live) setStats(s);
      } catch (e) {
        if (live) setFatal(describe(e));
      }
    })();
    return () => { live = false; };
  }, []);

  return (
    <div className="shop">
      <Layout route={route} stats={stats}>
        {fatal ? (
          <div className="wrap narrow"><div className="err">{fatal}</div></div>
        ) : !stats ? (
          <div className="spin">
            {waiting
              ? "Waiting for the API — it loads the model before it accepts requests."
              : "Scoring the corpus. The first request features and scores all 40,000 reviews (about 20 seconds); everything after it is served from that."}
          </div>
        ) : (
          <Page />
        )}
      </Layout>
    </div>
  );

  function Page() {
    if (!stats) return null;
    switch (route.name) {
      case "product": return <Product slug={route.slug} />;
      case "department":
      case "search": return <Listing route={route} stats={stats} />;
      case "cart": return <Cart />;
      case "checkout": return <Checkout />;
      case "order": return <OrderPage id={route.id} />;
      case "orders": return <Orders />;
      case "account": return <Account />;
      case "signin": return <SignIn next={route.next} />;
      default: return <Home stats={stats} />;
    }
  }
}
