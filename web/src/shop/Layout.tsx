/**
 * The shop's chrome: header, department nav, search, footer.
 *
 * The footer is not decoration. This shop makes an unusual claim -- that some
 * of the ratings it shows are wrong and it has removed the reviews responsible
 * -- and a claim like that needs its terms stated somewhere permanent rather
 * than only next to the products where it happens to be flattering.
 */

import { type FormEvent, type ReactNode, useEffect, useState } from "react";
import { type Route, href, navigate } from "../router";
import { useShop } from "../state/shop";
import type { ShopStats } from "../types";
import { ICON } from "./bits";

export function Layout(
  { route, stats, children }:
  { route: Route; stats: ShopStats | null; children: ReactNode },
) {
  const { cart, user } = useShop();
  const [q, setQ] = useState(route.name === "search" ? route.q : "");

  // Keep the box in step with the URL, so a back button out of a search
  // clears the field instead of leaving a stale term sitting in it.
  useEffect(() => { setQ(route.name === "search" ? route.q : ""); },
            [route.name, route.name === "search" ? route.q : ""]);

  const submit = (e: FormEvent) => {
    e.preventDefault();
    navigate(q.trim() ? { name: "search", q: q.trim() } : { name: "home" });
  };

  const active = route.name === "department" ? route.key : null;

  return (
    <>
      <header className="sh-head">
        <div className="wrap sh-head-in">
          <a className="sh-logo" href={href({ name: "home" })}>
            Verity<span>&nbsp;Market</span>
          </a>

          <nav className="sh-nav">
            {(stats?.departments ?? []).map((d) => (
              <a key={d.key}
                 className={active === d.key ? "on" : ""}
                 href={href({ name: "department", key: d.key })}>
                {d.name}
              </a>
            ))}
          </nav>

          <div className="sh-head-r">
            <form className="sh-search" onSubmit={submit} role="search">
              {ICON.search}
              <input value={q} onChange={(e) => setQ(e.target.value)}
                     placeholder="Search" aria-label="Search products" />
            </form>
            <a className="sh-icon" href={href({ name: user ? "account" : "signin" })}
               title={user ? user.email : "Sign in"} aria-label="Account">
              {ICON.user}
            </a>
            <a className="sh-icon" href={href({ name: "cart" })} aria-label="Basket">
              {ICON.bag}
              {cart && cart.n_items > 0 ? <span className="pip">{cart.n_items}</span> : null}
            </a>
          </div>
        </div>
      </header>

      <main>{children}</main>

      <footer className="sh-foot">
        <div className="wrap sh-foot-in">
          <div>
            <h4>About these ratings</h4>
            {stats ? (
              <>
                <p>
                  Every product here is rated twice: once as the reviews
                  arrived, and once with the reviews a fake-review detector
                  will not stand behind taken out. Across{" "}
                  {stats.counts.n_listed} listings and{" "}
                  {stats.n_reviews.toLocaleString()} reviews it withholds{" "}
                  {stats.n_withheld.toLocaleString()}.
                </p>
                <p>
                  Reviews are withheld at a score of{" "}
                  {stats.auto_remove_at.toFixed(2)} and above — the point at
                  which the decision policy behind this shop is willing to act
                  without a human moderator. Anything between the review
                  threshold ({stats.threshold.toFixed(2)}) and that line stays
                  visible and is sent to a person instead.
                </p>
              </>
            ) : null}
          </div>

          <div>
            <h4>Shop</h4>
            {(stats?.departments ?? []).slice(0, 5).map((d) => (
              <a key={d.key} href={href({ name: "department", key: d.key })}>{d.name}</a>
            ))}
          </div>

          <div>
            <h4>Account</h4>
            <a href={href({ name: user ? "account" : "signin" })}>
              {user ? "Your account" : "Sign in"}
            </a>
            <a href={href({ name: "orders" })}>Orders</a>
            <a href={href({ name: "cart" })}>Basket</a>
            <a href={href({ name: "console" })}>Moderation console</a>
          </div>
        </div>

        <div className="wrap">
          <p className="fine">
            This is a demonstration. The catalogue, the reviews, the accounts
            and the orders are all synthetic; the brands are invented and the
            photographs were chosen so that no real brand appears, because every
            listing carries a fabricated review history and a fabricated
            manipulation score. Nothing here describes a real product, a real
            company or a real person, no payment is taken and nothing ships.
            {stats ? (
              <>
                {" "}Model scores on reviews written before{" "}
                {new Date(stats.train_period_ends).toLocaleDateString()} come
                from the period the detector was fitted on and are flattered;
                those reviews are marked <em>fitted period</em> wherever they appear.
              </>
            ) : null}
          </p>
        </div>
      </footer>
    </>
  );
}
