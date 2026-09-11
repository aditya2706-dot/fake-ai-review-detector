/**
 * A hash router in thirty lines.
 *
 * The app has three routes and one dependency policy: react and react-dom.
 * Adding react-router for `#/shop`, `#/shop/:id` and `#/console` would be more
 * code in `package-lock.json` than in this file. Hash routing specifically --
 * not the History API -- because the built bundle is served as static files
 * from whatever path the API or a CDN puts it under, and a hash needs no
 * server-side rewrite rule to survive a refresh.
 */

import { useEffect, useState } from "react";

export type Route =
  | { name: "shop" }
  | { name: "product"; id: string }
  | { name: "console" };

export function parse(hash: string): Route {
  const path = hash.replace(/^#\/?/, "").split("?")[0];
  const [head, tail] = path.split("/");
  if (head === "console") return { name: "console" };
  if (head === "shop" && tail) return { name: "product", id: decodeURIComponent(tail) };
  return { name: "shop" };
}

export function href(route: Route): string {
  if (route.name === "console") return "#/console";
  if (route.name === "product") return `#/shop/${encodeURIComponent(route.id)}`;
  return "#/shop";
}

export function navigate(route: Route): void {
  window.location.hash = href(route);
}

export function useRoute(): Route {
  const [route, setRoute] = useState<Route>(() => parse(window.location.hash));
  useEffect(() => {
    const onHash = () => {
      setRoute(parse(window.location.hash));
      // A route change is a new page; inheriting the previous page's scroll
      // offset lands a product page halfway down its own review list.
      window.scrollTo(0, 0);
    };
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);
  return route;
}
