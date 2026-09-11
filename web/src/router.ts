/**
 * A hash router in a hundred lines.
 *
 * The dependency policy for this app is react and react-dom, and nothing else.
 * react-router for nine routes would be more code in `package-lock.json` than
 * in this file.
 *
 * Hash routing specifically -- not the History API -- because the built bundle
 * is served as static files from whatever path the API or a CDN puts it under,
 * and a hash needs no server-side rewrite rule to survive a refresh. The query
 * string lives inside the hash for the same reason: `#/search?q=kettle` is one
 * URL the server never has to know about.
 */

import { useEffect, useState } from "react";

export type Route =
  | { name: "home" }
  | { name: "department"; key: string }
  | { name: "search"; q: string }
  | { name: "product"; slug: string }
  | { name: "cart" }
  | { name: "checkout" }
  | { name: "order"; id: string }
  | { name: "orders" }
  | { name: "account" }
  | { name: "signin"; next?: string }
  | { name: "console" };

export function parse(hash: string): Route {
  const raw = hash.replace(/^#\/?/, "");
  const [path, query] = raw.split("?");
  const params = new URLSearchParams(query ?? "");
  const [head, tail] = path.split("/");
  const seg = tail ? decodeURIComponent(tail) : "";

  switch (head) {
    case "console": return { name: "console" };
    case "c": return seg ? { name: "department", key: seg } : { name: "home" };
    case "p": return seg ? { name: "product", slug: seg } : { name: "home" };
    case "search": return { name: "search", q: params.get("q") ?? "" };
    case "cart": return { name: "cart" };
    case "checkout": return { name: "checkout" };
    case "order": return seg ? { name: "order", id: seg } : { name: "orders" };
    case "orders": return { name: "orders" };
    case "account": return { name: "account" };
    case "signin": return { name: "signin", next: params.get("next") ?? undefined };
    default: return { name: "home" };
  }
}

export function href(route: Route): string {
  const enc = encodeURIComponent;
  switch (route.name) {
    case "department": return `#/c/${enc(route.key)}`;
    case "search": return `#/search?q=${enc(route.q)}`;
    case "product": return `#/p/${enc(route.slug)}`;
    case "cart": return "#/cart";
    case "checkout": return "#/checkout";
    case "order": return `#/order/${enc(route.id)}`;
    case "orders": return "#/orders";
    case "account": return "#/account";
    case "signin": return route.next ? `#/signin?next=${enc(route.next)}` : "#/signin";
    case "console": return "#/console";
    default: return "#/";
  }
}

export function navigate(route: Route): void {
  window.location.hash = href(route);
}

/** True for the routes that make up the shop, as opposed to the console. */
export function isShop(route: Route): boolean {
  return route.name !== "console";
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
