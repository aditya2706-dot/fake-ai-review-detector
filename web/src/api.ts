import type {
  Cart, Department, Health, Order, OrderHistory, ProductDetail, ProductPage,
  QueuePage, ReviewDetail, ShopStats, Stats, User,
} from "./types";

// Vite proxies /api -> the FastAPI service (see vite.config.ts). A deployed
// bundle can point elsewhere with VITE_API_BASE.
const BASE = import.meta.env.VITE_API_BASE ?? "/api";

export class ApiError extends Error {
  constructor(readonly status: number, message: string) {
    super(message);
  }
}

function url(path: string, params?: Record<string, unknown>): string {
  const u = new URL(`${BASE}${path}`, window.location.origin);
  for (const [k, v] of Object.entries(params ?? {})) {
    if (v !== undefined && v !== null && v !== "") u.searchParams.set(k, String(v));
  }
  return u.toString().replace(window.location.origin, "");
}

async function send<T>(
  method: string, path: string,
  opts: { params?: Record<string, unknown>; body?: unknown } = {},
): Promise<T> {
  const res = await fetch(url(path, opts.params), {
    method,
    // The session lives in an httpOnly cookie, so every call has to carry
    // credentials or the server sees an anonymous request. This is also why
    // the API sets an explicit CORS origin rather than "*" -- a browser
    // refuses to send credentials to a wildcard origin.
    credentials: "include",
    headers: opts.body === undefined ? undefined : { "Content-Type": "application/json" },
    body: opts.body === undefined ? undefined : JSON.stringify(opts.body),
  });
  if (!res.ok) {
    // FastAPI puts the useful part in `detail`; surface it rather than a bare
    // status code, because the failures a user will actually hit here -- no
    // trained model, wrong password, empty basket -- all explain themselves.
    let detail = res.statusText;
    try {
      const body = await res.json();
      if (typeof body?.detail === "string") detail = body.detail;
      else if (Array.isArray(body?.detail) && body.detail[0]?.msg) {
        detail = String(body.detail[0].msg).replace(/^Value error,\s*/, "");
      }
    } catch {
      /* non-JSON error body; keep statusText */
    }
    throw new ApiError(res.status, detail);
  }
  if (res.status === 204) return undefined as T;
  return res.json() as Promise<T>;
}

const get = <T>(path: string, params?: Record<string, unknown>) =>
  send<T>("GET", path, { params });

export const api = {
  health: () => get<Health>("/health"),

  // --- moderation console ---
  stats: (window: string) => get<Stats>("/queue/stats", { window }),
  categories: () => get<string[]>("/queue/categories"),
  detail: (id: string, threshold?: number) =>
    get<ReviewDetail>(`/queue/${encodeURIComponent(id)}`, { threshold }),
  queue: (p: {
    window: string; limit: number; offset: number; threshold?: number;
    decision?: string; category?: string; search?: string; sort: string;
    explain?: boolean;
  }) => get<QueuePage>("/queue", p),

  // --- storefront ---
  shopStats: (threshold?: number) => get<ShopStats>("/shop/stats", { threshold }),
  departments: () =>
    get<{ departments: Department[]; currency: string }>("/shop/departments"),
  products: (p: {
    limit: number; offset: number; category?: string; search?: string;
    tag?: string; min_reviews?: number; sort: string; threshold?: number;
  }) => get<ProductPage>("/shop/products", p),
  product: (slug: string, p?: {
    limit?: number; offset?: number; sort?: string; threshold?: number;
  }) => get<ProductDetail>(`/shop/products/${encodeURIComponent(slug)}`, p),

  // --- account ---
  me: () => get<{ user: User | null; provider: string }>("/auth/me"),
  register: (email: string, password: string, name: string) =>
    send<{ user: User }>("POST", "/auth/register", { body: { email, password, name } }),
  login: (email: string, password: string) =>
    send<{ user: User }>("POST", "/auth/login", { body: { email, password, name: "" } }),
  logout: () => send<{ ok: boolean }>("POST", "/auth/logout"),

  // --- basket ---
  cart: () => get<Cart>("/cart"),
  cartAdd: (slug: string, variant_sku: string, qty: number) =>
    send<Cart>("POST", "/cart/items", { body: { slug, variant_sku, qty } }),
  cartSetQty: (variant_sku: string, qty: number) =>
    send<Cart>("PATCH", "/cart/items", { body: { variant_sku, qty } }),
  cartRemove: (variant_sku: string) =>
    send<Cart>("DELETE", `/cart/items/${encodeURIComponent(variant_sku)}`),

  // --- orders ---
  checkout: (email: string, address: Record<string, string>) =>
    send<Order>("POST", "/orders", { body: { email, address } }),
  orders: () => get<OrderHistory>("/orders"),
  order: (id: string) => get<Order>(`/orders/${encodeURIComponent(id)}`),
};

export function describe(e: unknown): string {
  if (e instanceof ApiError) return e.message;
  if (e instanceof Error) return e.message;
  return String(e);
}
