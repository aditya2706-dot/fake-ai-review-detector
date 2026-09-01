import type { Health, QueuePage, ReviewDetail, Stats } from "./types";

// Vite proxies /api -> the FastAPI service (see vite.config.ts). A deployed
// bundle can point elsewhere with VITE_API_BASE.
const BASE = import.meta.env.VITE_API_BASE ?? "/api";

export class ApiError extends Error {
  constructor(readonly status: number, message: string) {
    super(message);
  }
}

async function get<T>(path: string, params?: Record<string, unknown>): Promise<T> {
  const url = new URL(`${BASE}${path}`, window.location.origin);
  for (const [k, v] of Object.entries(params ?? {})) {
    if (v !== undefined && v !== null && v !== "") url.searchParams.set(k, String(v));
  }
  const res = await fetch(url.toString().replace(window.location.origin, ""));
  if (!res.ok) {
    // FastAPI puts the useful part in `detail`; surface it rather than a bare
    // status code, because the two failures a user will actually hit here --
    // no trained model, no corpus -- both explain how to fix themselves.
    let detail = res.statusText;
    try {
      const body = await res.json();
      if (typeof body?.detail === "string") detail = body.detail;
    } catch {
      /* non-JSON error body; keep statusText */
    }
    throw new ApiError(res.status, detail);
  }
  return res.json() as Promise<T>;
}

export const api = {
  health: () => get<Health>("/health"),
  stats: (window: string) => get<Stats>("/queue/stats", { window }),
  categories: () => get<string[]>("/queue/categories"),
  detail: (id: string, threshold?: number) =>
    get<ReviewDetail>(`/queue/${encodeURIComponent(id)}`, { threshold }),
  queue: (p: {
    window: string;
    limit: number;
    offset: number;
    threshold?: number;
    decision?: string;
    category?: string;
    search?: string;
    sort: string;
    explain?: boolean;
  }) => get<QueuePage>("/queue", p),
};
