import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ApiError, api } from "./api";
import { DetailDrawer } from "./components/DetailDrawer";
import { QueueTable } from "./components/QueueTable";
import { Sidebar } from "./components/Sidebar";
import {
  decide, loadVerdicts, policyDisagreements, saveVerdicts,
} from "./lib";
import type {
  Decision, Health, ModeratorVerdict, QueuePage, ReviewDetail, Stats,
} from "./types";

const PAGE = 50;

export default function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [stats, setStats] = useState<Stats | null>(null);
  const [page, setPage] = useState<QueuePage | null>(null);
  const [categories, setCategories] = useState<string[]>([]);
  const [fatal, setFatal] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const [window_, setWindow] = useState<"test" | "all">("test");
  const [threshold, setThreshold] = useState<number | null>(null);
  const [tier, setTier] = useState<Decision | null>(null);
  const [category, setCategory] = useState("");
  const [search, setSearch] = useState("");
  const [sort, setSort] = useState("score_desc");
  const [offset, setOffset] = useState(0);
  const [showTruth, setShowTruth] = useState(true);

  const [selected, setSelected] = useState<string | null>(null);
  const [detail, setDetail] = useState<ReviewDetail | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [detailError, setDetailError] = useState<string | null>(null);

  const [verdicts, setVerdicts] = useState<Record<string, ModeratorVerdict>>(loadVerdicts);
  const [skew, setSkew] = useState<string[]>([]);

  // ---- initial load -------------------------------------------------------
  useEffect(() => {
    let live = true;
    (async () => {
      try {
        const h = await api.health();
        if (!live) return;
        setHealth(h);
        if (!h.model_loaded) {
          setFatal("No model is loaded. Run `make train` and restart the API.");
          setLoading(false);
          return;
        }
        const [s, c] = await Promise.all([api.stats("test"), api.categories()]);
        if (!live) return;
        setStats(s);
        setCategories(c);
        setThreshold(s.threshold);
      } catch (e) {
        if (live) setFatal(describe(e));
      } finally {
        if (live) setLoading(false);
      }
    })();
    return () => { live = false; };
  }, []);

  // Stats are per-window: switching to `all` changes the row count, the tier
  // counts and the threshold table, so they must be refetched, not reused.
  useEffect(() => {
    if (!health?.model_loaded) return;
    let live = true;
    api.stats(window_).then((s) => { if (live) setStats(s); }).catch((e) => setFatal(describe(e)));
    return () => { live = false; };
  }, [window_, health]);

  // ---- the queue page -----------------------------------------------------
  const debounced = useDebounced(search, 250);
  const reqId = useRef(0);

  useEffect(() => {
    if (threshold === null || !health?.model_loaded) return;
    const mine = ++reqId.current;
    setLoading(true);
    api
      .queue({
        window: window_, limit: PAGE, offset, threshold,
        decision: tier ?? undefined, category: category || undefined,
        search: debounced || undefined, sort, explain: true,
      })
      .then((p) => {
        // A slow earlier request must not overwrite a newer result. Without
        // this, typing in the search box makes the list flicker back to a
        // previous query's rows.
        if (mine !== reqId.current) return;
        setPage(p);
        setFatal(null);
      })
      .catch((e) => { if (mine === reqId.current) setFatal(describe(e)); })
      .finally(() => { if (mine === reqId.current) setLoading(false); });
  }, [window_, offset, threshold, tier, category, debounced, sort, health]);

  // Reset paging whenever the result set changes underneath it.
  useEffect(() => { setOffset(0); }, [window_, tier, category, debounced, sort, threshold]);

  // ---- client/server tiering agreement ------------------------------------
  useEffect(() => {
    if (!page || !stats || threshold === null) return;
    // Only meaningful at the server's own threshold; past that the client is
    // deliberately answering a different question than the row was built with.
    if (Math.abs(threshold - page.threshold) > 1e-9) { setSkew([]); return; }
    setSkew(policyDisagreements(page.items, threshold, stats.decision_policy));
  }, [page, stats, threshold]);

  // ---- detail -------------------------------------------------------------
  useEffect(() => {
    if (!selected || threshold === null) { setDetail(null); return; }
    let live = true;
    setDetailLoading(true);
    setDetailError(null);
    api
      .detail(selected, threshold)
      .then((d) => { if (live) setDetail(d); })
      .catch((e) => { if (live) setDetailError(describe(e)); })
      .finally(() => { if (live) setDetailLoading(false); });
    return () => { live = false; };
  }, [selected, threshold]);

  const setVerdict = useCallback((id: string, v: ModeratorVerdict) => {
    setVerdicts((prev) => {
      // Clicking the active verdict again clears it, so a misclick is undoable.
      const next = { ...prev };
      if (next[id] === v) delete next[id]; else next[id] = v;
      saveVerdicts(next);
      return next;
    });
  }, []);

  // Tier counts follow the slider, so they are recomputed from the histogram
  // rather than read from the server's fixed-threshold `tiers` block.
  const tierCounts = useMemo<Record<Decision, number>>(() => {
    if (!stats || threshold === null) {
      return { auto_remove: 0, queue_for_review: 0, allow: 0 };
    }
    if (Math.abs(threshold - stats.threshold) < 1e-9) return stats.tiers;
    const out: Record<Decision, number> = { auto_remove: 0, queue_for_review: 0, allow: 0 };
    for (const b of stats.score_histogram) {
      // Bucket midpoint: an approximation, and labelled as one in the UI.
      out[decide((b.lo + b.hi) / 2, threshold, stats.decision_policy)] += b.count;
    }
    return out;
  }, [stats, threshold]);

  if (fatal && !stats) {
    return (
      <div className="app">
        <div style={{ padding: 40, maxWidth: 640 }}>
          <h1 style={{ fontSize: 16 }}>Review moderation queue</h1>
          <div className="banner err" style={{ marginTop: 16 }}>{fatal}</div>
          <div className="note">
            The console needs the scoring API on :8000 with a trained model and a
            corpus. From the repository root: <code>make train</code> then{" "}
            <code>make serve</code>.
          </div>
        </div>
      </div>
    );
  }

  if (!stats || threshold === null) {
    return <div className="spinner" style={{ paddingTop: 120 }}>Scoring the corpus&hellip;
      <div className="note" style={{ maxWidth: 420, margin: "12px auto 0", textAlign: "left" }}>
        The first request features and scores the whole corpus (about 20 seconds
        for 40,000 reviews). Everything after it is served from that.
      </div>
    </div>;
  }

  const approxCounts = Math.abs(threshold - stats.threshold) >= 1e-9;

  return (
    <div className="app">
      <div className="topbar">
        <h1>Review moderation queue</h1>
        <span className="tag">{health?.backend}</span>
        <span className="tag">{health?.blocks.length} blocks</span>
        <span className="spacer" />
        {stats.has_labels ? (
          <label style={{ fontSize: 12.5, color: "var(--fg-dim)", display: "flex", gap: 6, alignItems: "center" }}>
            <input type="checkbox" checked={showTruth} onChange={(e) => setShowTruth(e.target.checked)} />
            show ground truth
          </label>
        ) : null}
        <div className="seg">
          <button className={window_ === "test" ? "on" : ""} onClick={() => setWindow("test")}>
            held-out
          </button>
          <button className={window_ === "all" ? "on" : ""} onClick={() => setWindow("all")}>
            all reviews
          </button>
        </div>
      </div>

      <div className="body">
        <Sidebar
          stats={stats}
          threshold={threshold}
          tunedThreshold={stats.threshold}
          onThreshold={setThreshold}
          onResetThreshold={() => setThreshold(stats.threshold)}
          tierCounts={tierCounts}
          activeTier={tier}
          onTier={setTier}
          verdicts={verdicts}
        />

        <div className="main">
          <div className="toolbar">
            <input
              className="field search"
              placeholder="Search text, reviewer, product, review id"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
            />
            <select className="field" value={category} onChange={(e) => setCategory(e.target.value)}>
              <option value="">all categories</option>
              {categories.map((c) => <option key={c} value={c}>{c}</option>)}
            </select>
            <select className="field" value={sort} onChange={(e) => setSort(e.target.value)}>
              <option value="score_desc">highest score first</option>
              <option value="score_asc">lowest score first</option>
              <option value="newest">newest first</option>
              <option value="oldest">oldest first</option>
            </select>
            <span className="spacer" style={{ flex: 1 }} />
            <span style={{ fontSize: 12.5, color: "var(--fg-dim)", fontFamily: "var(--mono)" }}>
              {page ? `${offset + 1}-${Math.min(offset + PAGE, page.total)} of ${page.total.toLocaleString()}` : "—"}
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

          <div style={{ padding: "0 16px" }}>
            {window_ === "all" ? (
              <div className="banner" style={{ marginTop: 12 }}>
                Showing all reviews, including the period the model was trained
                on. Scores there are partly memorised and look better than the
                model deserves. The held-out view is the honest one.
              </div>
            ) : null}
            {approxCounts ? (
              <div className="banner" style={{ marginTop: 12 }}>
                Threshold moved off the tuned {stats.threshold.toFixed(3)}. Sidebar
                tier counts are approximated from the score histogram; the queue
                itself is filtered exactly.
              </div>
            ) : null}
            {skew.length ? (
              <div className="banner err" style={{ marginTop: 12 }}>
                {skew.length} row(s) tiered differently by this browser than by
                the server &mdash; the decision policy has drifted between them.
                Trust the server. First: {skew[0]}.
              </div>
            ) : null}
            {fatal ? <div className="banner err" style={{ marginTop: 12 }}>{fatal}</div> : null}
          </div>

          {loading && !page ? (
            <div className="spinner">Loading&hellip;</div>
          ) : (
            <QueueTable
              items={page?.items ?? []}
              threshold={threshold}
              policy={stats.decision_policy}
              selectedId={selected}
              onSelect={setSelected}
              verdicts={verdicts}
              showTruth={showTruth && stats.has_labels}
            />
          )}
        </div>
      </div>

      {selected ? (
        <DetailDrawer
          detail={detail}
          loading={detailLoading}
          error={detailError}
          onClose={() => setSelected(null)}
          verdict={verdicts[selected]}
          onVerdict={(v) => setVerdict(selected, v)}
          showTruth={showTruth && stats.has_labels}
        />
      ) : null}
    </div>
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
