import { useEffect, useMemo } from "react";
import type { ModeratorVerdict, NeighbourReview, ReviewDetail } from "../types";
import { DECISION_LABEL, TIER_COLOUR, escapeRe, fmtDateTime } from "../lib";

interface Props {
  detail: ReviewDetail | null;
  loading: boolean;
  error: string | null;
  onClose: () => void;
  verdict?: ModeratorVerdict;
  onVerdict: (v: ModeratorVerdict) => void;
  showTruth: boolean;
}

export function DetailDrawer({
  detail, loading, error, onClose, verdict, onVerdict, showTruth,
}: Props) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <>
      <div className="drawer-scrim" onClick={onClose} />
      <div className="drawer" role="dialog" aria-label="Review detail">
        <header>
          {detail ? (
            <>
              <span className="s" style={{
                fontFamily: "var(--mono)", fontSize: 18, fontWeight: 600,
                color: TIER_COLOUR[detail.decision],
              }}>
                {detail.score.toFixed(3)}
              </span>
              <span className={`tag ${detail.decision}`}>
                {DECISION_LABEL[detail.decision]}
              </span>
              {showTruth && detail.label !== null ? (
                <span className={`tag ${detail.label === 1 ? "truth-fake" : "truth-real"}`}>
                  {detail.label === 1 ? "is fake" : "is real"}
                </span>
              ) : null}
              <span style={{ fontFamily: "var(--mono)", fontSize: 11, color: "var(--fg-faint)" }}>
                {detail.review_id}
              </span>
            </>
          ) : (
            <span style={{ color: "var(--fg-dim)" }}>Review detail</span>
          )}
          <span style={{ flex: 1 }} />
          <button className="ghost" onClick={onClose}>Close</button>
        </header>

        <div className="content">
          {loading ? <div className="spinner">Loading evidence&hellip;</div> : null}
          {error ? <div className="banner err">{error}</div> : null}
          {detail ? <Body detail={detail} showTruth={showTruth} /> : null}
        </div>

        {detail ? (
          <div className="verdict-bar">
            <button
              className={`confirm ${verdict === "confirmed" ? "on" : ""}`}
              onClick={() => onVerdict("confirmed")}
            >
              Confirm fake
            </button>
            <button
              className={`clear ${verdict === "cleared" ? "on" : ""}`}
              onClick={() => onVerdict("cleared")}
            >
              Clear
            </button>
          </div>
        ) : null}
      </div>
    </>
  );
}

function Body({ detail, showTruth }: { detail: ReviewDetail; showTruth: boolean }) {
  const maxContrib = useMemo(
    () => Math.max(...detail.feature_contributions.map((f) => Math.abs(f.contribution)), 1e-9),
    [detail]
  );

  return (
    <>
      <h3>The review</h3>
      <div className="review-body">
        <Highlighted text={detail.text} tokens={detail.token_contributions} />
      </div>
      <div style={{
        display: "flex", gap: 12, marginTop: 8, fontSize: 11.5,
        color: "var(--fg-faint)", fontFamily: "var(--mono)", flexWrap: "wrap",
      }}>
        <span>{"★".repeat(Math.round(detail.rating))} {detail.rating.toFixed(1)}</span>
        <span>{detail.reviewer_id}</span>
        <span>{detail.product_id}</span>
        <span>{fmtDateTime(detail.timestamp)}</span>
        <span>{detail.category}</span>
        <span>{detail.verified_purchase ? "verified" : "unverified"}</span>
        <span>{detail.helpful_votes} helpful</span>
      </div>

      {detail.reasons.length ? (
        <>
          <h3>Why it was flagged</h3>
          <ul className="reason-list">
            {detail.reasons.map((r) => <li key={r}>{r}</li>)}
          </ul>
        </>
      ) : (
        <>
          <h3>Why it was flagged</h3>
          <div className="note">
            No reason codes. Either nothing crossed the evidence guards, or the
            features driving this score have no moderator-facing template &mdash;
            they are still listed below.
          </div>
        </>
      )}

      <h3>
        Feature contributions{" "}
        <span style={{ textTransform: "none", color: "var(--fg-faint)", letterSpacing: 0 }}>
          ({detail.explain_method || "unavailable"})
        </span>
      </h3>
      {detail.feature_contributions.length ? (
        <div className="shap">
          {detail.feature_contributions.slice(0, 14).map((f) => (
            <div className="r" key={f.feature}>
              <span className="f" title={f.feature}>{f.feature}</span>
              <span className="track">
                <span className="mid" />
                <i
                  className={f.direction}
                  style={{ width: `${(Math.abs(f.contribution) / maxContrib) * 50}%` }}
                />
              </span>
              <span className="v">{f.value.toFixed(2)}</span>
            </div>
          ))}
        </div>
      ) : (
        <div className="note">
          This model's booster cannot produce exact per-feature contributions.
          The service says so rather than estimating them.
        </div>
      )}
      <div className="note">
        Red pushes towards fake, green towards genuine; bar length is the size
        of the push. The right-hand number is the feature's value for this
        review, where <code>-1</code> means &ldquo;no history&rdquo;.
      </div>

      <Neighbours
        title="The author's earlier reviews"
        shown={detail.author_prior_reviews}
        total={detail.author_prior_count}
        showTruth={showTruth}
        empty="First review by this account. Every behavioural and graph feature is at its no-history sentinel, so this score is text and metadata alone."
      />
      <Neighbours
        title="Earlier reviews on this product"
        shown={detail.product_prior_reviews}
        total={detail.product_prior_count}
        showTruth={showTruth}
        empty="No prior reviews on this product."
      />
    </>
  );
}

function Neighbours({
  title, shown, total, showTruth, empty,
}: {
  title: string;
  shown: NeighbourReview[];
  total: number;
  showTruth: boolean;
  empty: string;
}) {
  return (
    <>
      <h3>
        {title}{" "}
        <span style={{ textTransform: "none", letterSpacing: 0 }}>
          ({total})
        </span>
      </h3>
      {shown.length ? (
        <div className="neighbours">
          {shown.map((n) => (
            <div className="nb" key={n.review_id}>
              <div className="h">
                <span>{fmtDateTime(n.timestamp)}</span>
                <span>{n.reviewer_id}</span>
                <span>{n.product_id}</span>
                <span>{"★".repeat(Math.round(n.rating))}</span>
                {showTruth && n.label !== null ? (
                  <span className={`tag ${n.label === 1 ? "truth-fake" : "truth-real"}`}>
                    {n.label === 1 ? "fake" : "real"}
                  </span>
                ) : null}
                <span className="sc">{n.score.toFixed(3)}</span>
              </div>
              <div className="x">{n.text}</div>
            </div>
          ))}
          {total > shown.length ? (
            <div className="note">
              Showing the {shown.length} most recent of {total}.
            </div>
          ) : null}
        </div>
      ) : (
        <div className="note">{empty}</div>
      )}
    </>
  );
}

/**
 * Paint the linear model's per-token contributions onto the review text.
 *
 * Stage A is linear, so a token's contribution really is coefficient x TF-IDF
 * weight -- there is nothing approximate about these highlights. Tokens are
 * matched longest-first so that a bigram like "highly recommend" wins over the
 * unigram "recommend" sitting inside it, and each character is claimed at most
 * once so overlapping n-grams cannot double-highlight.
 */
function Highlighted({
  text, tokens,
}: {
  text: string;
  tokens: { token: string; weight: number }[];
}) {
  const spans = useMemo(() => {
    const claimed = new Array<boolean>(text.length).fill(false);
    const marks: { start: number; end: number; weight: number }[] = [];
    const ordered = [...tokens].sort((a, b) => b.token.length - a.token.length);

    for (const t of ordered) {
      if (!t.token.trim()) continue;
      const re = new RegExp(escapeRe(t.token), "gi");
      let m: RegExpExecArray | null;
      while ((m = re.exec(text)) !== null) {
        const start = m.index;
        const end = start + m[0].length;
        let free = true;
        for (let i = start; i < end; i++) if (claimed[i]) { free = false; break; }
        if (!free) continue;
        for (let i = start; i < end; i++) claimed[i] = true;
        marks.push({ start, end, weight: t.weight });
        // Zero-length matches would spin forever; n-grams are never empty, but
        // a defensive bump costs nothing.
        if (m[0].length === 0) re.lastIndex++;
      }
    }

    marks.sort((a, b) => a.start - b.start);
    const out: { text: string; weight: number | null }[] = [];
    let cursor = 0;
    for (const mk of marks) {
      if (mk.start > cursor) out.push({ text: text.slice(cursor, mk.start), weight: null });
      out.push({ text: text.slice(mk.start, mk.end), weight: mk.weight });
      cursor = mk.end;
    }
    if (cursor < text.length) out.push({ text: text.slice(cursor), weight: null });
    return out;
  }, [text, tokens]);

  return (
    <>
      {spans.map((s, i) =>
        s.weight === null ? (
          <span key={i}>{s.text}</span>
        ) : (
          <mark
            key={i}
            className={`tok ${s.weight > 0 ? "fake" : "genuine"}`}
            title={`${s.weight > 0 ? "+" : ""}${s.weight.toFixed(3)} towards ${s.weight > 0 ? "fake" : "genuine"}`}
          >
            {s.text}
          </mark>
        )
      )}
    </>
  );
}
