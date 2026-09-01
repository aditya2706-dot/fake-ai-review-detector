import type { Decision, ModeratorVerdict, QueueItem, Stats } from "../types";
import { DECISION_LABEL, TIER_COLOUR, decide, fmtDate } from "../lib";

interface Props {
  items: QueueItem[];
  threshold: number;
  policy: Stats["decision_policy"];
  selectedId: string | null;
  onSelect: (id: string) => void;
  verdicts: Record<string, ModeratorVerdict>;
  showTruth: boolean;
}

export function QueueTable({
  items, threshold, policy, selectedId, onSelect, verdicts, showTruth,
}: Props) {
  if (!items.length) {
    return <div className="empty">Nothing matches these filters.</div>;
  }
  return (
    <div className="queue">
      {items.map((it) => (
        <Row
          key={it.review_id}
          item={it}
          // Re-tiered client-side so the colours track the slider without a
          // round trip. The server's own `decision` is the reference; see
          // policyDisagreements in lib.ts.
          tier={decide(it.score, threshold, policy)}
          selected={it.review_id === selectedId}
          onSelect={onSelect}
          verdict={verdicts[it.review_id]}
          showTruth={showTruth}
        />
      ))}
    </div>
  );
}

function Row({
  item, tier, selected, onSelect, verdict, showTruth,
}: {
  item: QueueItem;
  tier: Decision;
  selected: boolean;
  onSelect: (id: string) => void;
  verdict?: ModeratorVerdict;
  showTruth: boolean;
}) {
  return (
    <div
      className={`row ${selected ? "selected" : ""}`}
      onClick={() => onSelect(item.review_id)}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          onSelect(item.review_id);
        }
      }}
    >
      <div className="score-cell">
        <div className="s" style={{ color: TIER_COLOUR[tier] }}>
          {item.score.toFixed(3)}
        </div>
        <div className="bar">
          <i style={{ width: `${item.score * 100}%`, background: TIER_COLOUR[tier] }} />
        </div>
      </div>

      <div className="text">
        <div className="t">
          {item.title || <span style={{ color: "var(--fg-faint)" }}>(no title)</span>}
        </div>
        <div className="b">{item.text}</div>
        {item.reasons?.length ? (
          <div className="reasons">
            {item.reasons.slice(0, 3).map((r) => (
              <span key={r} className="chip tone-fake">{r}</span>
            ))}
          </div>
        ) : null}
      </div>

      <div className="meta">
        <div className="line" style={{ marginBottom: 4 }}>
          <span className={`tag ${tier}`}>{DECISION_LABEL[tier]}</span>{" "}
          {showTruth && item.label !== null ? (
            <span className={`tag ${item.label === 1 ? "truth-fake" : "truth-real"}`}>
              {item.label === 1 ? "is fake" : "is real"}
            </span>
          ) : null}
          {verdict ? <span className="tag verdict">{verdict}</span> : null}
        </div>
        <div className="line">{"★".repeat(Math.round(item.rating))}</div>
        <div className="line">{item.reviewer_id} &middot; {item.product_id}</div>
        <div className="line">{fmtDate(item.timestamp)} &middot; {item.category}</div>
        <div className="line">
          {item.verified_purchase ? "verified" : "unverified"}
        </div>
      </div>
    </div>
  );
}
