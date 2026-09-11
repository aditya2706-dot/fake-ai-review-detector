import type { Decision, ModeratorVerdict, Stats } from "../types";
import { DECISION_LABEL, fmtDate, fmtPct } from "../lib";
import { ThresholdPanel } from "./ThresholdPanel";

interface Props {
  stats: Stats;
  threshold: number;
  tunedThreshold: number;
  onThreshold: (t: number) => void;
  onResetThreshold: () => void;
  tierCounts: Record<Decision, number>;
  activeTier: Decision | null;
  onTier: (d: Decision | null) => void;
  verdicts: Record<string, ModeratorVerdict>;
}

const TIERS: Decision[] = ["auto_remove", "queue_for_review", "allow"];

export function Sidebar({
  stats, threshold, tunedThreshold, onThreshold, onResetThreshold,
  tierCounts, activeTier, onTier, verdicts,
}: Props) {
  const reviewed = Object.keys(verdicts).length;
  const confirmed = Object.values(verdicts).filter((v) => v === "confirmed").length;

  return (
    <aside className="sidebar">
      <div className="section">
        <h2>Queue</h2>
        {TIERS.map((t) => (
          <button
            key={t}
            className={`tier-row ${activeTier === t ? "active" : ""}`}
            onClick={() => onTier(activeTier === t ? null : t)}
          >
            <span className={`dot ${t}`} />
            <span className="name">{DECISION_LABEL[t]}</span>
            <span className="count">{tierCounts[t].toLocaleString()}</span>
          </button>
        ))}
        <button
          className={`tier-row ${activeTier === null ? "active" : ""}`}
          onClick={() => onTier(null)}
        >
          <span className="dot" style={{ background: "var(--fg-faint)" }} />
          <span className="name">all</span>
          <span className="count">{stats.n_reviews.toLocaleString()}</span>
        </button>
      </div>

      <ThresholdPanel
        stats={stats}
        threshold={threshold}
        tunedThreshold={tunedThreshold}
        onChange={onThreshold}
        onReset={onResetThreshold}
      />

      <ScoreHistogram stats={stats} threshold={threshold} />

      <div className="section">
        <h2>Your session</h2>
        <dl className="kv"><dt>reviewed</dt><dd>{reviewed}</dd></dl>
        <dl className="kv"><dt>confirmed fake</dt><dd>{confirmed}</dd></dl>
        <dl className="kv"><dt>cleared</dt><dd>{reviewed - confirmed}</dd></dl>
        <div className="note">
          Held in this browser only. A real console would record these against
          the reason codes shown, for appeals.
        </div>
      </div>

      <div className="section">
        <h2>Corpus</h2>
        <dl className="kv"><dt>window</dt><dd>{stats.window}</dd></dl>
        <dl className="kv"><dt>reviews</dt><dd>{stats.n_reviews.toLocaleString()}</dd></dl>
        {stats.has_labels && stats.positive_rate !== null ? (
          <dl className="kv"><dt>actually fake</dt><dd>{fmtPct(stats.positive_rate)}</dd></dl>
        ) : null}
        <dl className="kv"><dt>train ends</dt><dd>{fmtDate(stats.train_period_ends)}</dd></dl>
        <dl className="kv"><dt>test starts</dt><dd>{fmtDate(stats.test_period_starts)}</dd></dl>
      </div>
    </aside>
  );
}

/**
 * Score distribution, with everything at or above the threshold highlighted.
 *
 * Hand-drawn from flexbox rather than a charting library: twenty bars do not
 * justify a dependency, and this way the bars use the same tier colours as the
 * rest of the console instead of a chart library's palette.
 */
function ScoreHistogram({ stats, threshold }: { stats: Stats; threshold: number }) {
  const max = Math.max(...stats.score_histogram.map((b) => b.count), 1);
  return (
    <div className="section">
      <h2>Score distribution</h2>
      <div className="hist">
        {stats.score_histogram.map((b) => (
          <div
            key={b.lo}
            className={`b ${b.hi > threshold ? "over" : ""}`}
            style={{ height: `${Math.max((b.count / max) * 100, 1)}%` }}
            title={`${b.lo.toFixed(2)}-${b.hi.toFixed(2)}: ${b.count.toLocaleString()}`}
          />
        ))}
      </div>
      <div className="hist-axis"><span>0.0</span><span>0.5</span><span>1.0</span></div>
      <div className="note">
        Calibrated probabilities, so a bar at 0.8 means roughly 80% of those
        reviews really are fake. Most of the mass sits near zero &mdash; the
        queue is the thin tail on the right.
      </div>
    </div>
  );
}
