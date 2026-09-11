import type { Stats } from "../types";
import { autoRemoveAt, fmtPct, nearestRow } from "../lib";

interface Props {
  stats: Stats;
  threshold: number;
  onChange: (t: number) => void;
  onReset: () => void;
  tunedThreshold: number;
}

/**
 * The operating-point control.
 *
 * A threshold is a policy decision, not a model parameter, so it gets a
 * first-class control and states its cost in the units an ops lead thinks in:
 * how many reviews land in front of a human, and how many of those are
 * innocent people. `false_accusations_per_day` is the number this project
 * treats as the expensive one, so it is the one shown in red.
 */
export function ThresholdPanel({ stats, threshold, onChange, onReset, tunedThreshold }: Props) {
  const row = nearestRow(stats.threshold_table, threshold);
  const removeAt = autoRemoveAt(threshold, stats.decision_policy);
  const moved = Math.abs(threshold - tunedThreshold) > 1e-9;

  return (
    <div className="section">
      <h2>Operating point</h2>
      <div className="slider-head">
        <span className="val">
          {threshold.toFixed(3)}{" "}
          <small>queue &ge; {threshold.toFixed(2)} &middot; remove &ge; {removeAt.toFixed(2)}</small>
        </span>
      </div>
      <input
        type="range"
        min={0.01}
        max={0.99}
        step={0.005}
        value={threshold}
        onChange={(e) => onChange(Number(e.target.value))}
        aria-label="Decision threshold"
      />

      {row ? (
        <div className="tradeoff">
          <div className="cell">
            <div className="n">{fmtPct(row.precision, 1)}</div>
            <div className="l">precision</div>
          </div>
          <div className="cell">
            <div className="n">{fmtPct(row.recall, 1)}</div>
            <div className="l">recall</div>
          </div>
          <div className="cell">
            <div className="n">{row.flagged_per_day.toLocaleString()}</div>
            <div className="l">sent to a human</div>
          </div>
          <div className="cell">
            <div className="n" style={{ color: "var(--remove)" }}>
              {row.false_accusations_per_day.toLocaleString()}
            </div>
            <div className="l">wrongly flagged</div>
          </div>
        </div>
      ) : (
        <div className="note">
          No labels in this corpus, so the precision/recall cost of a threshold
          cannot be shown &mdash; only how many reviews it flags.
        </div>
      )}

      {row ? (
        <div className="note">
          Measured on the {stats.n_reviews.toLocaleString()} reviews currently in
          view, at the nearest tabulated threshold ({row.threshold.toFixed(2)}).
          Counts are per {stats.n_reviews.toLocaleString()} reviews scored.
        </div>
      ) : null}

      {moved ? (
        <button className="ghost" style={{ marginTop: 10, width: "100%" }} onClick={onReset}>
          Reset to tuned threshold ({tunedThreshold.toFixed(3)})
        </button>
      ) : (
        <div className="note">
          <strong>This is the tuned threshold.</strong> Chosen on the validation
          split to hit 90% precision &mdash; never on the reviews shown here.
        </div>
      )}
    </div>
  );
}
