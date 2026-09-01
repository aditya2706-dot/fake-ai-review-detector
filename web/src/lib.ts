import type { Decision, ModeratorVerdict, Stats, ThresholdRow } from "./types";

/**
 * Re-tier a score client-side as the threshold slider moves.
 *
 * The policy constants come from the server (`/queue/stats` ->
 * `decision_policy`) rather than being hardcoded here. The Python side makes a
 * lot of noise about not keeping a second implementation of anything on the
 * serving path; a TypeScript copy of the decision thresholds would be exactly
 * that mistake in a new language. What is duplicated here is only the shape of
 * the comparison, and it is checked against the server's own tiering in
 * `assertPolicyAgrees` below.
 */
export function decide(score: number, threshold: number, policy: Stats["decision_policy"]): Decision {
  const removeAt = Math.min(policy.auto_remove_cap, threshold + policy.auto_remove_margin);
  if (score >= removeAt) return "auto_remove";
  if (score >= threshold) return "queue_for_review";
  return "allow";
}

export function autoRemoveAt(threshold: number, policy: Stats["decision_policy"]): number {
  return Math.min(policy.auto_remove_cap, threshold + policy.auto_remove_margin);
}

/** Nearest row of the server's threshold table, for the slider's readout. */
export function nearestRow(table: ThresholdRow[], threshold: number): ThresholdRow | null {
  if (!table.length) return null;
  return table.reduce((best, r) =>
    Math.abs(r.threshold - threshold) < Math.abs(best.threshold - threshold) ? r : best
  );
}

export const DECISION_LABEL: Record<Decision, string> = {
  auto_remove: "auto-remove",
  queue_for_review: "queue",
  allow: "allow",
};

export const TIER_COLOUR: Record<Decision, string> = {
  auto_remove: "var(--remove)",
  queue_for_review: "var(--queue)",
  allow: "var(--allow)",
};

export function fmtPct(x: number, digits = 1): string {
  return `${(x * 100).toFixed(digits)}%`;
}

export function fmtDate(iso: string): string {
  const d = new Date(iso);
  return d.toISOString().slice(0, 10);
}

export function fmtDateTime(iso: string): string {
  return new Date(iso).toISOString().slice(0, 16).replace("T", " ");
}

/**
 * Escape a string for use inside a RegExp.
 *
 * Token contributions come from TF-IDF n-grams, which routinely contain `.`,
 * `(`, `+` and friends. Interpolating one straight into a RegExp either throws
 * or, worse, silently matches the wrong span of the review text.
 */
export function escapeRe(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

// ---- moderator verdicts, held in the browser -------------------------------
//
// Deliberately localStorage and not a server table. A real moderation system
// records decisions durably, attaches the reason codes, and exposes them to an
// appeals process -- see docs/ROADMAP.md phase 9. Pretending to do that with a
// JSON blob in a browser would be worse than obviously not doing it, so this
// is scoped as what it is: a scratchpad for one person's session.

const KEY = "frd.verdicts.v1";

export function loadVerdicts(): Record<string, ModeratorVerdict> {
  try {
    const raw = localStorage.getItem(KEY);
    return raw ? (JSON.parse(raw) as Record<string, ModeratorVerdict>) : {};
  } catch {
    return {};
  }
}

export function saveVerdicts(v: Record<string, ModeratorVerdict>): void {
  try {
    localStorage.setItem(KEY, JSON.stringify(v));
  } catch {
    /* private browsing, quota, disabled storage: the queue still works */
  }
}

// ---- client/server tiering agreement ---------------------------------------

/**
 * Check the client's re-tiering against the server's own answer.
 *
 * Every row the server sends carries the `decision` it computed. When the
 * slider is untouched the client must reproduce those exactly; if it does not,
 * the policy has drifted between Python and TypeScript and every colour in the
 * queue is a lie. Cheap to check on each page, so it is checked on each page
 * rather than assumed.
 *
 * Returns the ids that disagree, empty when all is well.
 */
export function policyDisagreements(
  items: { review_id: string; score: number; decision: Decision }[],
  threshold: number,
  policy: Stats["decision_policy"]
): string[] {
  return items
    .filter((i) => decide(i.score, threshold, policy) !== i.decision)
    .map((i) => i.review_id);
}
