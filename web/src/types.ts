// Mirrors the payloads in src/frd/api/queue.py. Kept as one file so a server
// change this client has not caught up with shows up as a type error in one
// place rather than as `undefined` somewhere inside a component.

export type Decision = "allow" | "queue_for_review" | "auto_remove";

export interface QueueItem {
  review_id: string;
  product_id: string;
  reviewer_id: string;
  rating: number;
  title: string;
  text: string;
  timestamp: string;
  category: string;
  verified_purchase: boolean;
  helpful_votes: number;
  score: number;
  decision: Decision;
  /** Ground truth. null on any corpus that has none -- i.e. every real one. */
  label: number | null;
  reasons?: string[];
}

export interface QueuePage {
  items: QueueItem[];
  total: number;
  offset: number;
  limit: number;
  threshold: number;
  auto_remove_at: number;
  window: string;
}

export interface FeatureContribution {
  feature: string;
  contribution: number;
  value: number;
  direction: "fake" | "genuine";
}

export interface TokenContribution {
  token: string;
  weight: number;
  direction: string;
}

export interface NeighbourReview {
  review_id: string;
  product_id: string;
  reviewer_id: string;
  rating: number;
  text: string;
  timestamp: string;
  score: number;
  label: number | null;
}

export interface ReviewDetail extends QueueItem {
  reasons: string[];
  feature_contributions: FeatureContribution[];
  token_contributions: TokenContribution[];
  explain_method: string;
  threshold: number;
  auto_remove_at: number;
  author_prior_reviews: NeighbourReview[];
  author_prior_count: number;
  product_prior_reviews: NeighbourReview[];
  product_prior_count: number;
}

export interface ThresholdRow {
  threshold: number;
  flagged_rate: number;
  precision: number;
  recall: number;
  flagged_per_day: number;
  false_accusations_per_day: number;
  missed_fakes_per_day: number;
}

export interface Stats {
  window: string;
  n_reviews: number;
  corpus_path: string;
  build_seconds: number;
  train_period_ends: string;
  test_period_starts: string;
  threshold: number;
  auto_remove_at: number;
  decision_policy: { auto_remove_margin: number; auto_remove_cap: number };
  tiers: Record<Decision, number>;
  has_labels: boolean;
  positive_rate: number | null;
  score_histogram: { lo: number; hi: number; count: number }[];
  threshold_table: ThresholdRow[];
}

export interface Health {
  status: string;
  model_loaded: boolean;
  model_version: string;
  blocks: string[];
  backend: string;
  threshold: number | null;
  calibration: string;
}

/** A moderator's own call, held locally. See useDecisions. */
export type ModeratorVerdict = "confirmed" | "cleared";
