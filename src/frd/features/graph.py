"""Causal graph feature engineering over the reviewer-product bipartite graph.

WHY A GRAPH AT ALL
------------------
Every feature in `behavior.py` reasons about one account or one product in
isolation, and that is exactly the boundary a competent farm operates across.
Spread forty paid reviews over forty accounts and nobody bursts; give each
account a plausible posting rate and every per-account threshold stays green.
What the farm cannot hide is that the same accounts keep turning up on the
same products. Two strangers overlapping on one product is a coincidence and
happens constantly on a marketplace; overlapping on five is a working
relationship.

So the unit of analysis here is the *pair*, and the structure it induces:

    reviewers --(reviewed)-- products

Project that bipartite graph onto its reviewer side and a farm stops looking
like forty innocuous accounts and starts looking like what it is -- a dense
subgraph that keeps re-forming on whichever product is paying this week.

WHY THIS MODULE IS STREAMING STATISTICS AND NOT NODE2VEC
-------------------------------------------------------
`docs/ROADMAP.md` phase 7 floated node2vec or GNN embeddings. Taken literally
that is a leak, and a subtle enough one to be worth spelling out.

An embedding is fitted to a graph. Fit it to *the* graph -- the one built from
the whole corpus -- and every node vector is a summary of all the edges that
account will ever have, including the ones from next March. Attach that vector
to a review from January and the row now carries its own future. It is the
whole-dataset reviewer average from `behavior.py`'s docstring wearing a
different hat, except that here the leak arrives pre-laundered through 128
dimensions of matrix factorisation where no reviewer will ever spot it. The
model would look superb and would be worthless.

Computing it honestly means re-embedding the graph as it stood at row i, for
every i, which nothing can afford. So this module takes the other road: graph
statistics that can be *maintained incrementally* in a single chronological
pass, exactly as `behavior.py` maintains its counters. Every quantity below is
derived from edges committed strictly before the row being scored.

    A feature for review i may only use reviews strictly earlier than i.

`tests/test_graph.py` proves it the same way `tests/test_causality.py` does:
recompute on a truncated corpus, demand bit-identical rows.

Embeddings are still reachable without breaking the rule -- fit them on
periodic snapshots, each trained only on data before its cutoff, and serve
each review the most recent snapshot that predates it. That is a feature-store
project, not a function, which is why it is not here.

COST
----
Pair state is quadratic in the wrong hands: a product with 10,000 reviews
implies 50M pairs. Every neighbourhood here is therefore bounded by both a
time window and a hard cap on cohort size, which costs a little recall on
farms that work very slowly and buys a linear-time pass. The production answer
is a real graph store with the co-review projection maintained as a materialised
view; the bounded windows are what make the same features computable inside a
laptop pass over a CSV.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..schema import PRODUCT_ID, REVIEWER_ID, TIMESTAMP

ONE_DAY = 86_400.0

# How far back a product's reviewer cohort is remembered. A bought burst runs
# for hours to days, so 60 days is generous; the cap matters because every
# remembered reviewer is a pair to maintain.
COHORT_WINDOW_S = 60 * ONE_DAY

# Hard cap on the remembered cohort per product, independent of the time
# window. Protects against a single viral product turning the pass quadratic.
COHORT_MAX = 30

# Pair density is O(k^2) in the cohort, so it reads a shorter prefix than the
# tie features do. 20 reviewers is 190 pairs per row -- affordable, and a farm
# dense enough to matter shows up in any window this size.
DENSITY_COHORT = 20

# Co-occurrence inside this window counts as arriving *together* rather than
# merely arriving eventually.
TIGHT_WINDOW_S = 2 * ONE_DAY

# A tie this strong (see `_surprise`) promotes the pair to an edge in the
# component graph. Roughly: two overlaps on quiet products, or three or four
# on busy ones.
STRONG_TIE = 0.6

# ...and never on a single shared product, however obscure it was. One overlap
# is the base rate of a marketplace: without this guard the second reviewer of
# any newly listed item is "tied" to the first, and every quiet product in the
# catalogue manufactures a conspiracy.
TIE_MIN_SHARED = 2

# Jaccard overlap at which two accounts are "the same operator's", used for
# the counting features. 0.15 means a sixth of everything one of them has ever
# touched is shared with the other.
STRONG_JACCARD = 0.15


def _surprise(degree: int) -> float:
    """How much evidence one shared product is worth, given its popularity.

    THE FAILURE THIS EXISTS TO PREVENT
    ----------------------------------
    The first version of this module counted shared products raw, and every
    feature came out *anti*-correlated with fraud: AUC 0.36, pointing firmly
    at "genuine". The reason is that raw overlap measures how much an account
    posts, not who it works with. A prolific real reviewer buys the same
    bestsellers as thousands of other real people and accumulates an enormous
    neighbourhood; a farm account posts eight times and has a small one. Rank
    by neighbourhood size and you have built a detector for enthusiasm.

    Co-review on a product 40,000 people reviewed is not evidence of anything.
    Co-review on one that nine people reviewed is. This is the Adamic-Adar
    weighting from link prediction, and it is the difference between a feature
    that measures coordination and one that measures activity.
    """
    return 1.0 / np.log2(2.0 + degree)


GRAPH_COLUMNS: list[str] = [
    # --- the author's co-review neighbourhood, normalised for activity ---
    "gp_partner_count",
    "gp_partner_rate",
    "gp_max_shared_products",
    "gp_max_partner_jaccard",
    "gp_max_partner_surprise",
    "gp_partner_concentration",
    "gp_strong_partner_count",
    "gp_strong_partner_share",
    # --- the tight-tie component the author sits in ---
    "gp_component_size",
    "gp_component_density",
    # --- how the author relates to THIS product's existing reviewers ---
    "gp_tie_to_product_share",
    "gp_tie_to_product_surprise",
    # --- how tightly this product's own reviewers are interconnected ---
    "gp_cohort_size",
    "gp_cohort_strong_pair_density",
    "gp_cohort_mean_surprise",
    # --- temporal co-arrival ---
    "gp_tight_cooccur_max",
    "gp_tight_cooccur_rate",
    "gp_tight_partner_count",
]


@dataclass(slots=True)
class _ReviewerNode:
    """One account's co-review neighbourhood.

    `partners` maps a partner to [shared products, summed surprise, promoted].
    The third slot records whether the pair has already become a component
    edge, since promotion must happen exactly once. It is
    held in full because both the per-row summaries and the tie-to-product
    intersection need it; the scalars beside it are the ones cheap enough to
    maintain on every edge update.

    `products` doubles as the account's degree in the bipartite graph, which
    is what makes the Jaccard normalisation possible: overlap only means
    something relative to how much each side posts.
    """

    partners: dict[str, list[float]] = field(default_factory=dict)
    tight: dict[str, int] = field(default_factory=dict)
    products: set[str] = field(default_factory=set)
    n_reviews: int = 0
    surprise_sum: float = 0.0
    surprise_max: float = 0.0
    shared_max: int = 0
    tight_max: int = 0


class _Components:
    """Union-find over the strong-tie graph, with per-component edge counts.

    Union-find is the one clustering primitive that is natively streaming:
    unions only ever accumulate, never retract, so the component a review sees
    is by construction the component as it stood at that moment. Running
    community detection instead would mean re-partitioning the graph per row,
    which is the same intractability that rules out embeddings.

    Edges are admitted only once a pair's summed surprise clears `STRONG_TIE`.
    Admitting every co-review instead collapses the whole marketplace into one
    component within a few thousand rows -- popular products alone connect
    half of it -- and a feature reading "component size: 38,000" for every
    account measures nothing. Because surprise only accumulates, an edge that
    qualifies stays qualified, which is what keeps the union-find honest.
    """

    __slots__ = ("edges", "parent", "size")

    def __init__(self) -> None:
        self.parent: dict[str, str] = {}
        self.size: dict[str, int] = {}
        self.edges: dict[str, int] = {}

    def find(self, x: str) -> str:
        p = self.parent
        if x not in p:
            p[x] = x
            self.size[x] = 1
            self.edges[x] = 0
            return x
        root = x
        while p[root] != root:
            root = p[root]
        while p[x] != root:      # path compression
            p[x], x = root, p[x]
        return root

    def add_edge(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            self.edges[ra] += 1
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        self.edges[ra] += self.edges[rb] + 1

    def stats(self, x: str) -> tuple[int, float]:
        """(component size, edges per node) for the component holding `x`."""
        if x not in self.parent:
            return 1, 0.0
        r = self.find(x)
        n = self.size[r]
        return n, self.edges[r] / n if n else 0.0


def build_graph_features(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """Return causal co-review graph features aligned to `df`'s index.

    `df` must be sorted by timestamp ascending (schema.normalise guarantees it).
    """
    ts = df[TIMESTAMP]
    if not ts.is_monotonic_increasing:
        raise ValueError("input must be sorted by timestamp; call schema.normalise")

    unix = (ts.astype("int64") // 10 ** 9).to_numpy().astype(np.float64)
    reviewers = df[REVIEWER_ID].astype(str).to_numpy()
    products = df[PRODUCT_ID].astype(str).to_numpy()

    n = len(df)
    out = np.zeros((n, len(GRAPH_COLUMNS)), dtype=np.float32)
    col = {name: i for i, name in enumerate(GRAPH_COLUMNS)}

    nodes: dict[str, _ReviewerNode] = defaultdict(_ReviewerNode)
    # product -> chronological (time, reviewer) over the recent window.
    cohorts: dict[str, deque[tuple[float, str]]] = defaultdict(deque)
    # product -> distinct reviewers ever seen, which is the degree the
    # surprise weight is computed from. Counted causally, so being one of a
    # product's first reviewers is worth more than arriving once it is famous.
    degrees: dict[str, int] = defaultdict(int)
    comps = _Components()

    for i in range(n):
        t = unix[i]
        r = reviewers[i]
        p = products[i]
        node = nodes[r]
        cohort = cohorts[p]

        # Age the product's cohort out before reading it, so this review sees
        # the same window the next one will.
        while cohort and t - cohort[0][0] > COHORT_WINDOW_S:
            cohort.popleft()
        while len(cohort) > COHORT_MAX:
            cohort.popleft()

        my_degree = len(node.products)

        # ---------------- the author's neighbourhood ----------------------
        # One scan over the partner dict per row. The normalised quantities
        # cannot be maintained incrementally the way a sum can: a Jaccard
        # denominator moves whenever the *partner* posts somewhere, which is
        # an event this account never sees.
        n_partners = len(node.partners)
        out[i, col["gp_partner_count"]] = n_partners
        out[i, col["gp_partner_rate"]] = (
            n_partners / my_degree if my_degree else -1.0
        )

        if n_partners:
            best_jac = 0.0
            strong = 0
            for q, edge in node.partners.items():
                shared = edge[0]
                union = my_degree + len(nodes[q].products) - shared
                jac = shared / union if union > 0 else 0.0
                best_jac = max(best_jac, jac)
                if jac >= STRONG_JACCARD:
                    strong += 1
            out[i, col["gp_max_shared_products"]] = node.shared_max
            out[i, col["gp_max_partner_jaccard"]] = best_jac
            out[i, col["gp_max_partner_surprise"]] = node.surprise_max
            out[i, col["gp_strong_partner_count"]] = strong
            out[i, col["gp_strong_partner_share"]] = strong / n_partners
            # How much of the author's overlap sits with its single closest
            # associate. A real reviewer's overlap is smeared thinly over many
            # strangers; a sockpuppet pair's is concentrated in one account.
            out[i, col["gp_partner_concentration"]] = (
                node.surprise_max / node.surprise_sum if node.surprise_sum else 0.0
            )
        else:
            # No overlap with anyone yet -- distinct from "overlaps with many
            # accounts, weakly", which a 0-fill would merge.
            out[i, col["gp_max_shared_products"]] = -1.0
            out[i, col["gp_max_partner_jaccard"]] = -1.0
            out[i, col["gp_max_partner_surprise"]] = -1.0
            out[i, col["gp_strong_partner_share"]] = -1.0
            out[i, col["gp_partner_concentration"]] = -1.0

        comp_size, comp_density = comps.stats(r)
        out[i, col["gp_component_size"]] = comp_size
        out[i, col["gp_component_density"]] = comp_density

        out[i, col["gp_tight_cooccur_max"]] = node.tight_max if node.tight else -1.0
        out[i, col["gp_tight_partner_count"]] = len(node.tight)
        out[i, col["gp_tight_cooccur_rate"]] = (
            node.tight_max / node.n_reviews if node.n_reviews else -1.0
        )

        # ---------------- the author against this product's cohort --------
        # Distinct prior reviewers, keeping each one's most recent visit. Built
        # per row because the deque holds reviews, not accounts, and a repeat
        # reviewer must not count twice.
        seen: dict[str, float] = {}
        for tq, q in cohort:
            if q != r:
                seen[q] = tq
        cohort_ids = list(seen)
        k = len(cohort_ids)
        out[i, col["gp_cohort_size"]] = k

        if k:
            tie_surprise = 0.0
            tied = 0
            for q in cohort_ids:
                edge = node.partners.get(q)
                if edge is not None:
                    tied += 1
                    tie_surprise += edge[1]
            out[i, col["gp_tie_to_product_share"]] = tied / k
            out[i, col["gp_tie_to_product_surprise"]] = tie_surprise
        else:
            out[i, col["gp_tie_to_product_share"]] = -1.0
            out[i, col["gp_tie_to_product_surprise"]] = -1.0

        # ---------------- is this product's cohort a clique? --------------
        # The dense-subgraph signal, asked locally. Genuine buyers of a kettle
        # are strangers to each other and this is ~0; a farmed product's
        # reviewers have all met before, on products that were not bestsellers.
        # "Strong" here reads the promoted flag rather than re-testing the
        # threshold, so the component graph and this density can never drift
        # apart on what counts as a tie.
        recent = cohort_ids[-DENSITY_COHORT:]
        if len(recent) >= 2:
            pairs = 0
            strong_pairs = 0
            surprise_total = 0.0
            for a_idx in range(len(recent) - 1):
                a_partners = nodes[recent[a_idx]].partners
                for b_idx in range(a_idx + 1, len(recent)):
                    pairs += 1
                    edge = a_partners.get(recent[b_idx])
                    if edge is not None:
                        surprise_total += edge[1]
                        strong_pairs += int(edge[2])   # promoted, same rule
            out[i, col["gp_cohort_strong_pair_density"]] = strong_pairs / pairs
            out[i, col["gp_cohort_mean_surprise"]] = surprise_total / pairs
        else:
            out[i, col["gp_cohort_strong_pair_density"]] = -1.0
            out[i, col["gp_cohort_mean_surprise"]] = -1.0

        # ---------------- commit this review into the graph ---------------
        # Overlap is counted once per distinct shared PRODUCT, not per review
        # pair: an account reviewing the same item twice is a different
        # phenomenon, already covered by rv_product_repeat_share, and letting
        # it inflate an edge weight would make the tie threshold meaningless.
        first_visit = p not in node.products
        w = _surprise(degrees[p]) if first_visit else 0.0

        for q, tq in seen.items():
            other = nodes[q]
            if first_visit:
                edge = node.partners.get(q)
                if edge is None:
                    edge = [0.0, 0.0, 0.0]
                    node.partners[q] = edge
                    other.partners[r] = edge      # one list, shared both ways
                edge[0] += 1
                edge[1] += w
                shared, surprise, promoted = edge
                node.surprise_sum += w
                other.surprise_sum += w
                if shared > node.shared_max:
                    node.shared_max = int(shared)
                if shared > other.shared_max:
                    other.shared_max = int(shared)
                node.surprise_max = max(node.surprise_max, surprise)
                other.surprise_max = max(other.surprise_max, surprise)
                if not promoted and shared >= TIE_MIN_SHARED and surprise >= STRONG_TIE:
                    edge[2] = 1.0
                    comps.add_edge(r, q)
            if t - tq <= TIGHT_WINDOW_S:
                c = node.tight.get(q, 0) + 1
                node.tight[q] = c
                other.tight[r] = c
                node.tight_max = max(node.tight_max, c)
                other.tight_max = max(other.tight_max, c)

        if first_visit:
            node.products.add(p)
            degrees[p] += 1
        node.n_reviews += 1
        cohort.append((t, r))

        if verbose and i and i % 20_000 == 0:
            print(f"  graph features: {i:,}/{n:,}")

    return pd.DataFrame(out, columns=GRAPH_COLUMNS, index=df.index)
