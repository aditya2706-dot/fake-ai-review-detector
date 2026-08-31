"""Synthetic marketplace generator.

WHY THIS EXISTS
---------------
Public fake-review corpora are small, awkwardly licensed, or lack the
behavioural columns (reviewer id, timestamp, verified flag) that the strongest
signals come from. This module simulates a marketplace so the whole pipeline is
runnable on a laptop with no downloads, and swaps out cleanly for real data via
`frd.data.loaders`.

DESIGN CONSTRAINT -- the honest-generator rule
---------------------------------------------
A naive generator makes fakes trivially separable (fake = template, genuine =
free text), the model scores 0.99 AUC, and the project has measured nothing.
So this generator deliberately fights itself:

  * both classes draw from ONE shared vocabulary pool
  * `sophisticated_frac` of fakes are high-effort: specific, long, varied,
    sometimes verified via a refund-for-review scheme -- these are detectable
    only from behaviour, not text
  * `promo_genuine_frac` of genuine reviews are gushing brand-loyalist posts
    that look promotional -- these are the false-positive trap
  * a share of fakes carry deliberately MODERATE ratings (3-4 stars), the
    real tactic farms use to keep a product's rating histogram organic --
    without these the data would teach the model that "extreme rating" is
    nearly sufficient, and the model would miss the whole tactic
  * `label_noise` flips a small share of labels, because human moderators
    who produce real ground truth are themselves imperfect
  * farms are organisations with reusable account ROSTERS, so the same faces
    recur across one farm's products -- the coordinated structure that graph
    features exist to find, and that a flat account pool does not produce
  * farm accounts post genuine COVER reviews, so the account id is not itself
    the label and the model has to judge the review rather than the reviewer

The result is a dataset where an honest model lands in a realistic band
(PR-AUC roughly 0.75-0.90) instead of a suspicious 0.99.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import SynthConfig
from ..schema import (
    CATEGORY, HELPFUL, LABEL, PRODUCT_ID, RATING, REVIEW_ID, REVIEWER_ID,
    TEXT, TIMESTAMP, TITLE, VERIFIED, normalise,
)

CATEGORIES = [
    "electronics", "home_kitchen", "beauty", "sports", "toys",
    "clothing", "pet_supplies", "office",
]

# --- Shared vocabulary -----------------------------------------------------
# Deliberately shared between classes. Separation must come from HOW these are
# assembled (specificity, burstiness, repetition), not from magic words.

POSITIVE_ADJ = [
    "great", "excellent", "amazing", "solid", "decent", "fantastic", "sturdy",
    "comfortable", "lightweight", "reliable", "sleek", "affordable", "handy",
    "well-made", "impressive", "perfect", "good", "nice", "durable", "quality",
]
NEGATIVE_ADJ = [
    "flimsy", "disappointing", "cheap", "useless", "broken", "awful",
    "overpriced", "unreliable", "poor", "terrible", "defective", "shoddy",
]
ASPECTS = [
    "battery life", "build quality", "packaging", "delivery", "price",
    "instructions", "customer service", "material", "size", "colour",
    "weight", "noise level", "charging", "assembly", "warranty", "fit",
]
GENERIC_OPENERS = [
    "I love this product", "Highly recommend", "Best purchase ever",
    "Exactly what I wanted", "Five stars", "Great value for money",
    "Very happy with this", "Would buy again", "Excellent product",
    "This is a must buy", "Totally worth it", "Amazing quality",
]
SPECIFIC_DETAILS = [
    "arrived two days early in a battered box",
    "the second screw hole was slightly misaligned so I had to widen it",
    "measured 31cm not the 29cm listed",
    "ran for about 6 hours before needing a charge, not the advertised 9",
    "the app pairing failed twice before it stuck",
    "I've had it three months now and the finish has started to wear",
    "returned my first one because the seam split, the replacement is fine",
    "compared it side by side with the Anker version I already owned",
    "used it daily on my commute for six weeks",
    "my partner is 5'2\" and the medium was still too long",
    "the manual is a single badly translated sheet",
    "washed it four times and it hasn't shrunk",
    "the motor got noticeably warm after twenty minutes of use",
    "cheaper on the high street but this shipped faster",
]
HEDGES = [
    "though", "but honestly", "that said", "to be fair", "mind you",
    "for the price", "not a dealbreaker", "worth noting",
]
TYPO_MAP = {"the": "teh", "really": "realy", "definitely": "definately",
            "recommend": "recomend", "quality": "quallity", "received": "recieved"}


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _maybe_typo(text: str, rng: np.random.Generator, p: float) -> str:
    if rng.random() > p:
        return text
    for src, bad in TYPO_MAP.items():
        if src in text:
            return text.replace(src, bad, 1)
    return text


def _genuine_text(rng: np.random.Generator, rating: float, promo: bool) -> str:
    """A real customer: usually specific, variable length, imperfect."""
    positive = rating >= 4
    adj = POSITIVE_ADJ if positive else NEGATIVE_ADJ

    if promo:
        # Brand loyalist. Reads like an ad but is a genuine human -- the
        # false-positive trap the model has to survive.
        parts = [
            rng.choice(GENERIC_OPENERS) + "!",
            f"The {rng.choice(ASPECTS)} is {rng.choice(POSITIVE_ADJ)}.",
            rng.choice(["Buy it.", "You won't regret it.", "10/10.", "Obsessed."]),
        ]
        return " ".join(parts)

    n_details = int(rng.integers(1, 4))
    parts: list[str] = []

    # Real reviews often open on an aspect rather than a slogan.
    if rng.random() < 0.35:
        parts.append(rng.choice(GENERIC_OPENERS) + ".")
    parts.append(
        f"The {rng.choice(ASPECTS)} is {rng.choice(adj)} and the "
        f"{rng.choice(ASPECTS)} is {rng.choice(adj)}."
    )
    for _ in range(n_details):
        parts.append(rng.choice(SPECIFIC_DETAILS).capitalize() + ".")
    if rng.random() < 0.45:
        parts.append(
            f"{rng.choice(HEDGES).capitalize()}, the "
            f"{rng.choice(ASPECTS)} could be better."
        )
    if not positive and rng.random() < 0.5:
        parts.append(rng.choice([
            "Requested a refund.", "Sending it back.",
            "Contacted the seller, no reply after a week.",
        ]))

    # Genuine reviews are sometimes just a fragment.
    if rng.random() < 0.12:
        return rng.choice(["Works fine.", "Does the job.", "No complaints.",
                           "Broke after a week.", "As described."])

    return _maybe_typo(" ".join(parts), rng, p=0.22)


def _fake_text(
    rng: np.random.Generator,
    rating: float,
    template: str | None,
    sophisticated: bool,
) -> str:
    """A paid reviewer.

    Unsophisticated: superlative-dense, aspect-vague, template-derived.
    Sophisticated: borrows the genuine generator's specificity, so text
    features alone will not catch it.
    """
    if sophisticated:
        base = _genuine_text(rng, rating, promo=False)
        # Tell: still slightly over-eggs the close.
        if rng.random() < 0.5:
            base += " " + rng.choice([
                "Genuinely impressed.", "Great seller too.",
                "Fast shipping as well.",
            ])
        return base

    positive = rating >= 4
    adj_pool = POSITIVE_ADJ if positive else NEGATIVE_ADJ
    opener = template or rng.choice(GENERIC_OPENERS)

    parts = [opener + "!"]
    for _ in range(int(rng.integers(1, 4))):
        parts.append(
            f"{rng.choice(['Really','Very','Super','So','Absolutely'])} "
            f"{rng.choice(adj_pool)} {rng.choice(ASPECTS)}."
        )
    parts.append(rng.choice([
        "Highly recommend to everyone!", "Do not hesitate, just buy it!",
        "Best on the market!", "Will definitely order again!",
        "Five stars all the way!", "You need this in your life!",
    ]))
    return " ".join(parts)


def _title(rng: np.random.Generator, rating: float) -> str:
    if rating >= 4:
        return rng.choice(["Great!", "Love it", "Excellent", "Very pleased",
                           "Does the job", "Recommended", "Happy customer"])
    return rng.choice(["Disappointed", "Not as described", "Avoid",
                       "Broke quickly", "Meh", "Waste of money"])


def generate(cfg: SynthConfig | None = None) -> pd.DataFrame:
    """Build a labelled synthetic review corpus in canonical schema."""
    cfg = cfg or SynthConfig()
    rng = _rng(cfg.seed)

    products = np.array([f"P{i:05d}" for i in range(cfg.n_products)])
    prod_category = rng.choice(CATEGORIES, size=cfg.n_products)
    # Each product has a latent true quality -> genuine ratings centre on it.
    prod_quality = np.clip(rng.normal(3.9, 0.7, cfg.n_products), 1.2, 5.0)
    # Product popularity is heavy-tailed, as on any real marketplace.
    prod_weight = rng.pareto(1.4, cfg.n_products) + 0.2
    prod_weight /= prod_weight.sum()

    start = pd.Timestamp("2024-01-01", tz="UTC")
    rows: list[dict] = []
    rid = 0

    # ---------------- coordinated campaigns ----------------
    # A campaign = a seller buying a burst of reviews for one product, run by
    # a farm, within a short window, around one template.
    #
    # WHY FARMS HAVE ROSTERS
    # ----------------------
    # An earlier version of this generator drew each campaign's workers
    # independently from one flat pool of 1,200 accounts, and commented that
    # the reuse was signal. It was not. With that many accounts and 70 jobs,
    # no two of them ever persistently associate: measured on the resulting
    # corpus, farm-farm pairs shared two or more products *less* often than
    # genuine pairs did (0.8% against 6.8%). The simulated marketplace
    # contained no coordinated structure at all, which made the graph block of
    # `features/graph.py` untestable -- it correctly reported that there was
    # nothing there.
    #
    # Real farms are organisations. They hold a roster of aged accounts and
    # work it job after job, because accounts are the expensive asset. So a
    # campaign draws from one farm's roster, and the same faces keep appearing
    # together across that farm's products. That is the structure the co-review
    # graph exists to find, and simulating a marketplace without it was the
    # unrealistic choice, not the other way round.
    n_camp = cfg.n_campaigns
    n_farms = max(1, cfg.n_farms)
    roster_sizes = rng.integers(35, 90, size=n_farms)
    farm_rosters: list[np.ndarray] = []
    next_account = 0
    for f in range(n_farms):
        ids = np.array([
            f"F{i:05d}" for i in range(next_account, next_account + int(roster_sizes[f]))
        ])
        farm_rosters.append(ids)
        next_account += int(roster_sizes[f])
    all_farm_ids = np.concatenate(farm_rosters)

    camp_farm = rng.integers(0, n_farms, size=n_camp)
    camp_prod = rng.choice(cfg.n_products, size=n_camp, p=prod_weight)
    camp_start = rng.uniform(0, cfg.days - 12, size=n_camp)
    camp_span = rng.uniform(0.5, 9.0, size=n_camp)        # burst window, days
    camp_size = rng.integers(6, 55, size=n_camp)
    camp_attack = rng.random(n_camp) < 0.22               # 1-star competitor hit
    camp_template = [rng.choice(GENERIC_OPENERS) for _ in range(n_camp)]

    fake_rows: list[dict] = []

    for c in range(n_camp):
        roster = farm_rosters[camp_farm[c]]
        # One account does not review the same product twice for one job.
        size = min(int(camp_size[c]), len(roster))
        workers = rng.choice(roster, size=size, replace=False)
        for w in workers:
            attack = bool(camp_attack[c])
            sophisticated = bool(rng.random() < 0.30)
            # A crude campaign posts nothing but 5s (or 1s for an attack),
            # which any rating-distribution monitor spots. Sophisticated farms
            # deliberately mix in measured 3-4 star reviews to keep the
            # product's rating histogram looking organic. Those are the ones
            # that evade a detector keyed on extreme ratings -- so the data has
            # to contain them.
            if attack:
                rating = float(rng.choice([1, 1, 2, 3]))
            elif sophisticated and rng.random() < 0.40:
                rating = float(rng.choice([4, 3]))
            else:
                rating = float(rng.choice([5, 5, 5, 4]))
            day = camp_start[c] + rng.uniform(0, camp_span[c])
            fake_rows.append({
                REVIEW_ID: f"R{rid:07d}",
                PRODUCT_ID: products[camp_prod[c]],
                REVIEWER_ID: str(w),
                RATING: rating,
                TITLE: _title(rng, rating),
                TEXT: _fake_text(
                    rng, rating,
                    None if sophisticated else camp_template[c],
                    sophisticated,
                ),
                TIMESTAMP: start + pd.Timedelta(days=float(day)),
                # Refund-for-review schemes produce *verified* fakes.
                VERIFIED: bool(rng.random() < (0.55 if sophisticated else 0.20)),
                HELPFUL: int(rng.poisson(0.35)),
                CATEGORY: prod_category[camp_prod[c]],
                LABEL: 1,
            })
            rid += 1

    # Lone-wolf fakes: incentivised individuals, no campaign structure. These
    # have no burst signal at all and are the hardest positive class.
    n_fake_target = int(cfg.n_reviews * cfg.fake_ratio)
    n_lone = max(0, n_fake_target - len(fake_rows))
    for _ in range(n_lone):
        p = int(rng.choice(cfg.n_products, p=prod_weight))
        soph = bool(rng.random() < 0.35)
        rating = float(
            rng.choice([4, 3]) if (soph and rng.random() < 0.35)
            else rng.choice([5, 5, 4, 1])
        )
        fake_rows.append({
            REVIEW_ID: f"R{rid:07d}",
            PRODUCT_ID: products[p],
            REVIEWER_ID: str(rng.choice(all_farm_ids)),
            RATING: rating,
            TITLE: _title(rng, rating),
            TEXT: _fake_text(rng, rating, None, soph),
            TIMESTAMP: start + pd.Timedelta(days=float(rng.uniform(0, cfg.days))),
            VERIFIED: bool(rng.random() < 0.30),
            HELPFUL: int(rng.poisson(0.4)),
            CATEGORY: prod_category[p],
            LABEL: 1,
        })
        rid += 1

    # ---------------- farm cover traffic ----------------
    # Farm accounts also post ordinary, genuine, unpaid reviews. They have to:
    # an account whose entire history is paid work is trivially identifiable,
    # and farms know it, so they pad rosters with real purchases and organic
    # opinions.
    #
    # Omitting this was the second unrealism in the original generator, and a
    # worse one than the flat account pool. Without cover traffic the account
    # id IS the label -- 98% of rows written by an F-account were fake -- so
    # any feature that effectively identified the account scored brilliantly
    # for a reason that would not survive contact with a real marketplace.
    # These rows are labelled 0 because they genuinely are genuine, which is
    # what makes the problem honest: the model must judge the review, not the
    # reviewer.
    n_cover = int(len(fake_rows) * cfg.farm_cover_ratio)
    cover_rows: list[dict] = []
    for _ in range(n_cover):
        p = int(rng.choice(cfg.n_products, p=prod_weight))
        rating = float(np.clip(round(rng.normal(prod_quality[p], 1.05)), 1, 5))
        cover_rows.append({
            REVIEW_ID: f"R{rid:07d}",
            PRODUCT_ID: products[p],
            REVIEWER_ID: str(rng.choice(all_farm_ids)),
            RATING: rating,
            TITLE: _title(rng, rating),
            TEXT: _genuine_text(rng, rating, bool(rng.random() < 0.10)),
            TIMESTAMP: start + pd.Timedelta(days=float(rng.uniform(0, cfg.days))),
            VERIFIED: bool(rng.random() < 0.78),
            HELPFUL: int(rng.poisson(1.4)),
            CATEGORY: prod_category[p],
            LABEL: 0,
        })
        rid += 1

    # ---------------- genuine traffic ----------------
    # Sized last, so the corpus still totals n_reviews once the campaigns and
    # their cover have taken their share.
    n_genuine = max(0, cfg.n_reviews - len(fake_rows) - len(cover_rows))

    genuine_ids = np.array([f"U{i:06d}" for i in range(cfg.n_genuine_reviewers)])
    # Reviewer activity is also heavy-tailed: a few prolific reviewers.
    user_weight = rng.pareto(1.6, cfg.n_genuine_reviewers) + 0.1
    user_weight /= user_weight.sum()

    g_user = rng.choice(cfg.n_genuine_reviewers, size=n_genuine, p=user_weight)
    g_prod = rng.choice(cfg.n_products, size=n_genuine, p=prod_weight)
    g_day = rng.uniform(0, cfg.days, size=n_genuine)
    # J-shaped rating distribution around latent quality.
    g_rating = np.clip(
        np.round(rng.normal(prod_quality[g_prod], 1.05)), 1, 5
    )
    g_promo = rng.random(n_genuine) < 0.10   # brand loyalists

    for k in range(n_genuine):
        rating = float(g_rating[k])
        rows.append({
            REVIEW_ID: f"R{rid:07d}",
            PRODUCT_ID: products[g_prod[k]],
            REVIEWER_ID: genuine_ids[g_user[k]],
            RATING: rating,
            TITLE: _title(rng, rating),
            TEXT: _genuine_text(rng, rating, bool(g_promo[k])),
            TIMESTAMP: start + pd.Timedelta(days=float(g_day[k])),
            VERIFIED: bool(rng.random() < 0.78),
            HELPFUL: int(rng.poisson(1.4)),
            CATEGORY: prod_category[g_prod[k]],
            LABEL: 0,
        })
        rid += 1

    rows.extend(cover_rows)
    rows.extend(fake_rows)
    df = pd.DataFrame(rows)

    # ---------------- label noise ----------------
    # Ground truth in this domain comes from human moderators, who are wrong
    # sometimes. Baking that in keeps evaluation honest.
    noise_mask = rng.random(len(df)) < 0.02
    df.loc[noise_mask, LABEL] = 1 - df.loc[noise_mask, LABEL]

    return normalise(df)
