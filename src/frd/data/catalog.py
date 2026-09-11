"""The product catalogue: the shop's listings, and how reviews attach to them.

WHAT CHANGED AND WHY
--------------------
This module used to *generate* products: a blake2b hash of `P00042` picked a
brand from a tuple, a noun from another, a price from a log-uniform band, and
the storefront drew a CSS gradient with a category glyph where a photograph
should be. It was defensible -- no artifact to keep in sync, works on any
corpus -- and it read exactly like what it was. "Aurora XL Bluetooth Speaker"
next to a lilac gradient is a schematic of a shop, not a shop.

So the catalogue is now hand-authored data (`products.json`): 61 listings with
real copy, specs, variants, stock and photography. That buys the thing the
generator could not -- a page a person believes -- at the cost of the thing it
had, which was covering an unbounded product space for free. `assign()` below
is where that cost is paid.

WHY THE BRANDS ARE STILL INVENTED
---------------------------------
Every listing here carries a fabricated review history and a fabricated
manipulation score. Hanging those off a real company's product would attach
invented fraud to a real business, which is the one thing this repo's
governance section refuses to do. The photographs were selected on the same
rule: `web/public/products/CREDITS.md` records the source of each, and none of
them has a legible brand in it.

HOW 888 CORPUS PRODUCTS BECOME 61 LISTINGS
------------------------------------------
The review corpus has 888 product ids; the shop has 61 listings. `assign()`
maps the two by ranking each department's corpus products by review count and
taking the top N, where N is however many listings that department has.

Two things about that rule are deliberate:

  * It is neutral. Ranking by *manipulation* instead would stock the shop with
    exactly the products that best demonstrate the detector, and every number
    on the home page would then be a selection effect. Review count is chosen
    before anything is scored and correlates with nothing the model does.
  * It is not neutral about popularity, and that is fine -- a busy product is
    also a campaign's preferred target (`frd.data.synth` weights both by the
    same Pareto draw), so the shop inherits a realistic share of manipulated
    listings rather than an artificially clean or artificially dirty one.

The moderation console still works on all 888. Only the shop is a subset, and
`/shop/stats` reports the two counts separately so the page never implies it
is showing the whole corpus.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

CATALOGUE_PATH = Path(__file__).with_name("products.json")

# Listings the shop shows even if their corpus product has almost no reviews.
# Nothing depends on this today -- it is here so a department whose corpus
# products are all tiny still renders rather than silently emptying.
MIN_CORPUS_REVIEWS = 0


@dataclass(frozen=True, slots=True)
class Variant:
    """One buyable combination of a product's options."""

    sku: str
    label: str                       # "Graphite / Large"
    choices: dict[str, str]          # {"Colour": "Graphite", "Size": "Large"}
    price: float
    stock: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Product:
    """One listing, plus the corpus product whose reviews it carries."""

    product_id: str                  # the corpus id -- what reviews join on
    slug: str
    sku: str
    name: str
    brand: str
    category: str
    department: str                  # display name, e.g. "Home & Kitchen"
    price: float
    compare_at: float | None
    currency: str
    summary: str
    description: str
    bullets: tuple[str, ...]
    specs: dict[str, str]
    options: tuple[dict[str, Any], ...]
    variants: tuple[Variant, ...]
    stock: int
    tags: tuple[str, ...]
    image: str

    @property
    def on_sale(self) -> bool:
        return self.compare_at is not None and self.compare_at > self.price

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["bullets"] = list(self.bullets)
        d["tags"] = list(self.tags)
        d["options"] = [dict(o) for o in self.options]
        d["variants"] = [v.to_dict() for v in self.variants]
        d["on_sale"] = self.on_sale
        return d


@lru_cache(maxsize=1)
def _raw() -> dict[str, Any]:
    return json.loads(CATALOGUE_PATH.read_text(encoding="utf-8"))


def departments() -> list[dict[str, str]]:
    """The shop's departments, in the order they should be displayed."""
    return [dict(d) for d in _raw()["departments"]]


def currency() -> str:
    return str(_raw()["currency"])


def _sku(slug: str) -> str:
    """A stable, readable SKU: initials of the slug plus a digit tail."""
    parts = [p for p in slug.split("-") if p]
    head = "".join(p[0] for p in parts[:3]).upper()
    tail = str(sum(ord(c) for c in slug) % 9000 + 1000)
    return f"{head}-{tail}"


def _option_code(values: Iterable[str]) -> str:
    """A short, readable code for one combination of option values."""
    parts = []
    for v in values:
        letters = "".join(c for c in str(v) if c.isalnum()).upper()
        parts.append(letters[:3] or "X")
    return "-".join(parts)


def _variants(slug: str, base_sku: str, price: float, stock: int,
              options: Sequence[Mapping[str, Any]]) -> tuple[Variant, ...]:
    """The cartesian product of the option groups, one buyable variant each.

    Stock is split across variants rather than repeated: a product with 40 in
    stock and 4 variants has 40 units in total, not 160. Spreading it unevenly
    (the first variants get the remainder) keeps a "2 left" state reachable in
    the UI without needing a separate fixture for it.
    """
    if not options:
        return (Variant(base_sku, "Default", {}, price, stock),)

    combos: list[dict[str, str]] = [{}]
    for group in options:
        name = str(group["name"])
        combos = [c | {name: str(v)} for c in combos for v in group["values"]]

    out: list[Variant] = []
    base, extra = divmod(stock, len(combos))
    used: set[str] = set()
    for i, choices in enumerate(combos):
        sku = f"{base_sku}-{_option_code(choices.values())}"
        # Distinct option values can still abbreviate to the same code --
        # "Cobalt" and "Charcoal" both start CO/CH only by luck, and two-letter
        # collisions are common enough that this was a real defect. A colliding
        # variant gets a numeric tail rather than silently overwriting the
        # first one in the cart, where a SKU is the primary key of a line.
        if sku in used:
            sku = f"{sku}{i}"
        used.add(sku)
        out.append(Variant(
            sku=sku,
            label=" / ".join(choices.values()),
            choices=choices,
            price=price,
            stock=base + (1 if i < extra else 0),
        ))
    return tuple(out)


@lru_cache(maxsize=1)
def _listings() -> tuple[dict[str, Any], ...]:
    """The catalogue file, parsed and enriched, still without a corpus id."""
    raw = _raw()
    dept_name = {d["key"]: d["name"] for d in raw["departments"]}
    cur = str(raw["currency"])
    out = []
    for item in raw["products"]:
        slug = str(item["slug"])
        sku = _sku(slug)
        price = float(item["price"])
        stock = int(item.get("stock", 0))
        options = tuple(dict(o) for o in item.get("options", []))
        out.append({
            "slug": slug,
            "sku": sku,
            "name": str(item["name"]),
            "brand": str(item["brand"]),
            "category": str(item["category"]),
            "department": dept_name.get(str(item["category"]), str(item["category"])),
            "price": price,
            "compare_at": (
                float(item["compare_at"])
                if item.get("compare_at") is not None else None
            ),
            "currency": cur,
            "summary": str(item.get("summary", "")),
            "description": str(item.get("description", "")),
            "bullets": tuple(str(b) for b in item.get("bullets", [])),
            "specs": {str(k): str(v) for k, v in item.get("specs", {}).items()},
            "options": options,
            "variants": _variants(slug, sku, price, stock, options),
            "stock": stock,
            "tags": tuple(str(t) for t in item.get("tags", [])),
            # Served, not derived client-side: the file name is the catalogue's
            # business, and a component that rebuilds it from the slug breaks
            # silently the first time a listing is renamed.
            "image": f"/products/{slug}.webp",
        })
    return tuple(out)


def listings() -> list[dict[str, Any]]:
    """Every catalogue listing, with no corpus product attached yet."""
    return [dict(item) for item in _listings()]


def assign(corpus: Mapping[str, tuple[str, int]]) -> dict[str, Product]:
    """Attach each listing to one corpus product id.

    `corpus` maps a corpus product id to `(category, n_reviews)`. Within each
    category the busiest corpus products win, in review-count order, ties
    broken by id so the assignment does not move between runs on a frame whose
    row order changed.

    Returns a dict keyed by **corpus product id** -- the join key the rest of
    the API already uses -- so callers that previously held a generated
    catalogue keep the same shape.
    """
    by_category: dict[str, list[str]] = {}
    for pid, (category, n) in corpus.items():
        if n >= MIN_CORPUS_REVIEWS:
            by_category.setdefault(str(category), []).append(str(pid))

    for category, ids in by_category.items():
        ids.sort(key=lambda p: (-corpus[p][1], p))

    out: dict[str, Product] = {}
    taken: dict[str, int] = {}
    for item in _listings():
        pool = by_category.get(item["category"], [])
        i = taken.get(item["category"], 0)
        if i >= len(pool):
            # The corpus has fewer products in this department than the shop
            # has listings for it. Skipping is right: a listing with no reviews
            # would render a rating of "no rating" and a review count of zero,
            # which is a real state for a real shop but a confusing one here.
            continue
        taken[item["category"]] = i + 1
        pid = pool[i]
        out[pid] = Product(product_id=pid, **item)
    return out


def by_slug(catalogue: Mapping[str, Product]) -> dict[str, Product]:
    """The same catalogue keyed by slug, for URL lookups."""
    return {p.slug: p for p in catalogue.values()}


# --- compatibility ---------------------------------------------------------
#
# `catalog_for` was the old entry point and took (ids, categories). It has no
# review counts to rank on, so it assigns in id order -- fine for a caller that
# only wants "some listing per id", which is all the old tests wanted.

def catalog_for(
    product_ids: Sequence[str] | Iterable[str],
    categories: Sequence[str] | Iterable[str],
    currency: str | None = None,
) -> dict[str, Product]:
    seen: dict[str, str] = {}
    for pid, cat in zip(product_ids, categories):
        seen.setdefault(str(pid), str(cat))
    # Rank by id, descending, so the first id in the caller's order wins the
    # first listing -- `assign` sorts by (-count, id) and count is constant here.
    order = {pid: (cat, len(seen) - i) for i, (pid, cat) in enumerate(seen.items())}
    return assign(order)


@lru_cache(maxsize=1)
def _slug_index() -> dict[str, dict[str, Any]]:
    return {item["slug"]: item for item in _listings()}


def listing_for(slug: str) -> dict[str, Any] | None:
    """One catalogue listing by slug, with no corpus product attached.

    This is the lookup the cart and the order writer use. They deliberately do
    NOT go through `assign()`: pricing a basket must not require a trained
    model and a scored corpus to be in memory, and a listing's price is a
    property of the catalogue, not of anything the detector computed.
    """
    item = _slug_index().get(str(slug))
    return dict(item) if item else None


def variant_for(slug: str, variant_sku: str) -> tuple[dict[str, Any], Variant] | None:
    """A (listing, variant) pair, or None if either half is unknown."""
    item = listing_for(slug)
    if item is None:
        return None
    for v in item["variants"]:
        if v.sku == variant_sku:
            return item, v
    return None
