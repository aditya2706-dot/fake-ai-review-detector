"""Product catalogue: the storefront metadata the review corpus does not carry.

WHY THIS FILE EXISTS
--------------------
The canonical review schema (`frd.schema`) has a `product_id` and a
`category` and nothing else. That is correct -- a detector needs the review,
the account and the timing, and inventing a product title would be inventing a
feature. But a *storefront* cannot render `P00782`, so the shopper-facing
surface needs a name, a brand, a price and something to look at.

WHY IT IS DERIVED RATHER THAN STORED
------------------------------------
Every field here is a pure function of `(product_id, category)`, seeded by a
stable hash of the id. That buys three things:

1. No new artifact to keep in sync with the corpus. `data/processed/` holds
   reviews; regenerating the corpus does not orphan a catalogue file, and a
   catalogue file cannot drift from the reviews it describes.
2. It works on *any* corpus, including real marketplace data, where products
   arrive that this repo has never seen. An unknown id gets a name on first
   request rather than a KeyError.
3. `hashlib`, not `hash()`. Python salts `hash()` per process, so a stored
   catalogue built with it would rename every product on restart.

Real deployments replace this module with a join against the product table.
The storefront asks for `catalog_for(ids, categories)` and does not care
which of the two it got -- see `frd.api.storefront`.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, replace

import numpy as np
import pandas as pd

# --- vocabulary ------------------------------------------------------------
#
# Per category: who makes it, what it is, and how it is qualified. Deliberately
# invented brands -- naming a real one would put fabricated review scores
# against a real company's product, which is exactly the kind of thing this
# repo's own governance section (ROADMAP phase 10) exists to refuse.

@dataclass(frozen=True, slots=True)
class CategorySpec:
    brands: tuple[str, ...]
    nouns: tuple[str, ...]
    modifiers: tuple[str, ...]
    price: tuple[float, float]      # lognormal-ish band, in whole currency
    glyph: str
    hue: tuple[int, int]            # degrees, for the generated product art
    features: tuple[str, ...]


CATEGORY_SPECS: dict[str, CategorySpec] = {
    "electronics": CategorySpec(
        brands=("Aurora", "Kestrel", "Novaline", "Halcyon", "Tessel", "Voltaic",
                "Meridian", "Cobalt Ridge", "Signalcraft", "Orbex"),
        nouns=("Monitor", "Headphones", "Mechanical Keyboard", "Webcam",
               "Portable SSD", "Bluetooth Speaker", "USB-C Hub", "Power Bank",
               "Soundbar", "Wireless Mouse", "Studio Microphone", "Router"),
        modifiers=('27" 4K', "Pro", "Mk II", "Wireless", "Studio", "Compact",
                   "Series 7", "Nano", "XL", "Low-Latency"),
        price=(24.0, 649.0),
        glyph="◈",
        hue=(198, 262),
        features=("Two-year manufacturer warranty", "USB-C, 100W passthrough",
                  "Firmware updates for 5 years", "Recycled aluminium chassis",
                  "Works with Windows, macOS and Linux"),
    ),
    "office": CategorySpec(
        brands=("Fieldnote", "Cartwright", "Ledger & Co", "Ashgrove", "Penrose",
                "Quillard", "Straightedge", "Bindery", "Northmark", "Vellum"),
        nouns=("Notebook", "Desk Lamp", "Monitor Arm", "Fountain Pen",
               "Document Scanner", "Standing Desk Mat", "Label Maker",
               "Filing Box", "Whiteboard", "Ergonomic Footrest",
               "Laser Pointer", "Desk Organiser"),
        modifiers=("A5", "Dotted", "Adjustable", "Refillable", "Slimline",
                   "Executive", "Matte Black", "Two-Pack", "Heavy-Duty"),
        price=(6.0, 289.0),
        glyph="▤",
        hue=(28, 46),
        features=("FSC-certified paper", "Lies flat when open",
                  "Tool-free assembly", "Ships in plastic-free packaging",
                  "Refills sold separately"),
    ),
    "home_kitchen": CategorySpec(
        brands=("Copperfield", "Hearthstone", "Marrow & Bell", "Saltmoor",
                "Ovenbird", "Clay Lane", "Tindale", "Brightwell", "Ferro",
                "Kettleworth"),
        nouns=("Cast Iron Skillet", "Stand Mixer", "Chef's Knife",
               "French Press", "Air Fryer", "Storage Set", "Kettle",
               "Blender", "Baking Tray", "Espresso Grinder", "Dutch Oven",
               "Cutting Board"),
        modifiers=("10-Inch", "Pre-Seasoned", "Stainless", "1.7L", "Six-Piece",
                   "Non-Stick", "Professional", "Enamelled", "Burr"),
        price=(11.0, 379.0),
        glyph="◍",
        hue=(4, 22),
        features=("Dishwasher safe", "Oven safe to 260°C",
                  "Lifetime guarantee on the body", "BPA-free throughout",
                  "Induction, gas and electric compatible"),
    ),
    "sports": CategorySpec(
        brands=("Ridgeway", "Summit Fell", "Tailwind", "Cadence", "Ironbark",
                "Stridelab", "Peakform", "Windward", "Basalt", "Trailhead"),
        nouns=("Running Shoes", "Yoga Mat", "Resistance Bands", "Dumbbell Set",
               "Cycling Helmet", "Hydration Pack", "Foam Roller",
               "Jump Rope", "Trekking Poles", "Kettlebell", "Fitness Tracker",
               "Swim Goggles"),
        modifiers=("Trail", "6mm", "Adjustable", "Carbon", "Lightweight",
                   "All-Weather", "Competition", "Vented", "Pro"),
        price=(9.0, 329.0),
        glyph="▲",
        hue=(96, 156),
        features=("Tested to EN standards", "Machine washable",
                  "Non-slip textured surface", "Packs down to 1.2kg",
                  "Sweat and salt resistant"),
    ),
    "beauty": CategorySpec(
        brands=("Lumen & Rye", "Petalwork", "Osmyn", "Verawild", "Silke",
                "Bloomcourt", "Attar", "Northlight", "Cerulea", "Maren"),
        nouns=("Facial Serum", "Hair Dryer", "Sunscreen", "Cleansing Balm",
               "Lip Treatment", "Makeup Brush Set", "Body Lotion",
               "Retinol Cream", "Hair Oil", "Electric Trimmer",
               "Eye Cream", "Face Mask Set"),
        modifiers=("SPF 50", "Fragrance-Free", "Vitamin C", "Ionic",
                   "Ten-Piece", "Overnight", "Daily", "Ceramide",
                   "Dermatologist-Tested"),
        price=(7.0, 189.0),
        glyph="✦",
        hue=(300, 344),
        features=("Dermatologist tested", "Free from parabens and sulphates",
                  "Not tested on animals", "Suitable for sensitive skin",
                  "Recyclable glass bottle"),
    ),
    "pet_supplies": CategorySpec(
        brands=("Barkfield", "Whiskerton", "Paw & Pine", "Kibblecraft",
                "Meadowpaw", "Tuftly", "Rundle", "Companion Row", "Fetchwell",
                "Burrow"),
        nouns=("Dog Bed", "Cat Tree", "Slow Feeder Bowl", "Lead & Harness Set",
               "Litter Box", "Chew Toy", "Grooming Brush", "Pet Carrier",
               "Water Fountain", "Training Pads", "Scratching Post",
               "Puzzle Feeder"),
        modifiers=("Large", "Orthopaedic", "Washable", "No-Pull", "Enclosed",
                   "Indestructible", "Two-Pack", "Stainless", "Quiet"),
        price=(6.0, 219.0),
        glyph="❋",
        hue=(168, 196),
        features=("Machine washable cover", "Chew-resistant stitching",
                  "Vet recommended", "Non-toxic materials throughout",
                  "Suits pets up to 40kg"),
    ),
    "clothing": CategorySpec(
        brands=("Thornwood", "Ellery", "Saltmarsh", "Foldwear", "Craneleigh",
                "Wren & Ash", "Bellweather", "Loomhouse", "Idris", "Kittiwake"),
        nouns=("Merino Jumper", "Rain Jacket", "Chino Trousers", "Oxford Shirt",
               "Wool Socks", "Leather Belt", "Packable Gilet", "Linen Dress",
               "Fleece Hoodie", "Overshirt", "Knit Beanie", "Denim Jacket"),
        modifiers=("Slim Fit", "Waterproof", "Three-Pack", "Organic Cotton",
                   "Midweight", "Relaxed", "Insulated", "Stretch", "Unlined"),
        price=(12.0, 249.0),
        glyph="◇",
        hue=(212, 238),
        features=("Machine wash at 30°C", "Certified organic cotton",
                  "Reinforced seams at stress points",
                  "Sizes XS to 3XL", "Made in a Fair Wear audited factory"),
    ),
    "toys": CategorySpec(
        brands=("Puzzlecraft", "Tinkerfield", "Brightblock", "Cogsworth",
                "Marbleworks", "Playhaven", "Wonderbench", "Fablewood",
                "Rumpus", "Knacks"),
        nouns=("Building Set", "Wooden Train Set", "Jigsaw Puzzle",
               "Board Game", "Science Kit", "Ride-On Car", "Plush Bear",
               "Marble Run", "Craft Box", "Remote Control Drone",
               "Magnetic Tiles", "Card Game"),
        modifiers=("1000-Piece", "Ages 3+", "Deluxe", "Starter",
                   "100-Piece", "Family Edition", "Beginner", "Motorised",
                   "Collector's"),
        price=(8.0, 159.0),
        glyph="✧",
        hue=(44, 74),
        features=("CE and UKCA certified", "No small parts for under-3s",
                  "Batteries included", "FSC-certified beech",
                  "Compatible with major building-brick brands"),
    ),
}

# Any category the corpus contains that the vocabulary above does not know.
_FALLBACK = CategorySpec(
    brands=("Kestrel", "Meridian", "Northmark", "Ferro", "Trailhead"),
    nouns=("Everyday Essential", "Standard Kit", "Replacement Set"),
    modifiers=("Standard", "Two-Pack", "Large"),
    price=(9.0, 199.0),
    glyph="●",
    hue=(210, 220),
    features=("Ships within one working day", "30-day returns"),
)

_SERIES = ("", "Pro", "V2", "S7", "Air", "Mk II", "Plus", "Studio", "One",
           "Max", "Lite", "Edge")


@dataclass(frozen=True, slots=True)
class Product:
    """Storefront metadata for one product id."""

    product_id: str
    name: str
    brand: str
    category: str
    price: float
    currency: str
    features: tuple[str, ...]
    # Handed to the client so generated product art is identical everywhere it
    # is drawn. Deriving it in TypeScript from the id would be a second
    # implementation of this file's hash, and would drift the first time either
    # side changed.
    hue: int
    hue_shift: int
    glyph: str

    def to_dict(self) -> dict:
        d = asdict(self)
        d["features"] = list(self.features)
        return d


def _seed(product_id: str) -> int:
    """A process-stable seed. `hash()` is salted per interpreter; this is not."""
    digest = hashlib.blake2b(product_id.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def product_for(product_id: str, category: str, currency: str = "USD") -> Product:
    """Deterministic storefront metadata for one product.

    Same id in, same product out -- across processes, machines and restarts.
    """
    spec = CATEGORY_SPECS.get(str(category), _FALLBACK)
    rng = np.random.default_rng(_seed(str(product_id)))

    brand = str(rng.choice(spec.brands))
    noun = str(rng.choice(spec.nouns))
    modifier = str(rng.choice(spec.modifiers))
    series = str(rng.choice(_SERIES))

    pattern = int(rng.integers(0, 3))
    if pattern == 0:
        name = f"{brand} {modifier} {noun}"
    elif pattern == 1:
        name = f"{brand} {noun} {modifier}"
    else:
        name = f"{brand} {series} {noun}".replace("  ", " ")

    # Log-uniform inside the category band, so a category holds a few expensive
    # items and many cheap ones rather than a flat spread.
    lo, hi = spec.price
    raw = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    price = round(raw) - 0.01 if raw >= 20 else round(raw * 2) / 2 - 0.01
    price = max(price, 1.99)

    n_feat = min(len(spec.features), 3)
    features = tuple(
        str(f) for f in rng.choice(spec.features, size=n_feat, replace=False)
    )

    hue = int(rng.integers(spec.hue[0], spec.hue[1] + 1))

    return Product(
        product_id=str(product_id),
        name=name.strip(),
        brand=brand,
        category=str(category),
        price=round(float(price), 2),
        currency=currency,
        features=features,
        hue=hue,
        hue_shift=int(rng.integers(18, 55)),
        glyph=spec.glyph,
    )


def catalog_for(
    product_ids: Sequence[str] | Iterable[str],
    categories: Sequence[str] | Iterable[str],
    currency: str = "USD",
) -> dict[str, Product]:
    """Metadata for many products, keyed by id.

    `categories` is positional with `product_ids`; the first category seen for
    an id wins, so passing a full review frame's two columns works directly.
    """
    out: dict[str, Product] = {}
    for pid, cat in zip(product_ids, categories):
        key = str(pid)
        if key not in out:
            out[key] = product_for(key, cat, currency=currency)

    # Distinct ids can land on the same name -- ~2% of them on an 888-product
    # corpus, which on a grid reads as a rendering bug rather than as two
    # similar listings. `product_for` cannot see the collision (it is a pure
    # function of one id, which is the property that makes it cacheable), so
    # the set-aware caller resolves it. Order is by id, so the winner and the
    # renamed loser are stable across runs.
    seen: dict[str, str] = {}
    for key in sorted(out):
        prod = out[key]
        if prod.name not in seen:
            seen[prod.name] = key
            continue
        suffix = key[-3:] if len(key) >= 3 else key
        out[key] = replace(prod, name=f"{prod.name} ({suffix})")
        seen[out[key].name] = key
    return out


def catalog_frame(df: pd.DataFrame, currency: str = "USD") -> pd.DataFrame:
    """One row of catalogue metadata per distinct product in a review frame."""
    pairs = df[["product_id", "category"]].drop_duplicates("product_id")
    cat = catalog_for(pairs["product_id"], pairs["category"], currency=currency)
    return pd.DataFrame([p.to_dict() for p in cat.values()]).set_index("product_id")
