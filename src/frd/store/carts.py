"""Baskets.

WHY A CART HAS ITS OWN ID
-------------------------
A shopper fills a basket, then signs in at the checkout. If the cart were keyed
on the user id it would not exist until that moment, and everything added
before signing in would be lost at exactly the point the shop was about to be
paid. So a cart gets a random id the first time something is added to it, that
id rides in a cookie, and `claim()` attaches it to a user when one appears.

WHY THE SERVER PRICES EVERYTHING
--------------------------------
The client sends a slug, a variant SKU and a quantity. It never sends a price.
Line prices, the subtotal, shipping and tax are all recomputed here from the
catalogue on every read, so a tampered request buys nothing at a discount and
a catalogue price change is reflected in a basket that was filled yesterday.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from ..data.catalog import currency, variant_for
from .db import connect, to_cents, to_major

# Free delivery over this much, a flat fee under it. Ordinary shop policy, and
# the number is here rather than in the UI so the basket and the checkout
# cannot disagree about it.
FREE_SHIPPING_OVER_CENTS = 7500
SHIPPING_FLAT_CENTS = 695
TAX_RATE = 0.085

MAX_QTY_PER_LINE = 10


class CartError(Exception):
    """Something the shopper should be told: unknown product, out of stock."""


def _now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def new_id() -> str:
    return f"C{uuid.uuid4().hex[:16]}"


def ensure(cart_id: str | None, user_id: str | None = None) -> str:
    """Return a usable cart id, creating the row if it does not exist yet.

    A signed-in shopper's cart is looked up by user id first, so signing in on
    a second device finds the basket rather than starting an empty one.
    """
    with connect() as conn:
        if user_id:
            row = conn.execute(
                "SELECT id FROM carts WHERE user_id = ? ORDER BY updated_at DESC "
                "LIMIT 1", (user_id,)).fetchone()
            if row and (cart_id is None or cart_id == row["id"]):
                return row["id"]

        if cart_id:
            row = conn.execute("SELECT id FROM carts WHERE id = ?",
                               (cart_id,)).fetchone()
            if row:
                return cart_id

        cid = cart_id or new_id()
        now = _now()
        conn.execute(
            "INSERT OR IGNORE INTO carts (id, user_id, created_at, updated_at) "
            "VALUES (?, ?, ?, ?)", (cid, user_id, now, now))
        return cid


def claim(cart_id: str, user_id: str) -> str:
    """Attach an anonymous cart to a user at sign-in, merging any existing one.

    Quantities are summed rather than overwritten: someone who added a kettle
    on their phone and a kettle on their laptop meant two kettles, and silently
    dropping one of them is the kind of bug a shop never hears about because
    the shopper simply does not notice until it arrives.
    """
    with connect() as conn:
        existing = conn.execute(
            "SELECT id FROM carts WHERE user_id = ? AND id != ? "
            "ORDER BY updated_at DESC LIMIT 1", (user_id, cart_id)).fetchone()

        conn.execute("INSERT OR IGNORE INTO carts (id, user_id, created_at, "
                     "updated_at) VALUES (?, ?, ?, ?)",
                     (cart_id, user_id, _now(), _now()))
        conn.execute("UPDATE carts SET user_id = ?, updated_at = ? WHERE id = ?",
                     (user_id, _now(), cart_id))

        if existing:
            for row in conn.execute(
                    "SELECT variant_sku, slug, qty FROM cart_items WHERE cart_id = ?",
                    (existing["id"],)).fetchall():
                conn.execute(
                    "INSERT INTO cart_items (cart_id, variant_sku, slug, qty, "
                    "added_at) VALUES (?, ?, ?, ?, ?) "
                    "ON CONFLICT(cart_id, variant_sku) DO UPDATE SET "
                    "qty = min(?, cart_items.qty + excluded.qty)",
                    (cart_id, row["variant_sku"], row["slug"], row["qty"],
                     _now(), MAX_QTY_PER_LINE))
            conn.execute("DELETE FROM carts WHERE id = ?", (existing["id"],))
    return cart_id


def add(cart_id: str, slug: str, variant_sku: str, qty: int = 1) -> None:
    found = variant_for(slug, variant_sku)
    if found is None:
        raise CartError(f"No such product or variant: {slug} / {variant_sku}")
    _, variant = found
    if variant.stock <= 0:
        raise CartError(f"{slug} ({variant.label}) is out of stock.")
    qty = max(1, min(int(qty), MAX_QTY_PER_LINE, variant.stock))

    with connect() as conn:
        conn.execute(
            "INSERT INTO cart_items (cart_id, variant_sku, slug, qty, added_at) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(cart_id, variant_sku) DO UPDATE SET "
            "qty = min(?, cart_items.qty + excluded.qty)",
            (cart_id, variant_sku, slug, qty, _now(),
             min(MAX_QTY_PER_LINE, variant.stock)))
        conn.execute("UPDATE carts SET updated_at = ? WHERE id = ?", (_now(), cart_id))


def set_qty(cart_id: str, variant_sku: str, qty: int) -> None:
    qty = int(qty)
    with connect() as conn:
        if qty <= 0:
            conn.execute("DELETE FROM cart_items WHERE cart_id = ? AND variant_sku = ?",
                         (cart_id, variant_sku))
        else:
            conn.execute(
                "UPDATE cart_items SET qty = ? WHERE cart_id = ? AND variant_sku = ?",
                (min(qty, MAX_QTY_PER_LINE), cart_id, variant_sku))
        conn.execute("UPDATE carts SET updated_at = ? WHERE id = ?", (_now(), cart_id))


def remove(cart_id: str, variant_sku: str) -> None:
    set_qty(cart_id, variant_sku, 0)


def clear(cart_id: str) -> None:
    with connect() as conn:
        conn.execute("DELETE FROM cart_items WHERE cart_id = ?", (cart_id,))
        conn.execute("UPDATE carts SET updated_at = ? WHERE id = ?", (_now(), cart_id))


def lines(cart_id: str) -> list[dict[str, Any]]:
    """The basket's contents, priced from the catalogue as it stands now.

    A line whose product has left the catalogue is dropped rather than rendered
    half-resolved, and the drop is permanent -- leaving a dangling row would
    make the basket un-checkout-able with no way for the shopper to fix it.
    """
    with connect() as conn:
        rows = conn.execute(
            "SELECT variant_sku, slug, qty FROM cart_items WHERE cart_id = ? "
            "ORDER BY added_at", (cart_id,)).fetchall()

    out: list[dict[str, Any]] = []
    stale: list[str] = []
    for row in rows:
        found = variant_for(row["slug"], row["variant_sku"])
        if found is None:
            stale.append(row["variant_sku"])
            continue
        item, variant = found
        qty = min(int(row["qty"]), MAX_QTY_PER_LINE)
        unit = to_cents(variant.price)
        out.append({
            "variant_sku": variant.sku,
            "slug": item["slug"],
            "name": item["name"],
            "brand": item["brand"],
            "variant_label": "" if variant.label == "Default" else variant.label,
            "image": item["image"],
            "unit_price": to_major(unit),
            "qty": qty,
            "line_total": to_major(unit * qty),
            "stock": variant.stock,
            "in_stock": variant.stock > 0,
        })

    if stale:
        with connect() as conn:
            conn.executemany(
                "DELETE FROM cart_items WHERE cart_id = ? AND variant_sku = ?",
                [(cart_id, sku) for sku in stale])
    return out


def totals(cart_lines: list[dict[str, Any]]) -> dict[str, Any]:
    """Subtotal, shipping, tax and total -- the one implementation of the sums."""
    subtotal = sum(to_cents(line["line_total"]) for line in cart_lines)
    shipping = 0 if (subtotal == 0 or subtotal >= FREE_SHIPPING_OVER_CENTS) \
        else SHIPPING_FLAT_CENTS
    tax = round(subtotal * TAX_RATE)
    return {
        "subtotal": to_major(subtotal),
        "shipping": to_major(shipping),
        "tax": to_major(tax),
        "total": to_major(subtotal + shipping + tax),
        "currency": currency(),
        "free_shipping_over": to_major(FREE_SHIPPING_OVER_CENTS),
        "free_shipping_remaining": to_major(
            max(0, FREE_SHIPPING_OVER_CENTS - subtotal)),
        "tax_rate": TAX_RATE,
        "n_items": sum(line["qty"] for line in cart_lines),
    }


def state(cart_id: str) -> dict[str, Any]:
    """Everything the basket UI needs, in one shape used by every cart endpoint."""
    cart_lines = lines(cart_id)
    return {"cart_id": cart_id, "lines": cart_lines, **totals(cart_lines)}
