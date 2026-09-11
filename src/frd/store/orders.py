"""Placing and reading orders.

WHY AN ORDER COPIES THE PRODUCT INSTEAD OF POINTING AT IT
---------------------------------------------------------
Each order line stores the name, the variant label, the unit price and the
image as they were at checkout. Joining back to the catalogue would be less
data and would be wrong: catalogues change. A listing gets renamed, repriced,
or withdrawn, and an order that renders today's catalogue would quietly restate
what someone bought and what they paid. An order is a record of something that
happened; it has to keep saying what it said on the day.

The slug is kept alongside, so the UI can still link to the product when it is
still there -- but nothing on the order page depends on that link resolving.

WHAT "PAYMENT" MEANS HERE
-------------------------
Nothing. There is no payment processor, no card is taken, and `place()` will
happily confirm any basket. That is deliberate for a demo, and it is the reason
the checkout form asks for an address but never for card details: a form that
looked like it took a card number would be training people to type one into a
project that has no business receiving it.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from .carts import clear, lines, totals
from .db import connect, to_cents, to_major


class OrderError(Exception):
    """Something the shopper should be told: empty basket, missing address."""


REQUIRED_ADDRESS = ("name", "line1", "city", "postcode", "country")


def _now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def new_id() -> str:
    # Short, readable, and unambiguous when read aloud to support.
    return f"ORD-{uuid.uuid4().hex[:8].upper()}"


def place(user_id: str, cart_id: str, email: str,
          address: dict[str, str]) -> dict[str, Any]:
    """Turn a basket into an order and empty it.

    Totals are recomputed from the catalogue here rather than taken from
    whatever the checkout page last displayed, for the same reason the cart
    prices itself: the client is not a source of truth about money.
    """
    cart_lines = lines(cart_id)
    if not cart_lines:
        raise OrderError("Your basket is empty.")

    missing = [f for f in REQUIRED_ADDRESS if not str(address.get(f, "")).strip()]
    if missing:
        raise OrderError("Missing delivery details: " + ", ".join(missing))
    if "@" not in str(email):
        raise OrderError("A valid email address is required for the receipt.")

    sums = totals(cart_lines)
    oid = new_id()
    created = _now()

    with connect() as conn:
        conn.execute(
            "INSERT INTO orders (id, user_id, created_at, status, email, ship_name, "
            "ship_line1, ship_line2, ship_city, ship_postcode, ship_country, "
            "subtotal_cents, shipping_cents, tax_cents, total_cents, currency) "
            "VALUES (?, ?, ?, 'confirmed', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (oid, user_id, created, str(email).strip(),
             address["name"].strip(), address["line1"].strip(),
             str(address.get("line2", "")).strip(), address["city"].strip(),
             address["postcode"].strip(), address["country"].strip(),
             to_cents(sums["subtotal"]), to_cents(sums["shipping"]),
             to_cents(sums["tax"]), to_cents(sums["total"]), sums["currency"]),
        )
        conn.executemany(
            "INSERT INTO order_items (order_id, variant_sku, slug, name, "
            "variant_label, image, unit_cents, qty) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [(oid, ln["variant_sku"], ln["slug"], ln["name"], ln["variant_label"],
              ln["image"], to_cents(ln["unit_price"]), ln["qty"])
             for ln in cart_lines],
        )

    clear(cart_id)
    return get(user_id, oid)          # read it back, so one shape is ever returned


def _order_payload(conn, row) -> dict[str, Any]:
    items = conn.execute(
        "SELECT * FROM order_items WHERE order_id = ? ORDER BY rowid", (row["id"],)
    ).fetchall()
    return {
        "id": row["id"],
        "created_at": row["created_at"],
        "status": row["status"],
        "email": row["email"],
        "address": {
            "name": row["ship_name"], "line1": row["ship_line1"],
            "line2": row["ship_line2"], "city": row["ship_city"],
            "postcode": row["ship_postcode"], "country": row["ship_country"],
        },
        "subtotal": to_major(row["subtotal_cents"]),
        "shipping": to_major(row["shipping_cents"]),
        "tax": to_major(row["tax_cents"]),
        "total": to_major(row["total_cents"]),
        "currency": row["currency"],
        "n_items": sum(int(i["qty"]) for i in items),
        "items": [{
            "variant_sku": i["variant_sku"], "slug": i["slug"], "name": i["name"],
            "variant_label": i["variant_label"], "image": i["image"],
            "unit_price": to_major(i["unit_cents"]), "qty": int(i["qty"]),
            "line_total": to_major(i["unit_cents"] * int(i["qty"])),
        } for i in items],
    }


def get(user_id: str, order_id: str) -> dict[str, Any]:
    """One order. Scoped to the owner -- an order id is not an authorisation."""
    with connect() as conn:
        row = conn.execute("SELECT * FROM orders WHERE id = ? AND user_id = ?",
                           (order_id, user_id)).fetchone()
        if row is None:
            raise OrderError(f"No order {order_id}.")
        return _order_payload(conn, row)


def history(user_id: str, limit: int = 20, offset: int = 0) -> dict[str, Any]:
    with connect() as conn:
        total = conn.execute("SELECT count(*) c FROM orders WHERE user_id = ?",
                             (user_id,)).fetchone()["c"]
        rows = conn.execute(
            "SELECT * FROM orders WHERE user_id = ? ORDER BY created_at DESC "
            "LIMIT ? OFFSET ?", (user_id, limit, offset)).fetchall()
        return {"orders": [_order_payload(conn, r) for r in rows],
                "total": int(total), "limit": limit, "offset": offset}
