"""SQLite storage for accounts, carts and orders.

WHY SQLITE AND NOT A SERVICE
----------------------------
The whole repo's operating promise is that `make up` works on a laptop from a
clean checkout with nothing else running. A Postgres container or a hosted
auth provider breaks that on the first clone, and this data -- a demo shop's
accounts and baskets -- has no property that needs more than a file. SQLite in
WAL mode handles the concurrency a single uvicorn worker produces with room to
spare.

WHY MONEY IS AN INTEGER
-----------------------
Every amount in this module is in minor units (cents), never a float. A cart
that sums 0.1 + 0.2 in binary floating point and charges 0.30000000000000004
is a real class of bug, and totals are recomputed on the server for every
request rather than trusted from the client, so the arithmetic happens here a
lot. `frd.data.catalog` stores display prices as floats because that is what
the catalogue file is; `to_cents` is the one place the two meet.

WHY THE SCHEMA IS CREATED ON CONNECT
------------------------------------
There is no migration tool and there should not be one for four tables. The
schema is idempotent DDL run at startup; changing a column means deleting
`data/store.db`, which is the right cost for a demo shop and the wrong cost for
anything holding real orders. That trade is stated here rather than discovered.
"""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from ..config import DATA_DIR

SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS users (
    id            TEXT PRIMARY KEY,
    email         TEXT NOT NULL UNIQUE,
    name          TEXT NOT NULL DEFAULT '',
    password_hash TEXT NOT NULL DEFAULT '',
    provider      TEXT NOT NULL DEFAULT 'local',
    external_id   TEXT,
    created_at    TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS users_external
    ON users(provider, external_id) WHERE external_id IS NOT NULL;

-- Opaque session tokens rather than JWTs. A JWT would need a signing library
-- and would still need this table to support logout, because the one thing a
-- stateless token cannot do is stop being valid.
CREATE TABLE IF NOT EXISTS sessions (
    token      TEXT PRIMARY KEY,
    user_id    TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS sessions_user ON sessions(user_id);

-- A cart belongs to a user, or to nobody. The anonymous case is the whole
-- reason `id` is not just the user id: a shopper fills a basket, then signs in
-- at checkout, and the basket has to survive that. See carts.claim().
CREATE TABLE IF NOT EXISTS carts (
    id         TEXT PRIMARY KEY,
    user_id    TEXT REFERENCES users(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS carts_user ON carts(user_id);

CREATE TABLE IF NOT EXISTS cart_items (
    cart_id     TEXT NOT NULL REFERENCES carts(id) ON DELETE CASCADE,
    variant_sku TEXT NOT NULL,
    slug        TEXT NOT NULL,
    qty         INTEGER NOT NULL CHECK (qty > 0),
    added_at    TEXT NOT NULL,
    PRIMARY KEY (cart_id, variant_sku)
);

-- Orders copy the name, price and image of what was bought instead of joining
-- back to the catalogue. A catalogue is a live thing: prices change, listings
-- get renamed and withdrawn. An order is a record of a transaction that
-- happened, and it has to keep saying what it said on the day.
CREATE TABLE IF NOT EXISTS orders (
    id           TEXT PRIMARY KEY,
    user_id      TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at   TEXT NOT NULL,
    status       TEXT NOT NULL DEFAULT 'confirmed',
    email        TEXT NOT NULL,
    ship_name    TEXT NOT NULL,
    ship_line1   TEXT NOT NULL,
    ship_line2   TEXT NOT NULL DEFAULT '',
    ship_city    TEXT NOT NULL,
    ship_postcode TEXT NOT NULL,
    ship_country TEXT NOT NULL,
    subtotal_cents INTEGER NOT NULL,
    shipping_cents INTEGER NOT NULL,
    tax_cents      INTEGER NOT NULL,
    total_cents    INTEGER NOT NULL,
    currency     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS orders_user ON orders(user_id, created_at DESC);

CREATE TABLE IF NOT EXISTS order_items (
    order_id     TEXT NOT NULL REFERENCES orders(id) ON DELETE CASCADE,
    variant_sku  TEXT NOT NULL,
    slug         TEXT NOT NULL,
    name         TEXT NOT NULL,
    variant_label TEXT NOT NULL DEFAULT '',
    image        TEXT NOT NULL DEFAULT '',
    unit_cents   INTEGER NOT NULL,
    qty          INTEGER NOT NULL,
    PRIMARY KEY (order_id, variant_sku)
);
"""


def db_path() -> Path:
    override = os.environ.get("FRD_STORE_DB")
    return Path(override) if override else DATA_DIR / "store.db"


def to_cents(amount: float) -> int:
    """Dollars (as the catalogue stores them) to minor units, half-up."""
    return round(float(amount) * 100)


def to_major(cents: int) -> float:
    """Minor units back to a display amount. The only place this is done."""
    return round(int(cents) / 100, 2)


_INITIALISED: set[str] = set()


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    """A connection per call, with the schema guaranteed and a transaction around it.

    One connection per request rather than a shared one: FastAPI runs sync
    endpoints in a worker thread, and a sqlite3 connection is not safe to share
    across threads. Opening a file-backed connection is microseconds; the
    schema DDL runs once per process, guarded by `_INITIALISED`.
    """
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10.0)
    conn.row_factory = sqlite3.Row
    try:
        key = str(path)
        if key not in _INITIALISED:
            conn.executescript(SCHEMA)
            conn.commit()
            _INITIALISED.add(key)
        else:
            conn.execute("PRAGMA foreign_keys = ON")
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def reset() -> None:
    """Drop everything. Used by the tests, and by nothing else."""
    with connect() as conn:
        for table in ("order_items", "orders", "cart_items", "carts",
                      "sessions", "users"):
            conn.execute(f"DELETE FROM {table}")
