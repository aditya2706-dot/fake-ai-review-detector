"""Accounts, baskets and orders.

WHAT IS WORTH TESTING HERE
--------------------------
This is the only part of the repo that holds mutable state belonging to a
person, so the things worth pinning down are the ones that lose money or leak
data rather than the ones that lose accuracy:

  * the server prices the basket, and a client that sends its own price is
    ignored;
  * money is integer arithmetic end to end, so a basket of awkward prices does
    not drift by a cent;
  * an anonymous basket survives signing in, because that is the single moment
    a shop most often silently eats an order;
  * an order id is not an authorisation -- one account cannot read another's;
  * a password is never stored, never returned, and a wrong one is rejected in
    a way that does not disclose whether the account exists.

These run against the real SQLite store, pointed at a temp file. Mocking the
database would test the mock.
"""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """A fresh shop with an empty database, per test."""
    monkeypatch.setenv("FRD_STORE_DB", str(tmp_path / "store.db"))
    monkeypatch.setenv("FRD_AUTH", "local")

    from frd.store import auth as store_auth
    from frd.store import db as store_db

    store_db._INITIALISED.clear()
    store_auth.reset_provider()

    from frd.api import main as api_main

    with TestClient(api_main.app) as c:
        yield c


def _register(client, email="sam@example.com", password="correct-horse"):
    return client.post("/auth/register",
                       json={"email": email, "password": password, "name": "Sam"})


def _first_variant(client):
    """A real listing and variant, taken from the catalogue rather than invented."""
    from frd.data.catalog import listings

    item = listings()[0]
    return item["slug"], item["variants"][0].sku, item["price"]


# --- accounts --------------------------------------------------------------

def test_register_then_me_then_logout(client):
    r = _register(client)
    assert r.status_code == 200
    assert r.json()["user"]["email"] == "sam@example.com"
    # A password must never come back out, in any form.
    assert "password" not in r.text and "scrypt" not in r.text

    me = client.get("/auth/me").json()
    assert me["user"]["email"] == "sam@example.com"
    assert me["provider"] == "local"

    assert client.post("/auth/logout").status_code == 200
    assert client.get("/auth/me").json()["user"] is None


def test_email_is_normalised_and_unique(client):
    assert _register(client, "Sam@Example.COM ").status_code == 200
    client.post("/auth/logout")
    # Same account, different capitalisation and whitespace.
    assert client.post("/auth/login", json={
        "email": "sam@example.com", "password": "correct-horse"}).status_code == 200
    dup = _register(client, "SAM@example.com")
    assert dup.status_code == 400
    assert "already exists" in dup.json()["detail"]


def test_wrong_password_and_unknown_account_look_the_same(client):
    _register(client)
    client.post("/auth/logout")
    wrong = client.post("/auth/login",
                        json={"email": "sam@example.com", "password": "nope!!!!"})
    missing = client.post("/auth/login",
                          json={"email": "nobody@example.com", "password": "nope!!!!"})
    assert wrong.status_code == missing.status_code == 401
    # Identical wording: a different message enumerates registered addresses.
    assert wrong.json()["detail"] == missing.json()["detail"]


def test_short_passwords_are_refused(client):
    r = client.post("/auth/register",
                    json={"email": "a@b.com", "password": "short", "name": ""})
    assert r.status_code == 400


def test_session_cookie_is_not_readable_by_javascript(client):
    """An XSS bug in the storefront must not be able to walk off with a session."""
    r = _register(client)
    cookie = next(c for c in r.headers.get_list("set-cookie") if "frd_session" in c)
    assert "httponly" in cookie.lower()
    assert "samesite=lax" in cookie.lower()


def test_firebase_endpoint_is_closed_when_not_configured(client):
    r = client.post("/auth/firebase", json={"id_token": "anything"})
    assert r.status_code == 404


# --- the basket ------------------------------------------------------------

def test_add_to_cart_prices_from_the_catalogue(client):
    slug, sku, price = _first_variant(client)
    cart = client.post("/cart/items",
                       json={"slug": slug, "variant_sku": sku, "qty": 2}).json()
    assert cart["n_items"] == 2
    line = cart["lines"][0]
    assert line["unit_price"] == pytest.approx(price)
    assert line["line_total"] == pytest.approx(price * 2)
    assert cart["subtotal"] == pytest.approx(price * 2)


def test_client_cannot_set_its_own_price(client):
    """The request body has no price field, and sending one changes nothing."""
    slug, sku, price = _first_variant(client)
    cart = client.post("/cart/items", json={
        "slug": slug, "variant_sku": sku, "qty": 1,
        "unit_price": 0.01, "line_total": 0.01, "subtotal": 0.01,
    }).json()
    assert cart["subtotal"] == pytest.approx(price)


def test_unknown_product_is_rejected(client):
    r = client.post("/cart/items",
                    json={"slug": "no-such-thing", "variant_sku": "X", "qty": 1})
    assert r.status_code == 400


def test_quantities_update_and_remove(client):
    slug, sku, _ = _first_variant(client)
    client.post("/cart/items", json={"slug": slug, "variant_sku": sku, "qty": 1})
    cart = client.patch("/cart/items", json={"variant_sku": sku, "qty": 3}).json()
    assert cart["n_items"] == 3
    # Zero means remove, not a zero-quantity line.
    cart = client.patch("/cart/items", json={"variant_sku": sku, "qty": 0}).json()
    assert cart["lines"] == [] and cart["subtotal"] == 0
    assert cart["shipping"] == 0, "an empty basket is not charged delivery"


def test_totals_are_exact_on_awkward_prices(client):
    """Float arithmetic here charges 0.30000000000000004. Integers do not."""
    from frd.store.carts import TAX_RATE, totals

    lines = [{"line_total": 0.10, "qty": 1}, {"line_total": 0.20, "qty": 1}]
    sums = totals(lines)
    assert sums["subtotal"] == 0.30
    assert sums["total"] == pytest.approx(round(0.30 * (1 + TAX_RATE), 2) + 6.95)


def test_shipping_is_free_over_the_threshold(client):
    from frd.store.carts import FREE_SHIPPING_OVER_CENTS, totals

    under = totals([{"line_total": 10.0, "qty": 1}])
    assert under["shipping"] > 0
    assert under["free_shipping_remaining"] > 0

    over = totals([{"line_total": FREE_SHIPPING_OVER_CENTS / 100, "qty": 1}])
    assert over["shipping"] == 0
    assert over["free_shipping_remaining"] == 0


def test_anonymous_basket_survives_signing_in(client):
    """The one moment a shop most often silently loses an order."""
    slug, sku, price = _first_variant(client)
    client.post("/cart/items", json={"slug": slug, "variant_sku": sku, "qty": 2})
    assert client.get("/cart").json()["n_items"] == 2

    _register(client)
    after = client.get("/cart").json()
    assert after["n_items"] == 2
    assert after["subtotal"] == pytest.approx(price * 2)


# --- orders ----------------------------------------------------------------

ADDRESS = {"name": "Sam Reed", "line1": "12 Mill Lane", "line2": "",
           "city": "Bristol", "postcode": "BS1 4QA", "country": "United Kingdom"}


def _checkout(client, email="sam@example.com"):
    return client.post("/orders", json={"email": email, "address": ADDRESS})


def test_checkout_requires_an_account(client):
    slug, sku, _ = _first_variant(client)
    client.post("/cart/items", json={"slug": slug, "variant_sku": sku, "qty": 1})
    assert _checkout(client).status_code == 401


def test_checkout_empties_the_basket_and_records_the_totals(client):
    slug, sku, _ = _first_variant(client)
    _register(client)
    client.post("/cart/items", json={"slug": slug, "variant_sku": sku, "qty": 2})
    cart = client.get("/cart").json()

    order = _checkout(client).json()
    assert order["id"].startswith("ORD-")
    assert order["n_items"] == 2
    assert order["total"] == pytest.approx(cart["total"])
    assert order["items"][0]["slug"] == slug
    assert order["address"]["city"] == "Bristol"

    assert client.get("/cart").json()["lines"] == [], "basket not emptied"
    assert client.get("/orders").json()["total"] == 1


def test_empty_basket_cannot_be_ordered(client):
    _register(client)
    r = _checkout(client)
    assert r.status_code == 400
    assert "empty" in r.json()["detail"].lower()


def test_an_order_id_is_not_an_authorisation(client):
    """Someone else's order must 404, not render."""
    slug, sku, _ = _first_variant(client)
    _register(client, "first@example.com")
    client.post("/cart/items", json={"slug": slug, "variant_sku": sku, "qty": 1})
    order_id = _checkout(client, "first@example.com").json()["id"]
    client.post("/auth/logout")

    _register(client, "second@example.com")
    assert client.get(f"/orders/{order_id}").status_code == 404
    assert client.get("/orders").json()["total"] == 0


def test_order_keeps_what_was_bought_even_if_the_catalogue_changes(client, monkeypatch):
    """An order is a record of a transaction, not a view of today's catalogue."""
    slug, sku, price = _first_variant(client)
    _register(client)
    client.post("/cart/items", json={"slug": slug, "variant_sku": sku, "qty": 1})
    order = _checkout(client).json()

    from frd.data import catalog

    # Rename and reprice the listing out from under the placed order, the way a
    # real catalogue changes between the purchase and the customer looking at
    # their order history.
    original = catalog.listing_for(slug)
    moved = dict(original, name="RENAMED", price=original["price"] + 100)
    monkeypatch.setattr(catalog, "listing_for",
                        lambda s: moved if s == slug else None)

    again = client.get(f"/orders/{order['id']}").json()
    assert again["items"][0]["name"] == order["items"][0]["name"] != "RENAMED"
    assert again["items"][0]["unit_price"] == pytest.approx(price)
    assert again["total"] == pytest.approx(order["total"])


def test_store_db_path_is_configurable(tmp_path, monkeypatch):
    """`make up` must not drop a database into the working directory by surprise."""
    monkeypatch.setenv("FRD_STORE_DB", str(tmp_path / "nested" / "store.db"))
    from frd.store import db

    db._INITIALISED.clear()
    with db.connect() as conn:
        conn.execute("SELECT 1")
    assert (tmp_path / "nested" / "store.db").exists()
    assert not os.path.exists("store.db")
