"""The shop's transactional surface: accounts, baskets and orders.

WHY COOKIES AND NOT A BEARER TOKEN
----------------------------------
The session token is set as an httpOnly cookie, so the storefront's JavaScript
cannot read it and an XSS bug in the React app cannot exfiltrate a session. The
cost is that every request must carry credentials, which is why the client
sends `credentials: "include"` and why CORS here is configured with
`allow_credentials=True` against an explicit origin list rather than `*` --
the two are not allowed to coexist, and that restriction is the browser doing
the right thing.

`SameSite=Lax` is the CSRF defence: it stops a third-party page from making the
browser POST to /cart or /orders with this cookie attached, while still letting
an ordinary top-level navigation into the shop stay signed in.

WHY THE CART COOKIE IS SEPARATE FROM THE SESSION
------------------------------------------------
A basket exists before an account does. The cart id is its own cookie so an
anonymous shopper can fill one, and `carts.claim` merges it into the user's
basket at sign-in rather than throwing it away. That is also why the cart
cookie is NOT httpOnly-critical: it identifies a basket, not a person, and
leaking it costs nothing a shopper cannot already see.
"""

from __future__ import annotations

import os
from typing import Any

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, Field, field_validator

from ..store import carts, orders
from ..store.auth import (
    COOKIE_NAME, SESSION_DAYS, AuthError, FirebaseProvider, User, issue_session,
    provider, revoke_session, user_for_session,
)

auth_router = APIRouter(prefix="/auth", tags=["account"])
cart_router = APIRouter(prefix="/cart", tags=["cart"])
order_router = APIRouter(prefix="/orders", tags=["orders"])

CART_COOKIE = "frd_cart"
# Secure cookies require https, which a laptop running `make up` does not have.
# Opt in explicitly when deploying behind TLS rather than sniffing the request,
# because an X-Forwarded-Proto header is attacker-controlled unless a proxy is
# known to be stripping it.
SECURE_COOKIES = os.environ.get(
    "FRD_SECURE_COOKIES", "").lower() in {"1", "true", "yes"}


def _set_cookie(response: Response, key: str, value: str, *, days: int,
                http_only: bool = True) -> None:
    response.set_cookie(
        key, value, max_age=days * 24 * 3600, httponly=http_only,
        samesite="lax", secure=SECURE_COOKIES, path="/",
    )


def current_user(request: Request) -> User | None:
    return user_for_session(request.cookies.get(COOKIE_NAME))


def require_user(request: Request) -> User:
    user = current_user(request)
    if user is None:
        raise HTTPException(status_code=401, detail="Sign in to continue.")
    return user


def _cart_id(request: Request, user: User | None) -> str:
    return carts.ensure(request.cookies.get(CART_COOKIE), user.id if user else None)


def _cart_response(response: Response, cart_id: str) -> dict[str, Any]:
    _set_cookie(response, CART_COOKIE, cart_id, days=90, http_only=False)
    return carts.state(cart_id)


# --- accounts --------------------------------------------------------------

def _valid_email(v: str) -> str:
    """Enough validation to catch a typo, not enough to reject a real address.

    `pydantic.EmailStr` would be stricter and would add the `email-validator`
    dependency to a requirements file short enough that someone reads it. RFC
    5322 is also genuinely not worth implementing to reject `sam@@example` --
    the deliverability test a real shop needs is sending the confirmation
    email, which this project does not do.
    """
    v = str(v).strip()
    local, _, domain = v.rpartition("@")
    if not local or "." not in domain or domain.startswith(".") \
            or domain.endswith(".") or any(c.isspace() for c in v):
        raise ValueError("That does not look like an email address.")
    return v.lower()


class Credentials(BaseModel):
    email: str = Field(..., max_length=254)

    _norm_email = field_validator("email")(_valid_email)
    password: str = Field(..., min_length=1, max_length=256)
    name: str = Field("", max_length=120)


class FirebaseToken(BaseModel):
    id_token: str = Field(..., min_length=1)


def _sign_in(response: Response, request: Request, user: User) -> dict[str, Any]:
    """Issue a session and move whatever was in the anonymous basket across."""
    token = issue_session(user.id)
    _set_cookie(response, COOKIE_NAME, token, days=SESSION_DAYS)
    cart_id = carts.claim(carts.ensure(request.cookies.get(CART_COOKIE)), user.id)
    _set_cookie(response, CART_COOKIE, cart_id, days=90, http_only=False)
    return {"user": user.to_dict(), "cart_id": cart_id}


@auth_router.post("/register")
def register(body: Credentials, request: Request, response: Response) -> dict[str, Any]:
    try:
        user = provider().register(str(body.email), body.password, body.name)
    except AuthError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return _sign_in(response, request, user)


@auth_router.post("/login")
def login(body: Credentials, request: Request, response: Response) -> dict[str, Any]:
    try:
        user = provider().login(str(body.email), body.password)
    except AuthError as e:
        # 401, not 400: the credentials were well-formed and simply wrong.
        raise HTTPException(status_code=401, detail=str(e)) from e
    return _sign_in(response, request, user)


@auth_router.post("/firebase")
def firebase_login(body: FirebaseToken, request: Request,
                   response: Response) -> dict[str, Any]:
    """Exchange a Firebase ID token for a session on this API.

    Only reachable when FRD_AUTH=firebase. See `frd.store.auth.FirebaseProvider`
    -- that path is written but has never been run against a real project.
    """
    prov = provider()
    if not isinstance(prov, FirebaseProvider):
        raise HTTPException(
            status_code=404,
            detail="This deployment is not configured for Firebase (FRD_AUTH=local).")
    try:
        user = prov.verify_id_token(body.id_token)
    except AuthError as e:
        raise HTTPException(status_code=401, detail=str(e)) from e
    return _sign_in(response, request, user)


@auth_router.post("/logout")
def logout(request: Request, response: Response) -> dict[str, bool]:
    token = request.cookies.get(COOKIE_NAME)
    if token:
        revoke_session(token)
    response.delete_cookie(COOKIE_NAME, path="/")
    return {"ok": True}


@auth_router.get("/me")
def me(request: Request) -> dict[str, Any]:
    """Who is signed in, or nobody.

    Never a 401: the header calls this on every page load, and an error status
    for the ordinary signed-out case fills the console with noise.
    """
    user = current_user(request)
    return {"user": user.to_dict() if user else None,
            "provider": provider().name}


# --- basket ----------------------------------------------------------------

class CartAdd(BaseModel):
    slug: str
    variant_sku: str
    qty: int = Field(1, ge=1, le=carts.MAX_QTY_PER_LINE)


class CartQty(BaseModel):
    variant_sku: str
    qty: int = Field(..., ge=0, le=carts.MAX_QTY_PER_LINE)


@cart_router.get("")
def read_cart(request: Request, response: Response) -> dict[str, Any]:
    return _cart_response(response, _cart_id(request, current_user(request)))


@cart_router.post("/items")
def add_item(body: CartAdd, request: Request, response: Response) -> dict[str, Any]:
    cart_id = _cart_id(request, current_user(request))
    try:
        carts.add(cart_id, body.slug, body.variant_sku, body.qty)
    except carts.CartError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return _cart_response(response, cart_id)


@cart_router.patch("/items")
def update_item(body: CartQty, request: Request, response: Response) -> dict[str, Any]:
    cart_id = _cart_id(request, current_user(request))
    carts.set_qty(cart_id, body.variant_sku, body.qty)
    return _cart_response(response, cart_id)


@cart_router.delete("/items/{variant_sku}")
def delete_item(variant_sku: str, request: Request,
                response: Response) -> dict[str, Any]:
    cart_id = _cart_id(request, current_user(request))
    carts.remove(cart_id, variant_sku)
    return _cart_response(response, cart_id)


@cart_router.delete("")
def empty_cart(request: Request, response: Response) -> dict[str, Any]:
    cart_id = _cart_id(request, current_user(request))
    carts.clear(cart_id)
    return _cart_response(response, cart_id)


# --- orders ----------------------------------------------------------------

class Address(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    line1: str = Field(..., min_length=1, max_length=200)
    line2: str = Field("", max_length=200)
    city: str = Field(..., min_length=1, max_length=120)
    postcode: str = Field(..., min_length=1, max_length=32)
    country: str = Field(..., min_length=1, max_length=80)


class Checkout(BaseModel):
    email: str = Field(..., max_length=254)
    address: Address

    _norm_email = field_validator("email")(_valid_email)


@order_router.post("")
def checkout(body: Checkout, request: Request, response: Response) -> dict[str, Any]:
    user = require_user(request)
    cart_id = _cart_id(request, user)
    try:
        order = orders.place(user.id, cart_id, str(body.email),
                             body.address.model_dump())
    except orders.OrderError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    _set_cookie(response, CART_COOKIE, cart_id, days=90, http_only=False)
    return order


@order_router.get("")
def order_history(request: Request, limit: int = 20, offset: int = 0) -> dict[str, Any]:
    user = require_user(request)
    return orders.history(user.id, limit=min(limit, 50), offset=max(offset, 0))


@order_router.get("/{order_id}")
def order_detail(order_id: str, request: Request) -> dict[str, Any]:
    user = require_user(request)
    try:
        return orders.get(user.id, order_id)
    except orders.OrderError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
