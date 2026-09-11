"""Accounts and sessions.

WHICH IDENTITY PROVIDER, AND WHY THE SEAM
-----------------------------------------
Two providers sit behind one protocol:

  * `local` (the default) keeps accounts in this repo's SQLite file and hashes
    passwords with `hashlib.scrypt`. No dependency, no network, no project to
    create -- `make up` works from a clean clone, which is the property the
    rest of the repo is built around and the one a hosted identity service
    takes away first.
  * `firebase` delegates to Firebase Authentication: the browser signs in with
    the Firebase SDK and posts the resulting ID token here, and this process
    verifies it with `firebase-admin` and keeps a local row for the order
    history to hang off.

Select with `FRD_AUTH=local|firebase`. The rest of the API asks for
`current_user` and never learns which one answered -- that is the point of the
seam. Orders and carts always key on the local `users.id`, so switching
providers changes how someone proves who they are, not what they own.

WHY SCRYPT AND NOT BCRYPT OR ARGON2
-----------------------------------
Both are better-known choices and neither is in the standard library, and this
repo's dependency list is a deliberate artefact -- it is short enough to read.
`hashlib.scrypt` is memory-hard, has been in CPython since 3.6, and at the
parameters below costs about 100ms per verification, which is the right order
of magnitude. A deployment that already runs argon2 should replace
`_hash`/`_verify` and nothing else.

WHY OPAQUE TOKENS AND NOT JWTS
------------------------------
A JWT here would need a signing library and would still need a server-side
table, because a stateless token cannot be revoked and "log out" has to mean
something. So the token is 32 random bytes, stored hashed, in an httpOnly
cookie: not readable from JavaScript, so an XSS bug in the storefront cannot
walk off with a session.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Protocol

from .db import connect

SESSION_DAYS = 30
COOKIE_NAME = "frd_session"

# scrypt work factors. n=2**14 with r=8 lands around 100ms and ~16MB per call
# on a laptop -- slow enough to make offline cracking expensive, fast enough
# that a login is not a visible pause.
_N, _R, _P, _DKLEN = 2**14, 8, 1, 32


class AuthError(Exception):
    """Anything a caller is allowed to see: bad credentials, taken email."""


@dataclass(frozen=True, slots=True)
class User:
    id: str
    email: str
    name: str
    provider: str
    created_at: str

    def to_dict(self) -> dict:
        return {"id": self.id, "email": self.email, "name": self.name,
                "provider": self.provider, "created_at": self.created_at}


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


def _hash(password: str, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    dk = hashlib.scrypt(password.encode("utf-8"), salt=salt,
                        n=_N, r=_R, p=_P, dklen=_DKLEN)
    return f"scrypt${_N}${_R}${_P}${salt.hex()}${dk.hex()}"


def _verify(password: str, stored: str) -> bool:
    try:
        kind, n, r, p, salt_hex, want = stored.split("$")
        if kind != "scrypt":
            return False
        dk = hashlib.scrypt(password.encode("utf-8"), salt=bytes.fromhex(salt_hex),
                            n=int(n), r=int(r), p=int(p), dklen=len(want) // 2)
    except (ValueError, TypeError):
        return False
    # Constant-time: a timing difference here leaks how much of a hash matched.
    return hmac.compare_digest(dk.hex(), want)


def _row_to_user(row: sqlite3.Row) -> User:
    return User(id=row["id"], email=row["email"], name=row["name"],
                provider=row["provider"], created_at=row["created_at"])


def normalise_email(email: str) -> str:
    return email.strip().lower()


def _insert_user(conn, email: str, name: str, password_hash: str,
                 provider: str, external_id: str | None) -> User:
    uid = f"U{uuid.uuid4().hex[:16]}"
    created = _now().isoformat()
    try:
        conn.execute(
            "INSERT INTO users (id, email, name, password_hash, provider, "
            "external_id, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (uid, email, name, password_hash, provider, external_id, created),
        )
    except sqlite3.IntegrityError as e:
        raise AuthError("An account with that email already exists.") from e
    return User(uid, email, name, provider, created)


# --- sessions --------------------------------------------------------------
#
# The cookie carries the raw token; the table stores its SHA-256. A read of the
# database therefore does not hand over live sessions, which is the same reason
# passwords are not stored either.

def _token_digest(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def issue_session(user_id: str) -> str:
    token = secrets.token_urlsafe(32)
    now = _now()
    with connect() as conn:
        conn.execute(
            "INSERT INTO sessions (token, user_id, created_at, expires_at) "
            "VALUES (?, ?, ?, ?)",
            (_token_digest(token), user_id, now.isoformat(),
             (now + timedelta(days=SESSION_DAYS)).isoformat()),
        )
    return token


def revoke_session(token: str) -> None:
    with connect() as conn:
        conn.execute("DELETE FROM sessions WHERE token = ?", (_token_digest(token),))


def user_for_session(token: str | None) -> User | None:
    if not token:
        return None
    with connect() as conn:
        row = conn.execute(
            "SELECT u.* FROM sessions s JOIN users u ON u.id = s.user_id "
            "WHERE s.token = ? AND s.expires_at > ?",
            (_token_digest(token), _now().isoformat()),
        ).fetchone()
    return _row_to_user(row) if row else None


def purge_expired_sessions() -> int:
    with connect() as conn:
        cur = conn.execute("DELETE FROM sessions WHERE expires_at <= ?",
                           (_now().isoformat(),))
    return cur.rowcount


# --- providers -------------------------------------------------------------

class AuthProvider(Protocol):
    name: str

    def register(self, email: str, password: str, name: str) -> User: ...
    def login(self, email: str, password: str) -> User: ...


class LocalProvider:
    """Email and password, hashed and stored here."""

    name = "local"

    MIN_PASSWORD = 8

    def register(self, email: str, password: str, name: str) -> User:
        email = normalise_email(email)
        if "@" not in email or "." not in email.split("@")[-1]:
            raise AuthError("That does not look like an email address.")
        if len(password) < self.MIN_PASSWORD:
            raise AuthError(
                f"Password must be at least {self.MIN_PASSWORD} characters.")
        with connect() as conn:
            return _insert_user(conn, email, name.strip(), _hash(password),
                                "local", None)

    def login(self, email: str, password: str) -> User:
        email = normalise_email(email)
        with connect() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE email = ? AND provider = 'local'",
                (email,),
            ).fetchone()
        # The same message either way, and the hash is computed even when the
        # account does not exist, so response time does not disclose which
        # emails are registered.
        dummy = "scrypt$16384$8$1$" + "00" * 16 + "$" + "00" * 32
        ok = _verify(password, row["password_hash"] if row else dummy)
        if not row or not ok:
            raise AuthError("Email or password is incorrect.")
        return _row_to_user(row)


class FirebaseProvider:
    """Firebase Authentication: the browser signs in, this verifies the ID token.

    UNVERIFIED. This path has never been run -- it needs a Firebase project and
    a service account, and neither existed when it was written. It is here so
    the seam is real rather than hypothetical, and so switching is a config
    change and not a rewrite. Treat it as a starting point and test it against
    a real project before relying on it.

    Set FRD_AUTH=firebase, FIREBASE_PROJECT_ID, and GOOGLE_APPLICATION_CREDENTIALS
    (a service-account JSON), then `pip install firebase-admin`.
    """

    name = "firebase"

    def __init__(self) -> None:
        try:
            import firebase_admin
            from firebase_admin import auth as fb_auth
        except ImportError as e:
            raise AuthError(
                "FRD_AUTH=firebase needs the firebase-admin package: "
                "pip install firebase-admin"
            ) from e
        if not firebase_admin._apps:
            firebase_admin.initialize_app()
        self._auth = fb_auth

    def verify_id_token(self, id_token: str) -> User:
        """Exchange a Firebase ID token for a local user row, creating it if new."""
        try:
            claims = self._auth.verify_id_token(id_token)
        except Exception as e:
            raise AuthError("That sign-in token is not valid.") from e

        uid = str(claims["uid"])
        email = normalise_email(str(claims.get("email") or f"{uid}@firebase.local"))
        name = str(claims.get("name") or "")
        with connect() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE provider = 'firebase' AND external_id = ?",
                (uid,),
            ).fetchone()
            if row:
                return _row_to_user(row)
            return _insert_user(conn, email, name, "", "firebase", uid)

    # Password flows belong to Firebase, not here.
    def register(self, email: str, password: str, name: str) -> User:
        raise AuthError(
            "This deployment uses Firebase; register through the Firebase SDK "
            "and post the ID token to /auth/firebase.")

    def login(self, email: str, password: str) -> User:
        raise AuthError(
            "This deployment uses Firebase; sign in through the Firebase SDK "
            "and post the ID token to /auth/firebase.")


_PROVIDER: AuthProvider | None = None


def provider() -> AuthProvider:
    global _PROVIDER
    if _PROVIDER is None:
        kind = os.environ.get("FRD_AUTH", "local").strip().lower()
        _PROVIDER = FirebaseProvider() if kind == "firebase" else LocalProvider()
    return _PROVIDER


def reset_provider() -> None:
    """Re-read FRD_AUTH. For tests that switch providers mid-process."""
    global _PROVIDER
    _PROVIDER = None
