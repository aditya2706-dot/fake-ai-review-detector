/** Sign in, register, and the account page. */

import { type FormEvent, useEffect, useState } from "react";
import { href, navigate } from "../router";
import { useShop } from "../state/shop";
import { OrderCard } from "./Orders";
import { api } from "../api";
import type { Order } from "../types";

export function SignIn({ next }: { next?: string }) {
  const { signIn, signUp, busy, error, clearError, user, authProvider } = useShop();
  const [mode, setMode] = useState<"in" | "up">("in");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [name, setName] = useState("");

  // Signing in is always on the way to somewhere -- the basket, the checkout,
  // or the account page. Bouncing on success rather than showing a "you are
  // signed in" dead end is the difference between a flow and a form.
  useEffect(() => {
    if (!user) return;
    navigate(next === "checkout" ? { name: "checkout" } : { name: "account" });
  }, [user, next]);

  useEffect(() => { clearError(); }, [mode, clearError]);

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    try {
      if (mode === "in") await signIn(email, password);
      else await signUp(email, password, name);
    } catch {
      /* the provider already surfaced it */
    }
  };

  return (
    <div className="wrap">
      <div className="auth-card">
        <h1>{mode === "in" ? "Sign in" : "Create an account"}</h1>
        <p className="note">
          {next === "checkout"
            ? "You need an account to place an order. Your basket is waiting and will follow you in."
            : "An account keeps your basket and your order history."}
        </p>

        {error ? <div className="err" style={{ marginBottom: 16 }}>{error}</div> : null}

        <form onSubmit={submit}>
          <div className="fgrid">
            {mode === "up" ? (
              <div>
                <label className="lab" htmlFor="au-name">Name</label>
                <input id="au-name" className="field" value={name} autoComplete="name"
                       onChange={(e) => setName(e.target.value)} />
              </div>
            ) : null}
            <div>
              <label className="lab" htmlFor="au-email">Email</label>
              <input id="au-email" className="field" type="email" required value={email}
                     autoComplete="email"
                     onChange={(e) => setEmail(e.target.value)} />
            </div>
            <div>
              <label className="lab" htmlFor="au-pass">Password</label>
              <input id="au-pass" className="field" type="password" required
                     value={password} minLength={mode === "up" ? 8 : 1}
                     autoComplete={mode === "up" ? "new-password" : "current-password"}
                     onChange={(e) => setPassword(e.target.value)} />
              {mode === "up" ? (
                <p className="note" style={{ fontSize: 12, marginTop: 6 }}>
                  At least 8 characters. Stored hashed with scrypt; it is never
                  sent back to the browser.
                </p>
              ) : null}
            </div>
          </div>

          <button className="btn wide" type="submit" disabled={busy}>
            {busy ? "One moment…" : mode === "in" ? "Sign in" : "Create account"}
          </button>
        </form>

        <div className="auth-alt">
          {mode === "in" ? (
            <>New here? <button onClick={() => setMode("up")}>Create an account</button></>
          ) : (
            <>Already have one? <button onClick={() => setMode("in")}>Sign in</button></>
          )}
        </div>

        <p className="note" style={{ marginTop: 26, fontSize: 12 }}>
          This is a demonstration shop. Use a throwaway address and a password
          you do not use anywhere else. Identity provider: {authProvider}.
        </p>
      </div>
    </div>
  );
}

export function Account() {
  const { user, signOut } = useShop();
  const [recent, setRecent] = useState<Order[] | null>(null);

  useEffect(() => {
    if (!user) { navigate({ name: "signin" }); return; }
    let live = true;
    api.orders()
      .then((h) => { if (live) setRecent(h.orders.slice(0, 3)); })
      .catch(() => { if (live) setRecent([]); });
    return () => { live = false; };
  }, [user]);

  if (!user) return <div className="spin">Redirecting…</div>;

  return (
    <div className="wrap narrow">
      <div className="page-h">
        <h1>Your account</h1>
        <p className="note">{user.email}</p>
      </div>

      <div className="fset">
        <h3>Details</h3>
        <dl className="specs">
          <div><dt>Name</dt><dd>{user.name || "—"}</dd></div>
          <div><dt>Email</dt><dd>{user.email}</dd></div>
          <div><dt>Signed in with</dt><dd>{user.provider}</dd></div>
          <div>
            <dt>Member since</dt>
            <dd>{new Date(user.created_at).toLocaleDateString()}</dd>
          </div>
        </dl>
      </div>

      <div className="fset">
        <h3>Recent orders</h3>
        {recent === null ? <p className="note">Loading…</p>
          : recent.length === 0 ? (
            <p className="note">No orders yet.</p>
          ) : (
            <div className="orders">
              {recent.map((o) => <OrderCard key={o.id} order={o} />)}
            </div>
          )}
        <p style={{ marginTop: 14 }}>
          <a className="btn ghost sm" href={href({ name: "orders" })}>All orders</a>
        </p>
      </div>

      <button className="btn ghost" onClick={() => signOut().then(
        () => navigate({ name: "home" }))}>
        Sign out
      </button>
    </div>
  );
}
