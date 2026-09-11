/**
 * Who is signed in and what is in the basket.
 *
 * One provider rather than two, because the two are not independent: signing
 * in merges the anonymous basket into the account's (`carts.claim` on the
 * server), so every auth transition has to refresh the cart. Splitting them
 * would mean one context reaching into the other, which is the same coupling
 * with more files.
 *
 * The server is the only source of truth for both. Nothing here computes a
 * total, decides whether a session is still valid, or optimistically edits a
 * line: each mutation posts and stores whatever comes back. That costs a round
 * trip per click and buys the guarantee that the price shown is the price the
 * server will charge.
 */

import {
  type ReactNode, createContext, useCallback, useContext, useEffect, useMemo,
  useState,
} from "react";
import { api, describe } from "../api";
import type { Cart, User } from "../types";

interface ShopState {
  user: User | null;
  authProvider: string;
  ready: boolean;
  cart: Cart | null;
  busy: boolean;
  error: string | null;
  clearError: () => void;
  signIn: (email: string, password: string) => Promise<void>;
  signUp: (email: string, password: string, name: string) => Promise<void>;
  signOut: () => Promise<void>;
  addToCart: (slug: string, variantSku: string, qty: number) => Promise<void>;
  setQty: (variantSku: string, qty: number) => Promise<void>;
  removeLine: (variantSku: string) => Promise<void>;
  refreshCart: () => Promise<void>;
}

const Ctx = createContext<ShopState | null>(null);

export function ShopProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [authProvider, setAuthProvider] = useState("local");
  const [cart, setCart] = useState<Cart | null>(null);
  const [ready, setReady] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let live = true;
    (async () => {
      // Both are anonymous-safe: /auth/me answers with a null user rather than
      // a 401, and /cart mints an empty basket. So a first-time visitor takes
      // exactly this path too, with no error branch of its own.
      const [me, basket] = await Promise.allSettled([api.me(), api.cart()]);
      if (!live) return;
      if (me.status === "fulfilled") {
        setUser(me.value.user);
        setAuthProvider(me.value.provider);
      }
      if (basket.status === "fulfilled") setCart(basket.value);
      setReady(true);
    })();
    return () => { live = false; };
  }, []);

  /** Run a mutation, surface its error, and never leave `busy` stuck on. */
  const run = useCallback(async <T,>(fn: () => Promise<T>): Promise<T> => {
    setBusy(true);
    setError(null);
    try {
      return await fn();
    } catch (e) {
      setError(describe(e));
      throw e;
    } finally {
      setBusy(false);
    }
  }, []);

  const refreshCart = useCallback(async () => {
    setCart(await api.cart());
  }, []);

  const afterAuth = useCallback(async (next: User) => {
    setUser(next);
    // The server merged the anonymous basket into this account's during
    // sign-in; re-reading is how this side learns what that produced.
    await refreshCart();
  }, [refreshCart]);

  const value = useMemo<ShopState>(() => ({
    user, authProvider, ready, cart, busy, error,
    clearError: () => setError(null),
    signIn: (email, password) =>
      run(() => api.login(email, password)).then((r) => afterAuth(r.user)),
    signUp: (email, password, name) =>
      run(() => api.register(email, password, name)).then((r) => afterAuth(r.user)),
    signOut: async () => {
      await run(() => api.logout());
      setUser(null);
      await refreshCart();
    },
    addToCart: (slug, variantSku, qty) =>
      run(() => api.cartAdd(slug, variantSku, qty)).then(setCart),
    setQty: (variantSku, qty) =>
      run(() => api.cartSetQty(variantSku, qty)).then(setCart),
    removeLine: (variantSku) =>
      run(() => api.cartRemove(variantSku)).then(setCart),
    refreshCart,
  }), [user, authProvider, ready, cart, busy, error, run, afterAuth, refreshCart]);

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useShop(): ShopState {
  const ctx = useContext(Ctx);
  if (!ctx) throw new Error("useShop outside ShopProvider");
  return ctx;
}
