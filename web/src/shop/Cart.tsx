/** The basket. Every number on this page came from the server. */

import { href, navigate } from "../router";
import { useShop } from "../state/shop";
import type { Cart as CartType } from "../types";
import { Photo, money } from "./bits";

export function Cart() {
  const { cart, setQty, removeLine, busy, user } = useShop();

  if (!cart) return <div className="spin">Loading your basket…</div>;

  if (cart.lines.length === 0) {
    return (
      <div className="wrap narrow">
        <div className="page-h"><h1>Your basket</h1></div>
        <div className="empty">
          <p>There is nothing in your basket yet.</p>
          <a className="btn" href={href({ name: "home" })}>Start shopping</a>
        </div>
      </div>
    );
  }

  return (
    <div className="wrap cart-grid">
      <div>
        <div className="page-h">
          <h1>Your basket</h1>
          <p className="note">
            {cart.n_items} item{cart.n_items === 1 ? "" : "s"}
          </p>
        </div>

        {cart.lines.map((line) => (
          <div className="cl" key={line.variant_sku}>
            <a className="cl-img" href={href({ name: "product", slug: line.slug })}>
              <Photo src={line.image} alt={line.name} />
            </a>
            <div className="cl-b">
              <a className="cl-n" href={href({ name: "product", slug: line.slug })}>
                {line.name}
              </a>
              <div className="cl-v">
                {line.brand}
                {line.variant_label ? ` · ${line.variant_label}` : ""}
              </div>
              <div className="cl-r">
                <div className="qty">
                  <button onClick={() => setQty(line.variant_sku, line.qty - 1)}
                          disabled={busy} aria-label="Decrease quantity">−</button>
                  <span>{line.qty}</span>
                  <button onClick={() => setQty(line.variant_sku, line.qty + 1)}
                          disabled={busy || line.qty >= Math.min(10, line.stock)}
                          aria-label="Increase quantity">+</button>
                </div>
                <button className="cl-x" disabled={busy}
                        onClick={() => removeLine(line.variant_sku)}>
                  Remove
                </button>
              </div>
            </div>
            <div className="cl-p">
              {money(line.line_total, cart.currency)}
              {line.qty > 1 ? (
                <div className="note" style={{ fontWeight: 400, marginTop: 4 }}>
                  {money(line.unit_price, cart.currency)} each
                </div>
              ) : null}
            </div>
          </div>
        ))}
      </div>

      <Summary cart={cart}>
        <button className="btn wide" style={{ marginTop: 16 }} disabled={busy}
                onClick={() => navigate(
                  user ? { name: "checkout" } : { name: "signin", next: "checkout" })}>
          {user ? "Checkout" : "Sign in to check out"}
        </button>
      </Summary>
    </div>
  );
}

/** The totals box, shared by the basket and the checkout so they cannot differ. */
export function Summary(
  { cart, children }: { cart: CartType; children?: React.ReactNode },
) {
  const toFree = cart.free_shipping_remaining;
  return (
    <aside className="summary">
      <h3>Order summary</h3>
      <div className="sum-r">
        <span>Subtotal</span><span>{money(cart.subtotal, cart.currency)}</span>
      </div>
      <div className="sum-r">
        <span>Delivery</span>
        <span>{cart.shipping === 0 ? "Free" : money(cart.shipping, cart.currency)}</span>
      </div>
      <div className="sum-r">
        <span>Tax ({Math.round(cart.tax_rate * 1000) / 10}%)</span>
        <span>{money(cart.tax, cart.currency)}</span>
      </div>
      <div className="sum-r total">
        <span>Total</span><span>{money(cart.total, cart.currency)}</span>
      </div>

      {toFree > 0 ? (
        <>
          <div className="ship-bar">
            <i style={{
              width: `${Math.min(100, (100 * cart.subtotal) / cart.free_shipping_over)}%`,
            }} />
          </div>
          <p className="note" style={{ fontSize: 12.5 }}>
            {money(toFree, cart.currency)} more for free delivery.
          </p>
        </>
      ) : cart.subtotal > 0 ? (
        <p className="note" style={{ fontSize: 12.5, marginTop: 10 }}>
          Your order qualifies for free delivery.
        </p>
      ) : null}

      {children}
    </aside>
  );
}
