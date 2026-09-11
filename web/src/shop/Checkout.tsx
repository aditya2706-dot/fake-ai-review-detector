/**
 * Checkout.
 *
 * There is no card field, and that is deliberate rather than unfinished. This
 * project has no payment processor and no business receiving card details, and
 * a form that looked like it took them would be training people to type a card
 * number into a demo. The address is real input because the order record needs
 * somewhere to deliver to; the payment step is stated plainly as simulated.
 */

import { type FormEvent, useState } from "react";
import { describe } from "../api";
import { api } from "../api";
import { navigate } from "../router";
import { useShop } from "../state/shop";
import { Summary } from "./Cart";

const BLANK = { name: "", line1: "", line2: "", city: "", postcode: "", country: "" };

export function Checkout() {
  const { cart, user, refreshCart } = useShop();
  const [address, setAddress] = useState({ ...BLANK, name: user?.name ?? "" });
  const [email, setEmail] = useState(user?.email ?? "");
  const [error, setError] = useState<string | null>(null);
  const [placing, setPlacing] = useState(false);

  if (!cart) return <div className="spin">Loading…</div>;
  if (cart.lines.length === 0) {
    return (
      <div className="wrap narrow">
        <div className="empty">
          <p>Your basket is empty.</p>
          <button className="btn" onClick={() => navigate({ name: "home" })}>
            Back to the shop
          </button>
        </div>
      </div>
    );
  }

  const set = (k: keyof typeof BLANK) => (e: { target: { value: string } }) =>
    setAddress({ ...address, [k]: e.target.value });

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    setPlacing(true);
    setError(null);
    try {
      const order = await api.checkout(email, address);
      await refreshCart();
      navigate({ name: "order", id: order.id });
    } catch (err) {
      setError(describe(err));
      setPlacing(false);
    }
  };

  return (
    <form className="wrap checkout" onSubmit={submit}>
      <div>
        <div className="page-h">
          <h1>Checkout</h1>
          <p className="note">
            No payment is taken and nothing ships — this is a demonstration shop.
          </p>
        </div>

        {error ? <div className="err" style={{ marginBottom: 18 }}>{error}</div> : null}

        <div className="fset">
          <h3>Contact</h3>
          <div className="fgrid">
            <div>
              <label className="lab" htmlFor="co-email">Email for the receipt</label>
              <input id="co-email" className="field" type="email" required value={email}
                     onChange={(e) => setEmail(e.target.value)} autoComplete="email" />
            </div>
          </div>
        </div>

        <div className="fset">
          <h3>Delivery address</h3>
          <div className="fgrid">
            <div>
              <label className="lab" htmlFor="co-name">Full name</label>
              <input id="co-name" className="field" required value={address.name}
                     onChange={set("name")} autoComplete="name" />
            </div>
            <div>
              <label className="lab" htmlFor="co-l1">Address line 1</label>
              <input id="co-l1" className="field" required value={address.line1}
                     onChange={set("line1")} autoComplete="address-line1" />
            </div>
            <div>
              <label className="lab" htmlFor="co-l2">Address line 2 (optional)</label>
              <input id="co-l2" className="field" value={address.line2}
                     onChange={set("line2")} autoComplete="address-line2" />
            </div>
            <div className="row2">
              <div>
                <label className="lab" htmlFor="co-city">City</label>
                <input id="co-city" className="field" required value={address.city}
                       onChange={set("city")} autoComplete="address-level2" />
              </div>
              <div>
                <label className="lab" htmlFor="co-pc">Postcode</label>
                <input id="co-pc" className="field" required value={address.postcode}
                       onChange={set("postcode")} autoComplete="postal-code" />
              </div>
            </div>
            <div>
              <label className="lab" htmlFor="co-country">Country</label>
              <input id="co-country" className="field" required value={address.country}
                     onChange={set("country")} autoComplete="country-name" />
            </div>
          </div>
        </div>

        <div className="fset">
          <h3>Payment</h3>
          <p className="note">
            This shop deliberately does not ask for card details. Placing the
            order records it against your account so you can see the order
            history working; no money moves.
          </p>
        </div>
      </div>

      <Summary cart={cart}>
        <button className="btn wide" type="submit" style={{ marginTop: 16 }}
                disabled={placing}>
          {placing ? "Placing order…" : "Place order"}
        </button>
      </Summary>
    </form>
  );
}
