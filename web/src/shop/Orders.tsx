/** Order confirmation, and the order history behind it. */

import { useEffect, useState } from "react";
import { api, describe } from "../api";
import { href } from "../router";
import type { Order } from "../types";
import { Photo, money, when } from "./bits";

export function OrderPage({ id }: { id: string }) {
  const [order, setOrder] = useState<Order | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let live = true;
    api.order(id)
      .then((o) => { if (live) setOrder(o); })
      .catch((e) => { if (live) setError(describe(e)); });
    return () => { live = false; };
  }, [id]);

  if (error) return <div className="wrap narrow"><div className="err">{error}</div></div>;
  if (!order) return <div className="spin">Loading…</div>;

  return (
    <div className="wrap narrow">
      <div className="ok" style={{ marginBottom: 22 }}>
        Order placed. A confirmation would normally be sent to {order.email} —
        this shop does not send email.
      </div>
      <div className="page-h">
        <h1>Thank you</h1>
        <p className="note">
          Order <span className="mono">{order.id}</span>, placed{" "}
          {when(order.created_at)}.
        </p>
      </div>
      <OrderCard order={order} open />
      <p style={{ marginTop: 24 }}>
        <a className="btn ghost" href={href({ name: "home" })}>Continue shopping</a>
      </p>
    </div>
  );
}

export function Orders() {
  const [orders, setOrders] = useState<Order[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let live = true;
    api.orders()
      .then((h) => { if (live) setOrders(h.orders); })
      .catch((e) => { if (live) setError(describe(e)); });
    return () => { live = false; };
  }, []);

  return (
    <div className="wrap narrow">
      <div className="page-h"><h1>Your orders</h1></div>
      {error ? <div className="err">{error}</div>
        : orders === null ? <div className="spin">Loading…</div>
        : orders.length === 0 ? (
          <div className="empty">
            <p>You have not placed an order yet.</p>
            <a className="btn" href={href({ name: "home" })}>Start shopping</a>
          </div>
        ) : (
          <div className="orders">
            {orders.map((o) => <OrderCard key={o.id} order={o} />)}
          </div>
        )}
    </div>
  );
}

export function OrderCard({ order, open = false }: { order: Order; open?: boolean }) {
  return (
    <div className="order">
      <div className="order-h">
        <div>
          <div className="id">{order.id}</div>
          <div className="meta">
            {when(order.created_at)} · {order.n_items} item
            {order.n_items === 1 ? "" : "s"} · {order.status}
          </div>
        </div>
        <div>
          <div className="id">{money(order.total, order.currency)}</div>
          <div className="meta">
            {order.address.city}, {order.address.country}
          </div>
        </div>
      </div>
      <div className="order-b">
        {(open ? order.items : order.items.slice(0, 3)).map((i) => (
          <div className="oi" key={i.variant_sku}>
            <Photo src={i.image} alt={i.name} />
            <div className="n">
              <a href={href({ name: "product", slug: i.slug })}>{i.name}</a>
              {i.variant_label ? <div className="q">{i.variant_label}</div> : null}
            </div>
            <div className="q">× {i.qty}</div>
            <div className="mono">{money(i.line_total, order.currency)}</div>
          </div>
        ))}
        {!open && order.items.length > 3 ? (
          <p className="note">and {order.items.length - 3} more</p>
        ) : null}
      </div>
    </div>
  );
}
