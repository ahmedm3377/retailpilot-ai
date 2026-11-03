import { useEffect, useRef, useState } from "react";
import { formatUsd } from "../api/client";
import type { Order, OrderLine } from "../api/types";

const lineKey = (orderId: string, line: OrderLine) => `${orderId}/${line.line_id}`;

function returnedKeys(orders: Order[]): Set<string> {
  return new Set(orders.flatMap((o) => o.lines.filter((l) => l.returned).map((l) => lineKey(o.order_id, l))));
}

function outcomeText(line: OrderLine): string {
  if (line.outcome === "refund" && line.refund_cents !== null) return `Refunded ${formatUsd(line.refund_cents)}`;
  if (line.outcome === "exchange") return `Exchanged for ${line.replacement_sku}`;
  return line.returned ? "Returned" : "Not returned";
}

// Shows what the server last reported. Collapsed at first so the customer starts in the chat; it opens by
// itself when a change to an item commits, so that the customer sees the result.
export function OrderSummary({ orders }: { orders: Order[] }) {
  const [open, setOpen] = useState(false);
  const [updated, setUpdated] = useState<Set<string>>(new Set());
  const known = useRef(returnedKeys(orders));

  useEffect(() => {
    const now = returnedKeys(orders);
    const changed = [...now].filter((key) => !known.current.has(key));
    known.current = now;
    if (changed.length > 0) {
      setUpdated(new Set(changed));
      setOpen(true);
    }
  }, [orders]);

  return (
    <section className="card" aria-labelledby="orders-heading">
      <details className="orders" open={open} onToggle={(e) => setOpen(e.currentTarget.open)}>
        <summary>
          <h2 id="orders-heading">My orders</h2>
          <span className="count">{orders.length}</span>
          {updated.size > 0 && <span className="tag tag-completed" data-testid="orders-updated">Updated</span>}
        </summary>
        {orders.length === 0 && <p className="subtle">No orders on this account.</p>}
        {orders.map((order) => (
          <div key={order.order_id} className="order">
            <p className="order-title">
              <strong>{order.order_id}</strong>
              <span className="subtle small">{order.status}{order.delivered_on ? ` ${order.delivered_on}` : ""}</span>
            </p>
            <table>
              <thead><tr><th scope="col">Line</th><th scope="col">Item</th><th scope="col">Paid</th><th scope="col">Status</th></tr></thead>
              <tbody>
                {order.lines.map((line) => (
                  <tr key={line.line_id} className={updated.has(lineKey(order.order_id, line)) ? "line-updated" : undefined}>
                    <td>{line.line_id}</td>
                    <td>{line.sku}{line.size ? ` (size ${line.size})` : ""}</td>
                    <td>{formatUsd(line.paid_cents)}</td>
                    <td><span className={`line-status${line.returned ? " line-status-returned" : ""}`}>{outcomeText(line)}</span></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ))}
      </details>
    </section>
  );
}
