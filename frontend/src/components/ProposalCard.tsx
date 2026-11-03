import { formatUsd } from "../api/client";
import type { Proposal } from "../api/types";

/** Shows only persisted server values. Only the Confirm button can request a commit. */
export function ProposalCard({ proposal, busy, onConfirm, onCancel }: {
  proposal: Proposal;
  busy: boolean;
  onConfirm: (p: Proposal) => void;
  onCancel: (p: Proposal) => void;
}) {
  const pending = proposal.status === "pending";
  return (
    <section className={`card proposal proposal-${proposal.status}`} aria-labelledby="proposal-heading">
      <h2 id="proposal-heading">
        {proposal.remedy === "refund" ? "Refund proposal" : "Exchange proposal"}
        <span className="mono subtle"> {proposal.proposal_id}</span>
      </h2>
      <dl className="facts">
        <dt>Item</dt><dd>Order {proposal.order_id}, line {proposal.line_id}</dd>
        {proposal.remedy === "refund" ? (
          <><dt>Refund</dt><dd data-testid="refund-amount">{formatUsd(proposal.amount_cents)}</dd></>
        ) : (
          <>
            <dt>Exchange</dt><dd>{proposal.original_sku} → {proposal.replacement_sku}</dd>
            <dt>Charge</dt><dd>No additional charge</dd>
          </>
        )}
        <dt>Status</dt><dd>{pending ? "Waiting for your confirmation" : proposal.status === "completed" ? "Completed" : "No longer valid"}</dd>
      </dl>
      {pending && (
        <div className="row">
          <button type="button" onClick={() => onConfirm(proposal)} disabled={busy}>Confirm</button>
          <button type="button" className="secondary" onClick={() => onCancel(proposal)} disabled={busy}>Cancel</button>
        </div>
      )}
      {proposal.status === "completed" && (
        <div className="row">
          <button type="button" className="secondary" onClick={() => onConfirm(proposal)} disabled={busy}>
            Confirm again (duplicate-safety check)
          </button>
        </div>
      )}
    </section>
  );
}
