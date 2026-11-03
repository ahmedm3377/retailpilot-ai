import { formatUsd } from "../api/client";
import type { RequestStatus } from "../api/types";
import type { ExecState } from "../hooks/useDemo";

const TEXT: Record<ExecState, string> = {
  idle: "No request yet",
  clarification_required: "Needs more detail from you",
  awaiting_approval: "Proposal waiting for your confirmation",
  committing: "Committing…",
  completed: "Completed (verified in the database)",
  denied: "Declined",
  cancelled: "Proposal cancelled",
  reconciling: "Checking the stored result…",
  outcome_unverified: "Outcome not verified yet",
  error: "Not processed",
};

export function ExecutionStatus({ exec, status }: { exec: ExecState; status: RequestStatus | null }) {
  const result = status?.state === "completed" ? status.result : null;
  return (
    <section className={`card status status-${exec}`} aria-labelledby="status-heading">
      <h2 id="status-heading">Status</h2>
      <p role="status" aria-live="polite" data-testid="exec-state">{TEXT[exec]}</p>
      {result && (
        <p className="small" data-testid="stored-result">
          Stored result: {result.remedy === "refund" && result.refund_amount_cents !== null
            ? `refund of ${formatUsd(result.refund_amount_cents)}`
            : `exchange ${result.original_sku} → ${result.replacement_sku}`} for order {result.order_id}, line {result.line_id}.
        </p>
      )}
      {status && <p className="subtle small">Request {status.request_id}: {status.state.replaceAll("_", " ")}</p>}
    </section>
  );
}
