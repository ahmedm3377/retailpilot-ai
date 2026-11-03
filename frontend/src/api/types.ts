// Mirrors app/api/schemas.py (FastAPI/Pydantic is authoritative). No business logic here.

export type Decision = "propose" | "clarify" | "deny" | "completed" | "error";

export interface Proposal {
  proposal_id: string;
  remedy: "refund" | "exchange";
  order_id: string;
  line_id: string;
  original_sku: string;
  replacement_sku: string | null;
  amount_cents: number;
  status: "pending" | "completed" | "invalidated";
}

export interface OrderLine {
  line_id: string;
  sku: string;
  product_id: string | null;
  size: string | null;
  paid_cents: number;
  returned: boolean;
  outcome: "refund" | "exchange" | null;
  refund_cents: number | null;
  replacement_sku: string | null;
}

export interface Order {
  order_id: string;
  status: string;
  delivered_on: string | null;
  lines: OrderLine[];
}

export interface HistoryItem {
  seq: number;
  kind: "message" | "confirm" | "cancel" | "new_request";
  turn_number: number | null;
  request_id: string;
  message: string | null;
  decision: Decision | null;
  response: string | null;
}

export interface Session {
  session_id: string;
  scenario_id: string;
  customer_id: string;
  request_id: string;
  turn_number: number;
  status: "active" | "ended";
  latest_proposal: Proposal | null;
  orders: Order[];
  history: HistoryItem[];
}

export interface CaseOption {
  case_id: string;
  customer_id: string;
}

/** A case read from the person's file; the server validates everything else in it. */
export interface UploadedCase {
  case_id: string;
  customer_id: string;
  [field: string]: unknown;
}

/** Which data a session works on: nothing chosen means the main store. */
export interface CaseSelection {
  scenario_id?: string;
  uploaded_case?: UploadedCase;
}

export interface CreatedSession {
  token: string;
  session: Session;
}

export type RequestState = "completed" | "pending_confirmation" | "proposal_invalidated" | "none";

export interface CommittedResult {
  remedy: "refund" | "exchange" | null;
  order_id: string | null;
  line_id: string | null;
  refund_amount_cents: number | null;
  original_sku: string | null;
  replacement_sku: string | null;
}

export interface RequestStatus {
  request_id: string;
  state: RequestState;
  result: CommittedResult | null;
  latest_proposal: Proposal | null;
}

export interface TurnResponse {
  decision: Decision;
  response: string;
  proposal: Proposal | null;
  request_id: string;
  request_status: RequestState;
  turn_number: number;
  elapsed_ms: number;
  llm_calls: number;
  prompt_tokens: number | null;
  completion_tokens: number | null;
  guardrail_blocked: boolean;
  replayed: boolean;
  timeout_recovered: boolean;
  correlation_id: string;
}

export interface AuditEvent {
  event_id: number;
  turn: number | null;
  actor: string;
  actor_label: string;
  kind: "llm" | "tool" | "guardrail" | "handoff" | "proposal" | "commit" | "fault" | "other";
  summary: string;
  outcome: string;
  severity: "info" | "warning" | "error";
}

export interface Ready {
  state: "loading" | "ready" | "degraded";
  postgres: { ok: boolean; version: string | null };
  model_server: { ok: boolean; version: string | null; model: string | null; digest: string | null };
  guardrails: { loaded: boolean; unavailable: string[] };
}

export interface ApiErrorBody {
  error_code: string;
  message: string;
  correlation_id: string | null;
}
