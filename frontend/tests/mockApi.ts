// MOCKED backend for isolated component tests only. Real end-to-end checks live in e2e/ (Playwright).
import { vi } from "vitest";
import type { Proposal, RequestStatus, Session, TurnResponse } from "../src/api/types";

export interface Call {
  method: string;
  url: string;
  headers: Record<string, string>;
  body: Record<string, unknown> | null;
}

type Handler = (call: Call) => Response | Promise<Response>;

export const json = (status: number, body: unknown) =>
  new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });

export const TOKEN = "tok_secret_value_123";

export const proposal = (overrides: Partial<Proposal> = {}): Proposal => ({
  proposal_id: "PROP-1", remedy: "refund", order_id: "O100", line_id: "L1", original_sku: "TEE-M",
  replacement_sku: null, amount_cents: 3000, status: "pending", ...overrides,
});

export const session = (overrides: Partial<Session> = {}): Session => ({
  session_id: "S-1", scenario_id: "public_01_refund_boundary", customer_id: "C100", request_id: "REQ-1",
  turn_number: 0, status: "active", latest_proposal: null, history: [],
  orders: [{ order_id: "O100", status: "delivered", delivered_on: "2026-09-26",
             lines: [{ line_id: "L1", sku: "TEE-M", product_id: "TEE", size: "M", paid_cents: 3000, returned: false,
                        outcome: null, refund_cents: null, replacement_sku: null }] }],
  ...overrides,
});

export const status = (overrides: Partial<RequestStatus> = {}): RequestStatus => ({
  request_id: "REQ-1", state: "none", result: null, latest_proposal: null, ...overrides,
});

export const turn = (overrides: Partial<TurnResponse> = {}): TurnResponse => ({
  decision: "propose", response: "I can refund line L1 of order O100 for $30.00.", proposal: proposal(),
  request_id: "REQ-1", request_status: "pending_confirmation", turn_number: 1, elapsed_ms: 5200, llm_calls: 4,
  prompt_tokens: 1800, completion_tokens: 150, guardrail_blocked: false, replayed: false, timeout_recovered: false,
  correlation_id: "S-1-t1", ...overrides,
});

export function mockApi(overrides: Record<string, Handler> = {}) {
  const calls: Call[] = [];
  let currentStatus = status();
  const routes: Record<string, Handler> = {
    "GET /health/ready": () => json(200, { state: "ready", postgres: { ok: true, version: "17.11" },
      model_server: { ok: true, version: "1.0", model: "provider/test-model", digest: "abc123" },
      guardrails: { loaded: true, unavailable: [] } }),
    "GET /api/v1/demo/cases": () => json(200, [{ case_id: "public_01_refund_boundary", customer_id: "C100" },
                                               { case_id: "public_02_exchange", customer_id: "C100" }]),
    "POST /api/v1/demo/sessions": (call) => call.body?.customer_id === "C100"
      ? json(200, { token: TOKEN, session: session() })
      : json(404, { error_code: "unknown_customer", message: "We could not find that customer ID. Check it and try again.",
                    correlation_id: "c1" }),
    "POST /api/v1/demo/sessions/S-1/end": () => json(200, { ended: true }),
    "GET /api/v1/demo/session": () => json(200, session({ latest_proposal: currentStatus.latest_proposal })),
    "GET /api/v1/request-status": () => json(200, currentStatus),
    "GET /api/v1/audit": () => json(200, [
      { event_id: 1, turn: 1, actor: "retail_operations_agent", actor_label: "Retail Operations Agent", kind: "tool",
        summary: "check_return_eligibility(O100, L1) → ELIGIBLE", outcome: "ELIGIBLE", severity: "info" }]),
    "POST /api/v1/turns": () => {
      currentStatus = status({ state: "pending_confirmation", latest_proposal: proposal() });
      return json(200, turn());
    },
    "POST /api/v1/proposals/PROP-1/confirm": () => {
      currentStatus = status({ state: "completed", latest_proposal: proposal({ status: "completed" }),
        result: { remedy: "refund", order_id: "O100", line_id: "L1", refund_amount_cents: 3000, original_sku: "TEE-M",
                  replacement_sku: null } });
      return json(200, turn({ decision: "completed", response: "Your refund of $30.00 is complete.", proposal: null,
                              request_status: "completed", turn_number: 2 }));
    },
    "POST /api/v1/proposals/PROP-1/cancel": () => {
      currentStatus = status({ state: "proposal_invalidated", latest_proposal: proposal({ status: "invalidated" }) });
      return json(200, currentStatus);
    },
    ...overrides,
  };
  const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    const method = init?.method ?? "GET";
    const headers = (init?.headers ?? {}) as Record<string, string>;
    const body = typeof init?.body === "string" ? (JSON.parse(init.body) as Record<string, unknown>) : null;
    const call = { method, url, headers, body };
    calls.push(call);
    const handler = routes[`${method} ${url}`];
    if (!handler) return json(404, { error_code: "not_found", message: `no mock for ${method} ${url}` });
    return handler(call);
  });
  vi.stubGlobal("fetch", fetchMock);
  return { calls, posts: () => calls.filter((c) => c.method === "POST") };
}
