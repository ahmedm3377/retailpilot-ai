// Typed fetch client. Relative URLs only (Vite proxies to FastAPI on loopback).
// The demo session token travels in the X-Demo-Session header, never in a URL.
import type {
  ApiErrorBody, AuditEvent, CaseOption, CaseSelection, CreatedSession, Ready, RequestStatus, Session, TurnResponse,
} from "./types";

export class ApiError extends Error {
  readonly status: number;
  readonly errorCode: string;
  readonly correlationId: string | null;

  constructor(status: number, body: ApiErrorBody) {
    super(body.message);
    this.status = status;
    this.errorCode = body.error_code;
    this.correlationId = body.correlation_id;
  }
}

/** Network failure or client timeout: the server outcome is unknown and must be re-read. */
export class TransportError extends Error {
  readonly aborted: boolean;

  constructor(message: string, aborted: boolean) {
    super(message);
    this.aborted = aborted;
  }
}

const TURN_TIMEOUT_MS = 240_000;
const READ_TIMEOUT_MS = 20_000;

async function request<T>(method: "GET" | "POST", path: string, token: string | null, body?: unknown,
                          timeoutMs = READ_TIMEOUT_MS): Promise<T> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  const headers: Record<string, string> = { Accept: "application/json" };
  if (token) headers["X-Demo-Session"] = token;
  if (body !== undefined) headers["Content-Type"] = "application/json";
  let response: Response;
  try {
    response = await fetch(path, {
      method, headers, signal: controller.signal, body: body === undefined ? undefined : JSON.stringify(body),
    });
  } catch (error) {
    const aborted = error instanceof DOMException && error.name === "AbortError";
    throw new TransportError(aborted ? "The request timed out." : "The server could not be reached.", aborted);
  } finally {
    clearTimeout(timer);
  }
  const text = await response.text();
  let parsed: unknown = null;
  try {
    parsed = text ? JSON.parse(text) : null;
  } catch {
    parsed = null;
  }
  if (!response.ok) {
    const err = parsed as Partial<ApiErrorBody> | null;
    throw new ApiError(response.status, {
      error_code: err?.error_code ?? `http_${response.status}`,
      message: err?.message ?? "Unexpected server response.",
      correlation_id: err?.correlation_id ?? null,
    });
  }
  return parsed as T;
}

export const api = {
  ready: () => request<Ready>("GET", "/health/ready", null),
  cases: () => request<CaseOption[]>("GET", "/api/v1/demo/cases", null),
  // Demo sign-in: the customer ID, and optionally a case. The server binds the session to that data.
  createSession: (customerId: string, selection: CaseSelection = {}) =>
    request<CreatedSession>("POST", "/api/v1/demo/sessions", null, { customer_id: customerId, ...selection }, 120_000),
  session: (token: string) => request<Session>("GET", "/api/v1/demo/session", token),
  sendMessage: (token: string, message: string, clientActionId: string) =>
    request<TurnResponse>("POST", "/api/v1/turns", token, { message, client_action_id: clientActionId },
                          TURN_TIMEOUT_MS),
  confirm: (token: string, proposalId: string, clientActionId: string) =>
    request<TurnResponse>("POST", `/api/v1/proposals/${encodeURIComponent(proposalId)}/confirm`, token,
                          { client_action_id: clientActionId }, TURN_TIMEOUT_MS),
  cancel: (token: string, proposalId: string, clientActionId: string) =>
    request<RequestStatus>("POST", `/api/v1/proposals/${encodeURIComponent(proposalId)}/cancel`, token,
                           { client_action_id: clientActionId }),
  requestStatus: (token: string) => request<RequestStatus>("GET", "/api/v1/request-status", token),
  audit: (token: string) => request<AuditEvent[]>("GET", "/api/v1/audit", token),
  endSession: (token: string, sessionId: string) =>
    request<{ ended: boolean }>("POST", `/api/v1/demo/sessions/${encodeURIComponent(sessionId)}/end`, token, {}),
};

export function formatUsd(cents: number): string {
  return new Intl.NumberFormat("en-US", { style: "currency", currency: "USD" }).format(cents / 100);
}
