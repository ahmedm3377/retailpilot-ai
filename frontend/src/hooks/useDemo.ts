import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError, TransportError, api } from "../api/client";
import type { AuditEvent, CaseSelection, Decision, Proposal, Ready, RequestStatus, Session, TurnResponse } from "../api/types";

export type ExecState =
  | "idle" | "clarification_required" | "awaiting_approval" | "committing" | "completed" | "denied"
  | "cancelled" | "reconciling" | "outcome_unverified" | "error";

export interface ChatMessage {
  id: string;
  role: "user" | "assistant" | "system";
  text: string;
  decision?: Decision | null;
  turn?: number | null;
}

export interface TurnDiagnostics {
  turn_number: number;
  elapsed_ms: number;
  llm_calls: number;
  prompt_tokens: number | null;
  completion_tokens: number | null;
}

export interface Notice {
  kind: "info" | "warning" | "error";
  code: string;
  text: string;
}

type PendingAction =
  | { kind: "message"; id: string; text: string }
  | { kind: "confirm"; id: string; proposalId: string };

const STORAGE_KEY = "retail-demo-session";

export const DECISION_STATE: Record<Decision, ExecState> = {
  propose: "awaiting_approval",
  clarify: "clarification_required",
  deny: "denied",
  completed: "completed",
  error: "error",
};

function readStoredToken(): string | null {
  try {
    return window.sessionStorage.getItem(STORAGE_KEY);
  } catch {
    return null;
  }
}

function historyToMessages(session: Session): ChatMessage[] {
  const out: ChatMessage[] = [];
  for (const item of session.history) {
    if (item.kind === "new_request") {
      out.push({ id: `h${item.seq}`, role: "system", text: "New request" });
      continue;
    }
    if (item.kind === "cancel") {
      out.push({ id: `h${item.seq}`, role: "system", text: "You cancelled the proposal." });
      continue;
    }
    out.push({ id: `h${item.seq}u`, role: item.kind === "confirm" ? "system" : "user",
               text: item.kind === "confirm" ? "You pressed Confirm." : item.message ?? "", turn: item.turn_number });
    if (item.response) {
      out.push({ id: `h${item.seq}a`, role: "assistant", text: item.response, decision: item.decision,
                 turn: item.turn_number });
    }
  }
  return out;
}

function errorNotice(error: unknown): Notice {
  if (error instanceof ApiError) {
    return { kind: error.status >= 500 ? "warning" : "error", code: error.errorCode, text: error.message };
  }
  if (error instanceof TransportError) {
    return { kind: "warning", code: "outcome_unverified",
             text: `${error.message} The result is being checked with the server before anything else happens.` };
  }
  return { kind: "error", code: "unexpected", text: "Something went wrong in the page." };
}

export function useDemo() {
  const [ready, setReady] = useState<Ready | null>(null);
  const [token, setToken] = useState<string | null>(null);
  const [session, setSession] = useState<Session | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [status, setStatus] = useState<RequestStatus | null>(null);
  const [audit, setAudit] = useState<AuditEvent[]>([]);
  const [diagnostics, setDiagnostics] = useState<TurnDiagnostics[]>([]);
  const [exec, setExec] = useState<ExecState>("idle");
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState<Notice | null>(null);
  const [retryable, setRetryable] = useState<PendingAction | null>(null);
  const inFlight = useRef(false);

  // Read-only loads on mount (GET only; never a POST from an effect).
  useEffect(() => {
    let live = true;
    api.ready().then((r) => live && setReady(r)).catch(() => live && setReady(null));
    // The page never writes a token to storage. A token placed in this tab's session storage (the
    // Playwright suite does this to open a specific data set) resumes that session.
    const stored = readStoredToken();
    if (stored) {
      api.session(stored).then((s) => {
        if (!live) return;
        setToken(stored);
        setSession(s);
        setMessages(historyToMessages(s));
      }).catch(() => {
        try { window.sessionStorage.removeItem(STORAGE_KEY); } catch { /* storage unavailable */ }
      });
    }
    const timer = window.setInterval(() => {
      api.ready().then((r) => live && setReady(r)).catch(() => live && setReady(null));
    }, 15_000);
    return () => { live = false; window.clearInterval(timer); };
  }, []);

  const refresh = useCallback(async (t: string) => {
    const [s, st, a] = await Promise.all([api.session(t), api.requestStatus(t), api.audit(t)]);
    setSession(s);
    setStatus(st);
    setAudit(a);
    return { session: s, status: st };
  }, []);

  /** After an unknown outcome, re-read persisted status; never guess. */
  const reconcile = useCallback(async (t: string) => {
    setExec("reconciling");
    try {
      const { status: st } = await refresh(t);
      setExec(st.state === "completed" ? "completed"
        : st.state === "pending_confirmation" ? "awaiting_approval" : "outcome_unverified");
    } catch {
      setExec("outcome_unverified");
    }
  }, [refresh]);

  const applyTurn = useCallback((resp: TurnResponse) => {
    setMessages((m) => [...m, { id: `t${resp.turn_number}-${resp.correlation_id}`, role: "assistant",
                                text: resp.response, decision: resp.decision, turn: resp.turn_number }]);
    setExec(DECISION_STATE[resp.decision]);
    setDiagnostics((d) => [...d, { turn_number: resp.turn_number, elapsed_ms: resp.elapsed_ms,
                                   llm_calls: resp.llm_calls, prompt_tokens: resp.prompt_tokens,
                                   completion_tokens: resp.completion_tokens }]);
    if (resp.timeout_recovered) {
      setNotice({ kind: "info", code: "timeout_recovered",
                  text: "The confirmation timed out after it was saved; the stored result was read back." });
    } else if (resp.replayed) {
      setNotice({ kind: "info", code: "replayed",
                  text: "This approval was already completed. The original result was returned; nothing ran twice." });
    } else if (resp.guardrail_blocked) {
      setNotice({ kind: "warning", code: "guardrail_blocked",
                  text: "That message was blocked by a safety check. Nothing was changed." });
    }
  }, []);

  const run = useCallback(async (action: PendingAction) => {
    if (!token || inFlight.current) return;
    inFlight.current = true;
    setBusy(true);
    setNotice(null);
    if (action.kind === "confirm") setExec("committing");
    try {
      const resp = action.kind === "message"
        ? await api.sendMessage(token, action.text, action.id)
        : await api.confirm(token, action.proposalId, action.id);
      setRetryable(null);
      applyTurn(resp);
      await refresh(token);
    } catch (error) {
      setNotice(errorNotice(error));
      if (error instanceof TransportError || (error instanceof ApiError && error.status >= 500)) {
        setRetryable(action);
        await reconcile(token);
      } else {
        setExec("error");
        await refresh(token).catch(() => undefined);
      }
    } finally {
      inFlight.current = false;
      setBusy(false);
    }
  }, [token, applyTurn, refresh, reconcile]);

  const sendMessage = useCallback((text: string) => {
    const trimmed = text.trim();
    if (!trimmed || inFlight.current) return;
    const id = crypto.randomUUID(); // one ID per user action; retries reuse it
    setMessages((m) => [...m, { id: `u-${id}`, role: "user", text: trimmed }]);
    void run({ kind: "message", id, text: trimmed });
  }, [run]);

  const confirm = useCallback((proposal: Proposal) => {
    if (inFlight.current) return;
    setMessages((m) => [...m, { id: `c-${crypto.randomUUID()}`, role: "system", text: "You pressed Confirm." }]);
    void run({ kind: "confirm", id: crypto.randomUUID(), proposalId: proposal.proposal_id });
  }, [run]);

  const retry = useCallback(() => {
    if (retryable) void run(retryable);
  }, [retryable, run]);

  const cancel = useCallback(async (proposal: Proposal) => {
    if (!token || inFlight.current) return;
    inFlight.current = true;
    setBusy(true);
    try {
      await api.cancel(token, proposal.proposal_id, crypto.randomUUID());
      setMessages((m) => [...m, { id: `x-${crypto.randomUUID()}`, role: "system", text: "You cancelled the proposal." }]);
      setExec("cancelled");
      await refresh(token);
    } catch (error) {
      setNotice(errorNotice(error));
    } finally {
      inFlight.current = false;
      setBusy(false);
    }
  }, [token, refresh]);

  /** Demo sign-in with a customer ID; the token stays in memory only (a reload asks again). */
  const signIn = useCallback(async (customerId: string, selection: CaseSelection = {}) => {
    const id = customerId.trim();
    if (!id || inFlight.current) return;
    inFlight.current = true;
    setBusy(true);
    setNotice(null);
    try {
      const created = await api.createSession(id, selection);
      setToken(created.token);
      setSession(created.session);
      setMessages(historyToMessages(created.session));
      setDiagnostics([]);
      setRetryable(null);
      setExec("idle");
      await refresh(created.token);
    } catch (error) {
      setNotice(errorNotice(error));
    } finally {
      inFlight.current = false;
      setBusy(false);
    }
  }, [refresh]);

  /** Ends the demo session on the server (its data is discarded) and returns to sign-in. */
  const signOut = useCallback(async () => {
    if (inFlight.current) return;
    if (token && session) await api.endSession(token, session.session_id).catch(() => undefined);
    try { window.sessionStorage.removeItem(STORAGE_KEY); } catch { /* storage unavailable */ }
    setToken(null);
    setSession(null);
    setMessages([]);
    setStatus(null);
    setAudit([]);
    setDiagnostics([]);
    setRetryable(null);
    setNotice(null);
    setExec("idle");
  }, [token, session]);

  return {
    ready, session, messages, status, audit, diagnostics, exec, busy, notice, retryable,
    signIn, signOut, sendMessage, confirm, cancel, retry,
    refresh: () => (token ? refresh(token) : Promise.resolve(null)),
    dismissNotice: () => setNotice(null),
  };
}
