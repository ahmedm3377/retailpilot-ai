import { useEffect, useRef, useState, type FormEvent } from "react";
import type { ChatMessage } from "../hooks/useDemo";

// Static UI text, not a model turn: it is never sent to the agents or counted as a turn.
const GREETING = "Hello! How can I help with a return or an exchange today? Tell me what you'd like to do, " +
  "and I'll ask for anything I need, such as the order or the item.";

// Fills the box only: the customer still reads, edits and sends the message themselves.
const SUGGESTIONS = ["I'd like to return an item", "I'd like to exchange an item for a different size"];

const DECISION_LABEL: Record<string, string> = {
  propose: "Proposal", clarify: "Needs detail", deny: "Declined", completed: "Completed", error: "Not processed",
};

const DATA_LABEL: Record<string, string> = { live_store: "main store", uploaded_case: "uploaded case" };

export function ChatPanel({ messages, busy, customerId, scenarioId, onSend, onSignOut }: {
  messages: ChatMessage[];
  busy: boolean;
  customerId: string;
  scenarioId: string;
  onSend: (text: string) => void;
  onSignOut: () => void;
}) {
  const [draft, setDraft] = useState("");
  const list = useRef<HTMLOListElement>(null);
  useEffect(() => { // keep the newest message in view
    if (list.current) list.current.scrollTop = list.current.scrollHeight;
  }, [messages, busy]);
  const submit = (event: FormEvent) => {
    event.preventDefault();
    if (busy || !draft.trim()) return;
    onSend(draft);
    setDraft("");
  };
  return (
    <section className="card chat" aria-labelledby="chat-heading">
      <div className="chat-header">
        <h2 id="chat-heading">Conversation</h2>
        <span className="subtle small">
          Signed in as <strong>{customerId}</strong> · <span data-testid="data-source">{DATA_LABEL[scenarioId] ?? scenarioId}</span>
        </span>
        <button type="button" className="link" onClick={onSignOut} disabled={busy}>Sign out</button>
      </div>
      <ol ref={list} className="messages" aria-live="polite" aria-relevant="additions">
        <li className="msg msg-assistant msg-greeting">
          <span className="msg-role">Desk</span>
          <p>{GREETING}</p>
        </li>
        {messages.map((m) => (
          <li key={m.id} className={`msg msg-${m.role}`}>
            <span className="msg-role">{m.role === "user" ? "You" : m.role === "assistant" ? "Desk" : "Action"}</span>
            {m.decision && <span className={`tag tag-${m.decision}`}>{DECISION_LABEL[m.decision]}</span>}
            <p>{m.text}</p>
          </li>
        ))}
        {busy && (
          <li className="msg msg-pending" aria-busy="true">
            <span className="dots" aria-hidden="true"><i /><i /><i /></span>
            Working… (local model, usually 5–20 s)
          </li>
        )}
      </ol>
      {messages.length === 0 && !busy && (
        <div className="suggestions">
          {SUGGESTIONS.map((text) => <button key={text} type="button" onClick={() => setDraft(text)}>{text}</button>)}
        </div>
      )}
      <form onSubmit={submit} className="composer">
        <label htmlFor="composer-input">Message</label>
        <textarea id="composer-input" value={draft} maxLength={2000} rows={2}
                  placeholder="Describe what you want to return or exchange"
                  onChange={(e) => setDraft(e.target.value)}
                  onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) submit(e); }} />
        <div className="row">
          <button type="submit" disabled={busy || !draft.trim()}>Send</button>
          <span className="subtle small">Typing “yes” never approves anything — use the Confirm button.</span>
        </div>
      </form>
    </section>
  );
}
