import { useMemo, useState } from "react";
import type { AuditEvent } from "../api/types";

const KIND_LABEL: Record<AuditEvent["kind"], string> = {
  llm: "Model", tool: "Tool", guardrail: "Guardrail", handoff: "Handoff", proposal: "Proposal", commit: "Transaction", fault: "External",
  other: "Event",
};

export function AuditTimeline({ events }: { events: AuditEvent[] }) {
  const [showGuards, setShowGuards] = useState(false);
  const turns = useMemo(() => {
    const visible = events.filter((e) => showGuards || e.kind !== "guardrail" || e.severity !== "info");
    const groups = new Map<string, AuditEvent[]>();
    for (const e of visible) {
      const key = e.turn === null ? "Session" : `Turn ${e.turn}`;
      groups.set(key, [...(groups.get(key) ?? []), e]);
    }
    return [...groups.entries()];
  }, [events, showGuards]);
  return (
    <section className="card audit" aria-labelledby="audit-heading">
      <h2 id="audit-heading">Audit trail</h2>
      <label className="check">
        <input type="checkbox" checked={showGuards} onChange={(e) => setShowGuards(e.target.checked)} />
        Show passing guardrail checks
      </label>
      {turns.length === 0 && <p className="subtle">No events yet.</p>}
      {turns.map(([label, items]) => (
        <div key={label} className="audit-turn">
          <h3>{label}</h3>
          <ol>
            {items.map((e) => (
              <li key={e.event_id} className={`audit-event sev-${e.severity} kind-${e.kind}`}>
                <span className="tag">{KIND_LABEL[e.kind]}</span>
                <span className="actor">{e.actor_label}</span>
                <span className="summary">{e.summary}</span>
              </li>
            ))}
          </ol>
        </div>
      ))}
    </section>
  );
}
