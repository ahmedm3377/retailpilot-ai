import { AppShell } from "./components/AppShell";
import { AuditTimeline } from "./components/AuditTimeline";
import { ChatPanel } from "./components/ChatPanel";
import { DiagnosticsPanel } from "./components/DiagnosticsPanel";
import { ErrorNotice } from "./components/ErrorNotice";
import { ExecutionStatus } from "./components/ExecutionStatus";
import { OrderSummary } from "./components/OrderSummary";
import { ProposalCard } from "./components/ProposalCard";
import { SignIn } from "./components/SignIn";
import { useDemo } from "./hooks/useDemo";

export function App() {
  const demo = useDemo();
  const proposal = demo.status?.latest_proposal ?? demo.session?.latest_proposal ?? null;
  const notice = demo.notice && (
    <ErrorNotice notice={demo.notice} canRetry={demo.retryable !== null && !demo.busy} onRetry={demo.retry}
                 onDismiss={demo.dismissNotice} />
  );
  if (!demo.session) {
    return (
      <AppShell ready={demo.ready}>
        <div className="column center sign-in-column">
          {notice}
          <SignIn busy={demo.busy} onSignIn={(id, selection) => void demo.signIn(id, selection)} />
        </div>
      </AppShell>
    );
  }
  return (
    <AppShell ready={demo.ready}>
      <div className="column center">
        {notice}
        <ChatPanel messages={demo.messages} busy={demo.busy} customerId={demo.session.customer_id}
                   scenarioId={demo.session.scenario_id} onSend={demo.sendMessage} onSignOut={() => void demo.signOut()} />
      </div>
      <div className="column side">
        <ExecutionStatus exec={demo.exec} status={demo.status} />
        {proposal && proposal.status !== "invalidated" && (
          <ProposalCard proposal={proposal} busy={demo.busy} onConfirm={demo.confirm}
                        onCancel={(p) => void demo.cancel(p)} />
        )}
        <OrderSummary key={demo.session.session_id} orders={demo.session.orders} />
        <AuditTimeline events={demo.audit} />
        <DiagnosticsPanel turns={demo.diagnostics} />
      </div>
    </AppShell>
  );
}
