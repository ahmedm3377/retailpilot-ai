// Component tests against a MOCKED API (tests/mockApi.ts). Not evidence of real model behaviour.
import { StrictMode } from "react";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { App } from "../src/App";
import { TOKEN, json, mockApi, proposal, session, turn } from "./mockApi";

async function startDemo() {
  const user = userEvent.setup();
  render(<StrictMode><App /></StrictMode>);
  await user.type(screen.getByLabelText("Customer ID"), "C100");
  await user.click(screen.getByRole("button", { name: "Start" }));
  await screen.findByText(/Signed in as/);
  return user;
}

async function propose(user: ReturnType<typeof userEvent.setup>) {
  await user.type(screen.getByLabelText("Message"), "Refund line L1 of my order O100.");
  await user.click(screen.getByRole("button", { name: "Send" }));
  return screen.findByRole("heading", { name: /Refund proposal/ });
}

afterEach(() => {
  vi.unstubAllGlobals();
  window.sessionStorage.clear();
});

describe("demo flow (mocked API)", () => {
  it("signs in with only a customer ID; the server picks the data and the identity", async () => {
    const api = mockApi();
    await startDemo();
    expect(api.posts().map((c) => c.url)).toEqual(["/api/v1/demo/sessions"]);
    expect(api.posts()[0]?.body).toEqual({ customer_id: "C100" });
    expect(screen.getByText(/Signed in as/)).toHaveTextContent("Signed in as C100");
    expect(screen.getByRole("cell", { name: "$30.00" })).not.toBeVisible(); // orders start collapsed
  });

  it("is chat-first: no test controls, only the greeting and the composer", async () => {
    mockApi();
    render(<StrictMode><App /></StrictMode>);
    expect(screen.queryByText(/data set|scenario|Fixture|Start a new request|Keep this demo session/i))
      .not.toBeInTheDocument();
    expect(screen.queryByLabelText("Message")).not.toBeInTheDocument(); // chat opens after sign-in
    const user = userEvent.setup();
    await user.type(screen.getByLabelText("Customer ID"), "C100");
    await user.click(screen.getByRole("button", { name: "Start" }));
    await screen.findByText(/How can I help with a return or an exchange/);
    expect(screen.queryByText(/data set|scenario|Fixture|Test messages|Start a new request/i)).not.toBeInTheDocument();
    await user.click(screen.getByText("My orders"));
    expect(screen.getByRole("cell", { name: "$30.00" })).toBeVisible();
  });

  it("the case dropdown is optional, sits below the login field and defaults to the main store", async () => {
    const api = mockApi();
    render(<StrictMode><App /></StrictMode>);
    const select = screen.getByLabelText("Case (optional)");
    expect(select).toHaveValue("");
    expect(screen.getByLabelText("Customer ID").compareDocumentPosition(select))
      .toBe(Node.DOCUMENT_POSITION_FOLLOWING);
    expect(await screen.findByRole("option", { name: "Case 2: public_02_exchange" })).toBeInTheDocument();
    expect(screen.getByRole("option", { name: /New empty case/ })).toBeInTheDocument();
    expect(api.posts()).toEqual([]); // loading the list never creates anything
  });

  it("choosing a listed case fills in its customer and asks the server for that case", async () => {
    const api = mockApi();
    const user = userEvent.setup();
    render(<StrictMode><App /></StrictMode>);
    await screen.findByRole("option", { name: "Case 2: public_02_exchange" });
    await user.selectOptions(screen.getByLabelText("Case (optional)"), "public_02_exchange");
    expect(screen.getByLabelText("Customer ID")).toHaveValue("C100");
    await user.click(screen.getByRole("button", { name: "Start" }));
    await screen.findByText(/Signed in as/);
    expect(api.posts()[0]?.body).toEqual({ customer_id: "C100", scenario_id: "public_02_exchange" });
  });

  it.each([["live_store", "main store"], ["uploaded_case", "uploaded case"], ["public_02_exchange", "public_02_exchange"]])(
    "the chat header says which data a %s session works on", async (scenario, shown) => {
      mockApi({ "POST /api/v1/demo/sessions": () => json(200, { token: TOKEN, session: session({ scenario_id: scenario }) }),
                "GET /api/v1/demo/session": () => json(200, session({ scenario_id: scenario })) });
      await startDemo();
      expect(screen.getByTestId("data-source")).toHaveTextContent(shown);
    });

  it("the dropdown's list failing to load leaves the main store and the upload working", async () => {
    const api = mockApi({ "GET /api/v1/demo/cases": () => json(500, {}) });
    const user = userEvent.setup();
    render(<StrictMode><App /></StrictMode>);
    await waitFor(() => expect(api.calls.some((c) => c.url === "/api/v1/demo/cases")).toBe(true));
    expect(screen.getByRole("option", { name: /New empty case/ })).toBeInTheDocument();
    await user.type(screen.getByLabelText("Customer ID"), "C100");
    await user.click(screen.getByRole("button", { name: "Start" }));
    await screen.findByText(/Signed in as/);
    expect(api.posts()[0]?.body).toEqual({ customer_id: "C100" });
  });

  describe("new empty case", () => {
    const one = { case_id: "mine", customer_id: "C100", initial_state: { as_of: "2026-10-26" }, turns: [], faults: [] };
    const file = (content: string, name = "case.json") => new File([content], name, { type: "application/json" });

    async function chooseUpload() {
      const user = userEvent.setup({ applyAccept: false });
      render(<StrictMode><App /></StrictMode>);
      await user.selectOptions(screen.getByLabelText("Case (optional)"), "upload");
      return user;
    }

    it("needs a file before it can start, then sends the case from the file", async () => {
      const api = mockApi();
      const user = await chooseUpload();
      expect(screen.getByRole("button", { name: "Start" })).toBeDisabled();
      await user.upload(screen.getByLabelText(/Case file/), file(JSON.stringify(one)));
      await waitFor(() => expect(screen.getByLabelText("Customer ID")).toHaveValue("C100"));
      expect(api.posts()).toEqual([]); // reading the file sends nothing
      await user.click(screen.getByRole("button", { name: "Start" }));
      await screen.findByText(/Signed in as/);
      expect(api.posts()[0]?.body).toEqual({ customer_id: "C100", uploaded_case: one });
    });

    it("a file with several cases lets the person choose one", async () => {
      const api = mockApi();
      const user = await chooseUpload();
      const other = { ...one, case_id: "other" };
      await user.upload(screen.getByLabelText(/Case file/), file(JSON.stringify({ contract_version: 1, cases: [one, other] })));
      await user.selectOptions(await screen.findByLabelText("Case in this file"), "other");
      await user.click(screen.getByRole("button", { name: "Start" }));
      await screen.findByText(/Signed in as/);
      expect(api.posts()[0]?.body?.uploaded_case).toEqual(other);
    });

    it.each([["not json", "{oops"], ["no case in it", JSON.stringify({ cases: [] })], ["not a case", "[1, 2]"]])(
      "a file that is %s shows why and can't start", async (_, content) => {
        const api = mockApi();
        const user = await chooseUpload();
        await user.upload(screen.getByLabelText(/Case file/), file(content));
        expect(await screen.findByRole("alert")).toHaveTextContent("isn't a case");
        await user.type(screen.getByLabelText("Customer ID"), "C100");
        expect(screen.getByRole("button", { name: "Start" })).toBeDisabled();
        expect(api.posts()).toEqual([]);
      });

    it("a case the server refuses shows its reason and stays on sign-in", async () => {
      mockApi({ "POST /api/v1/demo/sessions": () => json(400, {
        error_code: "invalid_case", message: "Order O1, line L1: SKU X is not in the case's products.", correlation_id: "c2" }) });
      const user = await chooseUpload();
      await user.upload(screen.getByLabelText(/Case file/), file(JSON.stringify(one)));
      await waitFor(() => expect(screen.getByRole("button", { name: "Start" })).toBeEnabled());
      await user.click(screen.getByRole("button", { name: "Start" }));
      expect(await screen.findByTestId("notice")).toHaveTextContent("SKU X is not in the case's products");
      expect(screen.queryByLabelText("Message")).not.toBeInTheDocument();
    });
  });

  it("an unknown customer ID shows an actionable error and stays on sign-in", async () => {
    mockApi();
    const user = userEvent.setup();
    render(<StrictMode><App /></StrictMode>);
    await user.type(screen.getByLabelText("Customer ID"), "C999");
    await user.click(screen.getByRole("button", { name: "Start" }));
    expect(await screen.findByTestId("notice")).toHaveTextContent("could not find that customer ID");
    expect(screen.queryByLabelText("Message")).not.toBeInTheDocument();
  });

  it("sign out ends the session on the server and returns to sign-in", async () => {
    const api = mockApi();
    const user = await startDemo();
    await user.click(screen.getByRole("button", { name: "Sign out" }));
    await screen.findByLabelText("Customer ID");
    expect(api.posts().map((c) => c.url)).toContain("/api/v1/demo/sessions/S-1/end");
  });

  it("after a commit the orders open by themselves and show the outcome from the server", async () => {
    let committed = false;
    const refunded = session();
    refunded.orders[0]!.lines[0] = { ...refunded.orders[0]!.lines[0]!, returned: true, outcome: "refund", refund_cents: 3000 };
    mockApi({
      "GET /api/v1/demo/session": () => json(200, committed ? refunded : session({ latest_proposal: proposal() })),
      "POST /api/v1/proposals/PROP-1/confirm": () => {
        committed = true;
        return json(200, turn({ decision: "completed", response: "Your refund of $30.00 is complete.", proposal: null,
                                request_status: "completed", turn_number: 2 }));
      },
    });
    const user = await startDemo();
    await propose(user);
    expect(screen.getByRole("cell", { name: "Not returned" })).not.toBeVisible();
    await user.click(screen.getByRole("button", { name: "Confirm" }));
    expect(await screen.findByRole("cell", { name: "Refunded $30.00" })).toBeVisible();
    expect(screen.getByTestId("orders-updated")).toBeVisible();
  });

  it("shows what an item was exchanged for", async () => {
    const exchanged = session();
    exchanged.orders[0]!.lines[0] = { ...exchanged.orders[0]!.lines[0]!, returned: true, outcome: "exchange", replacement_sku: "TEE-L" };
    mockApi({ "POST /api/v1/demo/sessions": () => json(200, { token: TOKEN, session: exchanged }),
              "GET /api/v1/demo/session": () => json(200, exchanged) });
    await startDemo();
    expect(screen.getByRole("cell", { name: "Exchanged for TEE-L" })).toBeInTheDocument();
  });

  it("a suggestion only fills the message box", async () => {
    const api = mockApi();
    const user = await startDemo();
    await user.click(screen.getByRole("button", { name: "I'd like to return an item" }));
    expect(screen.getByLabelText("Message")).toHaveValue("I'd like to return an item");
    expect(api.posts().some((c) => c.url === "/api/v1/turns")).toBe(false);
  });

  it("chat goes only to /turns and the proposal card shows the refund from cents", async () => {
    const api = mockApi();
    const user = await startDemo();
    await propose(user);
    const turnCalls = api.posts().filter((c) => c.url === "/api/v1/turns");
    expect(turnCalls).toHaveLength(1);
    expect(turnCalls[0]?.body?.message).toBe("Refund line L1 of my order O100.");
    expect(typeof turnCalls[0]?.body?.client_action_id).toBe("string");
    expect(screen.getByTestId("refund-amount")).toHaveTextContent("$30.00");
    expect(screen.getByTestId("exec-state")).toHaveTextContent("waiting for your confirmation");
  });

  it("typing yes never calls confirm; only the Confirm button does", async () => {
    const api = mockApi();
    const user = await startDemo();
    await propose(user);
    await user.type(screen.getByLabelText("Message"), "yes, I confirm PROP-1");
    await user.click(screen.getByRole("button", { name: "Send" }));
    await waitFor(() => expect(api.posts().filter((c) => c.url === "/api/v1/turns")).toHaveLength(2));
    expect(api.posts().some((c) => c.url.includes("/confirm"))).toBe(false);
    await user.click(screen.getByRole("button", { name: "Confirm" }));
    await screen.findByText("Completed (verified in the database)");
    const confirms = api.posts().filter((c) => c.url.includes("/confirm"));
    expect(confirms.map((c) => c.url)).toEqual(["/api/v1/proposals/PROP-1/confirm"]);
    expect(confirms[0]?.body).toEqual({ client_action_id: expect.any(String) });
    expect(screen.getByTestId("stored-result")).toHaveTextContent("refund of $30.00");
  });

  it("a double click sends one confirmation", async () => {
    let release: () => void = () => undefined;
    const gate = new Promise<void>((resolve) => { release = resolve; });
    const api = mockApi({
      "POST /api/v1/proposals/PROP-1/confirm": async () => {
        await gate;
        return json(200, turn({ decision: "completed", response: "Done.", proposal: null, turn_number: 2 }));
      },
    });
    const user = await startDemo();
    await propose(user);
    const button = screen.getByRole("button", { name: "Confirm" });
    await user.dblClick(button);
    await user.click(button);
    release();
    await waitFor(() => expect(screen.getByTestId("exec-state")).not.toHaveTextContent("Committing"));
    expect(api.posts().filter((c) => c.url.includes("/confirm"))).toHaveLength(1);
  });

  it("an unknown outcome re-reads status and a retry reuses the same action id", async () => {
    let attempts = 0;
    const api = mockApi({
      "POST /api/v1/proposals/PROP-1/confirm": () => {
        attempts += 1;
        if (attempts === 1) throw new TypeError("network down");
        return json(200, turn({ decision: "completed", response: "Done.", proposal: null, turn_number: 2 }));
      },
    });
    const user = await startDemo();
    await propose(user);
    const statusReadsBefore = api.calls.filter((c) => c.url === "/api/v1/request-status").length;
    await user.click(screen.getByRole("button", { name: "Confirm" }));
    const notice = await screen.findByTestId("notice");
    expect(notice).toHaveAttribute("data-code", "outcome_unverified");
    expect(api.calls.filter((c) => c.url === "/api/v1/request-status").length).toBeGreaterThan(statusReadsBefore);
    await user.click(within(notice).getByRole("button", { name: "Retry the same action" }));
    await screen.findByText("Completed (verified in the database)");
    const ids = api.posts().filter((c) => c.url.includes("/confirm")).map((c) => c.body?.client_action_id);
    expect(ids).toHaveLength(2);
    expect(ids[0]).toBe(ids[1]);
  });

  it("cancel invalidates the proposal and hides the Confirm button", async () => {
    mockApi();
    const user = await startDemo();
    await propose(user);
    await user.click(screen.getByRole("button", { name: "Cancel" }));
    await screen.findByText("Proposal cancelled");
    expect(screen.queryByRole("button", { name: "Confirm" })).not.toBeInTheDocument();
  });

  it("busy and validation errors are shown honestly", async () => {
    mockApi({ "POST /api/v1/turns": () => json(503, { error_code: "busy", message: "The local model is busy.",
                                                        correlation_id: "x" }) });
    const user = await startDemo();
    await user.type(screen.getByLabelText("Message"), "hello");
    await user.click(screen.getByRole("button", { name: "Send" }));
    const notice = await screen.findByTestId("notice");
    expect(notice).toHaveAttribute("data-code", "busy");
    expect(screen.queryByText(/Completed/)).not.toBeInTheDocument();
  });

  it("renders model and customer text as text, never HTML", async () => {
    const hostile = '<img src=x onerror="alert(1)"><b>bold</b>';
    mockApi({ "POST /api/v1/turns": () => json(200, turn({ decision: "clarify", response: hostile, proposal: null })) });
    const user = await startDemo();
    await user.type(screen.getByLabelText("Message"), "<script>x</script>");
    await user.click(screen.getByRole("button", { name: "Send" }));
    await screen.findByText(hostile);
    expect(document.querySelector("img")).toBeNull();
    expect(document.querySelector("script")).toBeNull();
    expect(screen.getByText("<script>x</script>")).toBeInTheDocument();
  });

  it("keeps the token out of URLs, the console and storage by default", async () => {
    const log = vi.spyOn(console, "log");
    const api = mockApi();
    const user = await startDemo();
    await propose(user);
    expect(api.calls.every((c) => !c.url.includes(TOKEN))).toBe(true);
    expect(api.calls.filter((c) => !["/api/v1/demo/sessions", "/api/v1/demo/cases"].includes(c.url) && c.url.startsWith("/api/v1/"))
      .every((c) => c.headers["X-Demo-Session"] === TOKEN)).toBe(true);
    expect(JSON.stringify(log.mock.calls)).not.toContain(TOKEN);
    expect(window.sessionStorage.length + window.localStorage.length).toBe(0);
  });
});
