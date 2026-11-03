// REAL browser + REAL stack (no mocks): every decision comes from the local agents and PostgreSQL.
import { expect, test, type APIRequestContext, type Page } from "@playwright/test";

const API = "http://127.0.0.1:5173";
const nonLocalRequests: string[] = [];

interface Started { token: string }

test.beforeAll(async ({ request }) => {
  // The API loads guardrail models and warms the model in the background; wait for it.
  await expect.poll(async () => (await (await request.get(`${API}/health/ready`)).json()).state,
                    { timeout: 300_000, intervals: [2_000] }).toBe("ready");
});

test.beforeEach(async ({ page }) => {
  page.on("request", (req) => {
    const url = new URL(req.url());
    if (!["127.0.0.1", "localhost"].includes(url.hostname)) nonLocalRequests.push(req.url());
  });
});

test.afterAll(() => {
  expect(nonLocalRequests, "browser requests to non-local hosts").toEqual([]);
});

// To run each public case quickly, the test creates the session through the API with that case and resumes
// it in the page from this tab's session storage. The sign-in dropdown's own path is tested at the end.
async function start(page: Page, scenarioId: string): Promise<Started> {
  const created = await page.request.post(`${API}/api/v1/demo/sessions`,
                                          { data: { customer_id: "C100", scenario_id: scenarioId } });
  expect(created.ok()).toBeTruthy();
  const token = (await created.json()).token as string; // test-side only, to read status via the API
  await page.addInitScript((t) => window.sessionStorage.setItem("retail-demo-session", t), token);
  await page.goto("/");
  await expect(page.getByText(/Signed in as/)).toBeVisible();
  return { token };
}

async function send(page: Page, text: string) {
  await page.getByLabel("Message").fill(text);
  await page.getByRole("button", { name: "Send" }).click();
  await expect(page.getByText(/Working…/)).toBeHidden({ timeout: 180_000 });
}

async function confirm(page: Page, name: RegExp = /^Confirm$/) {
  await page.getByRole("button", { name }).click();
  await expect(page.getByText(/Working…/)).toBeHidden({ timeout: 180_000 });
}

async function apiGet(request: APIRequestContext, s: Started, path: string) {
  const response = await request.get(`${API}${path}`, { headers: { "X-Demo-Session": s.token } });
  expect(response.ok()).toBeTruthy();
  return response.json();
}

async function summaries(request: APIRequestContext, s: Started): Promise<string[]> {
  return (await apiGet(request, s, "/api/v1/audit")).map((e: { summary: string }) => e.summary);
}

async function expectBothAgents(request: APIRequestContext, s: Started) {
  const labels = new Set((await apiGet(request, s, "/api/v1/audit")).map((e: { actor_label: string }) => e.actor_label));
  expect(labels).toContain("Customer Resolution Agent");
  expect(labels).toContain("Retail Operations Agent");
  const events = await summaries(request, s);
  expect(events.some((e) => e.startsWith("check_return_eligibility("))).toBeTruthy();
}

test("public_01 refund at the 30-day boundary: propose $30.00, Confirm, one refund", async ({ page, request }) => {
  const s = await start(page, "public_01_refund_boundary");
  await send(page, "Refund line L1 of my order O100. It is unopened.");
  await expect(page.getByTestId("refund-amount")).toHaveText("$30.00");
  await confirm(page);
  await expect(page.getByTestId("exec-state")).toHaveText("Completed (verified in the database)");
  await expect(page.getByRole("cell", { name: "Refunded $30.00" })).toBeVisible(); // the orders panel opens by itself
  const status = await apiGet(request, s, "/api/v1/request-status");
  expect(status.state).toBe("completed");
  expect(status.result.refund_amount_cents).toBe(3000);
  await expectBothAgents(request, s);
  await confirm(page, /Confirm again/);
  await expect(page.getByTestId("notice")).toHaveAttribute("data-code", "replayed");
  const events = await summaries(request, s);
  expect(events.filter((e) => e.startsWith("Committed "))).toHaveLength(1);
  expect(events.some((e) => e.includes("nothing re-run"))).toBeTruthy();
});

test("public_02 exchange M -> L: Confirm decrements TEE-L 3 -> 2 once", async ({ page, request }) => {
  const s = await start(page, "public_02_exchange");
  await send(page, "Exchange line L1 in order O100 for the same tee in size L. It is unopened.");
  await expect(page.getByText("TEE-M → TEE-L", { exact: true })).toBeVisible();
  await confirm(page);
  await expect(page.getByTestId("exec-state")).toHaveText("Completed (verified in the database)");
  const committed = (await summaries(request, s)).filter((e) => e.startsWith("Committed "));
  expect(committed).toEqual([expect.stringContaining("exchange TEE-M → TEE-L (TEE-L stock now 2)")]);
  await expectBothAgents(request, s);
});

test("public_03 expired return is declined with no effects", async ({ page, request }) => {
  const s = await start(page, "public_03_expired");
  await send(page, "Please refund line L1 of order O100.");
  await expect(page.getByTestId("exec-state")).toHaveText("Declined");
  expect((await apiGet(request, s, "/api/v1/request-status")).state).toBe("none");
  expect((await summaries(request, s)).some((e) => e.startsWith("Committed "))).toBeFalsy();
});

test("public_04 ambiguous request clarifies, then proposes and completes", async ({ page, request }) => {
  const s = await start(page, "public_04_clarify");
  await send(page, "I want to return an item from O100.");
  await expect(page.getByTestId("exec-state")).toHaveText("Needs more detail from you");
  await send(page, "I mean line L1, the size M tee. Refund it please. It is unopened.");
  await expect(page.getByTestId("refund-amount")).toHaveText("$30.00");
  await confirm(page);
  await expect(page.getByTestId("exec-state")).toHaveText("Completed (verified in the database)");
  expect((await apiGet(request, s, "/api/v1/request-status")).result.line_id).toBe("L1");
});

test("public_05 stale stock: external change before Confirm, exchange refused", async ({ page, request }) => {
  const s = await start(page, "public_05_stale_stock");
  await send(page, "Exchange line L1 of O100 for the same tee in size L. It is unopened.");
  await expect(page.getByText("TEE-M → TEE-L", { exact: true })).toBeVisible();
  await confirm(page);
  await expect(page.getByTestId("exec-state")).toHaveText("Declined");
  const events = await summaries(request, s);
  expect(events).toContain("Stock of TEE-L changed 3 → 0 (before turn 2)");
  expect(events.some((e) => e.startsWith("Committed "))).toBeFalsy();
  expect((await apiGet(request, s, "/api/v1/request-status")).state).toBe("proposal_invalidated");
});

test("public_06 timeout after commit: reconciled once, repeat approval returns the original", async ({ page, request }) => {
  const s = await start(page, "public_06_timeout_replay");
  await send(page, "Refund line L1 of order O100. It is unopened.");
  await confirm(page);
  await expect(page.getByTestId("exec-state")).toHaveText("Completed (verified in the database)");
  await expect(page.getByTestId("notice")).toHaveAttribute("data-code", "timeout_recovered");
  await confirm(page, /Confirm again/);
  await expect(page.getByTestId("notice")).toHaveAttribute("data-code", "replayed");
  const events = await summaries(request, s);
  expect(events.filter((e) => e.startsWith("Committed "))).toHaveLength(1);
  expect(events.some((e) => e.startsWith("Simulated timeout"))).toBeTruthy();
  expect((await apiGet(request, s, "/api/v1/request-status")).result.refund_amount_cents).toBe(3000);
});

// ---- Approval semantics, retries and client-side privacy (real stack) ----

async function proposeRefund(page: Page): Promise<string> {
  const turn = page.waitForResponse((r) => r.url().endsWith("/api/v1/turns") && r.request().method() === "POST");
  await send(page, "Refund line L1 of my order O100. It is unopened.");
  const body = await (await turn).json();
  expect(body.decision).toBe("propose");
  return body.proposal.proposal_id as string;
}

test("typed 'yes' never commits; only the Confirm action does", async ({ page, request }) => {
  const s = await start(page, "public_01_refund_boundary");
  await proposeRefund(page);
  await send(page, "Yes, I confirm. Please commit the refund now.");
  expect((await apiGet(request, s, "/api/v1/request-status")).state).not.toBe("completed");
  expect((await summaries(request, s)).some((e) => e.startsWith("Committed "))).toBeFalsy();
});

test("Cancel invalidates the proposal: a later confirm is refused with no effect", async ({ page, request }) => {
  const s = await start(page, "public_01_refund_boundary");
  const proposalId = await proposeRefund(page);
  await page.getByRole("button", { name: "Cancel" }).click();
  await expect(page.getByRole("button", { name: /^Confirm$/ })).toHaveCount(0);
  const late = await request.post(`${API}/api/v1/proposals/${proposalId}/confirm`, {
    headers: { "X-Demo-Session": s.token }, data: { client_action_id: "late-confirm-after-cancel" } });
  expect(late.ok()).toBeFalsy();
  expect((await apiGet(request, s, "/api/v1/request-status")).state).not.toBe("completed");
  expect((await summaries(request, s)).some((e) => e.startsWith("Committed "))).toBeFalsy();
});

test("double-click Confirm and a retried client_action_id commit exactly once", async ({ page, request }) => {
  const s = await start(page, "public_01_refund_boundary");
  const proposalId = await proposeRefund(page);
  const actionIds: string[] = [];
  page.on("request", (r) => {
    if (r.url().includes("/confirm") && r.method() === "POST") actionIds.push(r.postDataJSON().client_action_id);
  });
  await page.getByRole("button", { name: /^Confirm$/ }).dblclick();
  await expect(page.getByTestId("exec-state")).toHaveText("Completed (verified in the database)", { timeout: 180_000 });
  expect(actionIds.length).toBe(1); // the in-flight guard drops the second click
  const retry = await request.post(`${API}/api/v1/proposals/${proposalId}/confirm`, {
    headers: { "X-Demo-Session": s.token }, data: { client_action_id: actionIds[0] } });
  expect(retry.ok()).toBeTruthy(); // same client_action_id: stored response, nothing re-run
  const events = await summaries(request, s);
  expect(events.filter((e) => e.startsWith("Committed "))).toHaveLength(1);
  expect((await apiGet(request, s, "/api/v1/request-status")).result.refund_amount_cents).toBe(3000);
});

test("session token and business IDs stay out of the URL, console and localStorage", async ({ page }) => {
  const consoleText: string[] = [];
  page.on("console", (m) => consoleText.push(m.text()));
  const s = await start(page, "public_01_refund_boundary");
  await proposeRefund(page);
  await confirm(page);
  await expect(page.getByTestId("exec-state")).toHaveText("Completed (verified in the database)");
  const local = await page.evaluate(() => JSON.stringify({ ...window.localStorage }));
  for (const secret of [s.token, "REQ-public_01_refund_boundary", "PROP-"]) {
    expect(page.url()).not.toContain(secret);
    expect(local).not.toContain(secret);
    expect(consoleText.join("\n")).not.toContain(secret);
  }
});

test("the audit trail explains an out-of-scope refusal and shows checks in the order they ran", async ({ page, request }) => {
  const s = await start(page, "public_01_refund_boundary");
  await send(page, "what is the weather today?");
  await expect(page.getByText(/I can only help with returns and exchanges/)).toBeVisible();
  await expect(page.getByText(/Message read as out of scope/)).toBeVisible();
  const events = await summaries(request, s);
  const position = (start: string) => events.findIndex((e) => e.startsWith(start));
  expect(position("input check · TaskInjection")).toBeLessThan(position("Business checks read the store"));
  expect(position("Business checks read the store")).toBeLessThan(position("Model call"));
  expect(position("Model call")).toBeLessThan(position("Message read as out of scope"));
  expect(position("lookup_customer_orders")).toBe(-1); // off-topic: the agent reads nothing about the customer
});

test("the Customer agent's own tool call shows in the trail, before the handoff", async ({ page, request }) => {
  const s = await start(page, "public_01_refund_boundary");
  await send(page, "Refund line L1 of my order O100. It is unopened.");
  const events = await summaries(request, s);
  const position = (start: string) => events.findIndex((e) => e.startsWith(start));
  expect(position("input check · TaskInjection")).toBeLessThan(position("lookup_customer_orders"));
  expect(position("lookup_customer_orders")).toBeGreaterThan(-1);
  expect(position("lookup_customer_orders")).toBeLessThan(position("Customer Resolution Agent → Retail Operations Agent"));
  expect(position("Customer Resolution Agent → Retail Operations Agent")).toBeLessThan(position("check_return_eligibility"));
});

test("sign-in by customer ID in the page opens the chat; an unknown ID is refused", async ({ page }) => {
  await page.goto("/");
  await page.getByLabel("Customer ID").fill("NOPE-1");
  await page.getByRole("button", { name: "Start" }).click();
  await expect(page.getByTestId("notice")).toContainText("could not find that customer ID");
  await page.getByLabel("Customer ID").fill("C100");
  await page.getByRole("button", { name: "Start" }).click();
  await expect(page.getByText(/How can I help with a return or an exchange/)).toBeVisible();
  await expect(page.getByText(/Signed in as/)).toContainText("C100");
  await page.getByRole("button", { name: "Sign out" }).click();
  await expect(page.getByLabel("Customer ID")).toBeVisible();
});


test("the dropdown runs a listed case on its own data, faults included", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("option", { name: /^Case 5: / }).waitFor({ state: "attached" });
  await page.getByLabel("Case (optional)").selectOption({ label: "Case 5: public_05_stale_stock" });
  await expect(page.getByLabel("Customer ID")).toHaveValue("C100");
  await page.getByRole("button", { name: "Start" }).click();
  await expect(page.getByText(/Signed in as/)).toBeVisible();
  await send(page, "Exchange line L1 of O100 for the same tee in size L. It is unopened.");
  await confirm(page);
  await expect(page.getByText(/out of stock/).first()).toBeVisible();
  await page.getByRole("button", { name: "Sign out" }).click();
  await expect(page.getByLabel("Customer ID")).toBeVisible();
});

test("an uploaded case runs on the file's data, and a bad file is refused", async ({ page }) => {
  await page.goto("/");
  await page.getByLabel("Case (optional)").selectOption({ label: "New empty case (upload a JSON file)" });
  await page.getByLabel(/Case file/).setInputFiles({ name: "bad.json", mimeType: "application/json", buffer: Buffer.from("{oops") });
  await expect(page.getByRole("alert")).toContainText("isn't a case");

  const product = (sku: string, size: string, stock: number) => ({ sku, product_id: "CAP", size, price_cents: 2500, stock });
  const mine = {
    case_id: "my_new_case", customer_id: "Z900", request_id: "REQ-mine", turns: [], faults: [],
    initial_state: {
      as_of: "2026-10-26", currency: "USD", products: [product("CAP-S", "S", 3), product("CAP-L", "L", 2)],
      orders: [{ order_id: "O900", customer_id: "Z900", status: "delivered", delivered_on: "2026-10-20",
                 lines: [{ line_id: "L1", sku: "CAP-S", paid_cents: 2100, unopened: true, final_sale: false, returned: false }] }],
    },
  };
  await page.getByLabel(/Case file/).setInputFiles({ name: "mine.json", mimeType: "application/json",
                                                     buffer: Buffer.from(JSON.stringify(mine)) });
  await expect(page.getByLabel("Customer ID")).toHaveValue("Z900");
  await page.getByRole("button", { name: "Start" }).click();
  await expect(page.getByText(/Signed in as/)).toContainText("Z900");
  await send(page, "Refund line L1 of order O900.");
  await expect(page.getByText("is the item opened or unopened?")).toBeVisible();
  await expect(page.getByRole("button", { name: /^Confirm$/ })).toHaveCount(0);
  await send(page, "unopened");
  await expect(page.getByRole("heading", { name: /Refund proposal/ })).toBeVisible();
  await confirm(page);
  await expect(page.getByTestId("exec-state")).toHaveText("Completed (verified in the database)");
  await expect(page.getByRole("cell", { name: "$21.00" }).first()).toBeVisible();
  await page.getByRole("button", { name: "Sign out" }).click();

  // The upload never reached the main store: its customer doesn't exist there.
  await page.getByLabel("Customer ID").fill("Z900");
  await page.getByRole("button", { name: "Start" }).click();
  await expect(page.getByTestId("notice")).toContainText("could not find that customer ID");
});


test("the desk asks about the item's condition first: opened gets an apology and is recorded", async ({ page, request }) => {
  const s = await start(page, "public_01_refund_boundary");
  await send(page, "Refund line L1 of order O100.");
  await expect(page.getByText("is the item opened or unopened?")).toBeVisible();
  await expect(page.getByRole("button", { name: /^Confirm$/ })).toHaveCount(0);
  await send(page, "opened");
  await expect(page.getByText(/I'm sorry, I can't proceed with this request because the item has been opened/)).toBeVisible();
  await expect(page.getByTestId("exec-state")).toHaveText("Declined");
  await expect(page.getByRole("button", { name: /^Confirm$/ })).toHaveCount(0);
  const events = await summaries(request, s);
  expect(events).toContain("Customer said line L1 of order O100 was opened; the store now records it as opened");
  expect(events.some((e) => e.startsWith("Committed "))).toBeFalsy();
  await send(page, "Refund line L1 of order O100.");  // the store now says it is opened: no question, no offer
  await expect(page.getByText(/the item has been opened/).last()).toBeVisible();
  await expect(page.getByRole("button", { name: /^Confirm$/ })).toHaveCount(0);
});
