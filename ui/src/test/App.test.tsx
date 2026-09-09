import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import App from "../App";
import type { SolveResponse } from "../api";

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

function mockFetch(handler: (url: string, init?: RequestInit) => Promise<Response>) {
  const fn = vi.fn(async (url: string, init?: RequestInit) => handler(url, init));
  vi.stubGlobal("fetch", fn);
  return fn;
}

const jsonResponse = (body: unknown, status = 200, headers: Record<string, string> = {}) =>
  new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json", ...headers } });

const CONVERGED_RESULT: SolveResponse = {
  problem: "least_squares",
  method: "gd",
  ground_truth_source: "closed-form normal equations (numpy.linalg.solve)",
  iterations: 214,
  converged: true,
  tol: 1e-10,
  final_objective: 1.2345678e-12,
  ground_truth_objective: 1.2e-27,
  final_objective_gap: 1.23e-15,
  final_residual: 3.4e-16,
  history_tail: [
    { iteration: 210, objective: 5e-13, residual: 1e-11 },
    { iteration: 214, objective: 1.23e-15, residual: 8e-13 },
  ],
  parameters: null,
};

async function solve(user: ReturnType<typeof userEvent.setup>) {
  await user.click(screen.getByRole("button", { name: /^Solve$/ }));
}

describe("convex_optimization workbench UI", () => {
  it("plain-language explainer renders with problems, methods and the worked example", () => {
    render(<App />);
    expect(
      screen.getByRole("heading", { name: /plain-language guide/i }),
    ).toBeInTheDocument();
    // The three problem types, by their real-world plain names.
    expect(screen.getByText(/draw the best trend line/i)).toBeInTheDocument();
    expect(screen.getByText(/find the trend line that ignores junk/i)).toBeInTheDocument();
    expect(screen.getByText(/yes\/no questions/i)).toBeInTheDocument();
    // The four methods.
    expect(screen.getByText(/steady steps with a glide/i)).toBeInTheDocument();
    // The worked example with its reproducible parameters.
    expect(screen.getByText(/rent-prediction example/i)).toBeInTheDocument();
    expect(screen.getAllByText(/seed/i).length).toBeGreaterThan(0);
    // Honesty note about the iteration cap is part of the guide.
    expect(screen.getAllByText(/2000-iteration cap/i).length).toBeGreaterThan(0);
  });

  it("frozen instance: empty params note shown and /solve body has no params", async () => {
    const fetchSpy = mockFetch(async (url, init) => {
      if (url === "/solve") return jsonResponse(CONVERGED_RESULT);
      throw new Error(`unexpected fetch ${url} ${init?.body}`);
    });
    const user = userEvent.setup();
    render(<App />);
    expect(screen.getByTestId("instance-note")).toHaveTextContent(/FROZEN Stage 1 benchmark instance/i);
    await solve(user);
    await screen.findByTestId("converged-badge");
    const body = JSON.parse(String(fetchSpy.mock.calls[0][1]?.body));
    expect(body).toEqual({ problem: "least_squares", method: "gd", tail: 20 });
  });

  it("success state renders converged badge, numbers and chart caption", async () => {
    mockFetch(async (url) => {
      if (url === "/solve") return jsonResponse(CONVERGED_RESULT);
      throw new Error(`unexpected fetch ${url}`);
    });
    const user = userEvent.setup();
    render(<App />);
    await solve(user);
    expect(await screen.findByTestId("converged-badge")).toHaveTextContent(/converged/i);
    expect(screen.getByText(/^Result:/i)).toBeInTheDocument();
    expect(screen.getByText("214")).toBeInTheDocument();
    expect(screen.getByText(/log scale/i)).toBeInTheDocument();
    expect(screen.getByText(/Ground truth source:/i)).toBeInTheDocument();
  });

  it("converged=false is rendered honestly as the iteration-cap outcome, not an error", async () => {
    mockFetch(async (url) => {
      if (url === "/solve") {
        return jsonResponse({
          ...CONVERGED_RESULT,
          converged: false,
          iterations: 2000,
          final_objective: 0.42,
          final_objective_gap: 0.4,
        });
      }
      throw new Error(`unexpected fetch ${url}`);
    });
    const user = userEvent.setup();
    render(<App />);
    await solve(user);
    const badge = await screen.findByTestId("converged-badge");
    expect(badge).toHaveTextContent(/hit the 2000-iteration cap/i);
    expect(screen.queryByText(/converged \(tol/i)).not.toBeInTheDocument();
    expect(screen.getByTestId("status-banner")).toHaveTextContent(/iteration cap was reached/i);
  });

  it("loading state is announced and disables the button", async () => {
    let release: (v: Response) => void = () => {};
    mockFetch(
      () =>
        new Promise<Response>((resolve) => {
          release = resolve;
        }),
    );
    const user = userEvent.setup();
    render(<App />);
    await solve(user);
    expect(screen.getByTestId("status-banner")).toHaveTextContent(/Solving on the backend/i);
    expect(screen.getByRole("button", { name: /Solving/i })).toBeDisabled();
    release(jsonResponse(CONVERGED_RESULT));
    await screen.findByTestId("converged-badge");
    expect(screen.getByTestId("status-banner")).toHaveTextContent(/Solve complete/i);
  });

  it("network error shows an actionable message", async () => {
    mockFetch(async () => {
      throw new TypeError("network down");
    });
    const user = userEvent.setup();
    render(<App />);
    await solve(user);
    expect(await screen.findByTestId("status-banner")).toHaveTextContent(/Could not reach the API/i);
    expect(screen.getByTestId("status-banner")).toHaveTextContent(/backend running/i);
  });

  it("422 inapplicable pair shows the server's applicable-methods list", async () => {
    mockFetch(async (url) => {
      if (url === "/solve") {
        return jsonResponse(
          {
            detail: {
              error: "method 'fista' is not applicable to problem 'least_squares'",
              applicable_methods: ["gd", "nesterov"],
            },
          },
          422,
        );
      }
      throw new Error(`unexpected fetch ${url}`);
    });
    const user = userEvent.setup();
    render(<App />);
    await user.selectOptions(screen.getByLabelText(/^Method$/), "fista");
    await solve(user);
    const banner = await screen.findByTestId("status-banner");
    expect(banner).toHaveTextContent(/422/);
    expect(banner).toHaveTextContent(/not applicable to problem/i);
    expect(banner).toHaveTextContent(/applicable methods: gd, nesterov/);
  });

  it("429 shows a Retry-After countdown and disables Solve", async () => {
    mockFetch(async () => jsonResponse({ detail: { error: "rate_limited" } }, 429, { "Retry-After": "30" }));
    const user = userEvent.setup();
    render(<App />);
    await solve(user);
    const banner = await screen.findByTestId("status-banner");
    expect(banner).toHaveTextContent(/Rate limited \(429\)/);
    expect(banner).toHaveTextContent(/retrying in 30s/);
    expect(screen.getByRole("button", { name: /Rate limited - wait/ })).toBeDisabled();
  });

  it("client-side validation blocks an out-of-range seed and shows inline errors", async () => {
    const fetchSpy = mockFetch(async () => {
      throw new Error("solve should not be called");
    });
    const user = userEvent.setup();
    render(<App />);
    await user.type(screen.getByLabelText(/seed/i), "99999999999");
    expect(screen.getByRole("alert")).toHaveTextContent(/seed: integer in 0\.\.2147483647/i);
    expect(screen.getByRole("button", { name: /^Solve$/ })).toBeDisabled();
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it("parse verified=true: spec shown and Run this spec POSTs it to /solve", async () => {
    const fetchSpy = mockFetch(async (url, init) => {
      if (url === "/parse") {
        return jsonResponse({
          problem_request: {
            problem: "lasso",
            method: "fista",
            tail: 5,
            params: { seed: 7, n_rows: 60, n_vars: 10, lam: 0.5 },
          },
          parse_method: "llm",
          verified: true,
          mismatches: [],
        });
      }
      if (url === "/solve") return jsonResponse({ ...CONVERGED_RESULT, problem: "lasso", method: "fista" });
      throw new Error(`unexpected fetch ${url} ${init?.body}`);
    });
    const user = userEvent.setup();
    render(<App />);
    await user.type(screen.getByLabelText(/Describe the problem/i), "lasso, 60 rows, 10 vars, lambda 0.5, seed 7, fista");
    await user.click(screen.getByRole("button", { name: /Parse description/i }));
    expect(await screen.findByText(/verified: the text supports this spec/i)).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /Run this spec/i }));
    await screen.findByTestId("converged-badge");
    const solveCall = fetchSpy.mock.calls.find(([u]) => u === "/solve");
    expect(JSON.parse(String(solveCall?.[1]?.body))).toEqual({
      problem: "lasso",
      method: "fista",
      tail: 5,
      params: { seed: 7, n_rows: 60, n_vars: 10, lam: 0.5 },
    });
  });

  it("parse verified=false: mismatches are surfaced prominently", async () => {
    mockFetch(async (url) => {
      if (url === "/parse") {
        return jsonResponse({
          problem_request: { problem: "lasso", method: "fista", tail: 5, params: { seed: 7 } },
          parse_method: "llm",
          verified: false,
          mismatches: ["the text mentions lambda 0.1 but the spec says 0.5"],
        });
      }
      throw new Error(`unexpected fetch ${url}`);
    });
    const user = userEvent.setup();
    render(<App />);
    await user.type(screen.getByLabelText(/Describe the problem/i), "some contradictory text");
    await user.click(screen.getByRole("button", { name: /Parse description/i }));
    expect(await screen.findByText(/NOT verified/i)).toBeInTheDocument();
    expect(screen.getByText(/the text mentions lambda 0.1 but the spec says 0.5/)).toBeInTheDocument();
  });

  it("parse 503 hides the NL feature with the honest one-line explanation", async () => {
    mockFetch(async (url) => {
      if (url === "/parse") {
        return jsonResponse(
          { detail: { error: "no LLM provider configured", hint: "set LLM_API_KEY" } },
          503,
        );
      }
      throw new Error(`unexpected fetch ${url}`);
    });
    const user = userEvent.setup();
    render(<App />);
    await user.type(screen.getByLabelText(/Describe the problem/i), "anything");
    await user.click(screen.getByRole("button", { name: /Parse description/i }));
    await screen.findByText(/no LLM_API_KEY configured/i);
    expect(screen.queryByLabelText(/Describe the problem/i)).not.toBeInTheDocument();
    // Not an error banner: the solve status region is untouched.
    expect(screen.getByTestId("status-banner")).toHaveTextContent(/Enter a configuration and solve/i);
  });
});
