import type { Problem } from "./api";

export const SEED_MAX = 2 ** 31 - 1;

export interface ParamFields {
  seed: string;
  nRows: string;
  nVars: string;
  lam: string;
  ridge: string;
  condition: string;
}

export const EMPTY_FIELDS: ParamFields = { seed: "", nRows: "", nVars: "", lam: "", ridge: "", condition: "" };

/** Optional integer: null = unset, string = error message. */
export function optInt(raw: string, min: number, max: number): number | null | string {
  const t = raw.trim();
  if (t === "") return null;
  const v = Number(t);
  if (!Number.isInteger(v) || v < min || v > max) return `integer in ${min}..${max}`;
  return v;
}

/** Optional float; `inclusiveMin` picks >= vs > for the lower bound. */
export function optFloat(raw: string, min: number, max: number, inclusiveMin: boolean): number | null | string {
  const t = raw.trim();
  if (t === "") return null;
  const v = Number(t);
  const minOk = inclusiveMin ? v >= min : v > min;
  if (!Number.isFinite(v) || !minOk || v > max) {
    return `number ${inclusiveMin ? ">=" : ">"} ${min} and <= ${max}`;
  }
  return v;
}

export interface ValidatedParams {
  /** Mirrors the server's ProblemParams: only filled, valid fields, or null when empty. */
  params: Record<string, number> | null;
  errors: string[];
  anyFilled: boolean;
}

/**
 * Client-side validation mirroring the server caps exactly. Non-applicable
 * fields for the selected problem are IGNORED (their inputs are disabled),
 * matching the server's kind-specific params.
 */
export function validateParams(problem: Problem, f: ParamFields): ValidatedParams {
  const errors: string[] = [];
  const params: Record<string, number> = {};

  const seed = optInt(f.seed, 0, SEED_MAX);
  if (typeof seed === "string") errors.push(`seed: ${seed}`);
  else if (seed !== null) params.seed = seed;

  const nRows = optInt(f.nRows, 4, 200);
  if (typeof nRows === "string") errors.push(`n_rows: ${nRows}`);
  else if (nRows !== null) params.n_rows = nRows;

  const nVars = optInt(f.nVars, 2, 200);
  if (typeof nVars === "string") errors.push(`n_vars: ${nVars}`);
  else if (nVars !== null) params.n_vars = nVars;

  if (problem === "lasso") {
    const lam = optFloat(f.lam, 0, 100, false);
    if (typeof lam === "string") errors.push(`lam: ${lam}`);
    else if (lam !== null) params.lam = lam;
  }
  if (problem === "logistic") {
    const ridge = optFloat(f.ridge, 0, 100, false);
    if (typeof ridge === "string") errors.push(`ridge: ${ridge}`);
    else if (ridge !== null) params.ridge = ridge;
  }
  if (problem === "least_squares") {
    const condition = optFloat(f.condition, 1, 1e6, true);
    if (typeof condition === "string") errors.push(`condition: ${condition}`);
    else if (condition !== null) params.condition = condition;
  }

  const anyFilled =
    f.seed.trim() !== "" ||
    f.nRows.trim() !== "" ||
    f.nVars.trim() !== "" ||
    (problem === "lasso" && f.lam.trim() !== "") ||
    (problem === "logistic" && f.ridge.trim() !== "") ||
    (problem === "least_squares" && f.condition.trim() !== "");

  return { params: Object.keys(params).length > 0 ? params : null, errors, anyFilled };
}

/** Random seed within the server cap [0, 2^31-1]. */
export function randomSeed(): number {
  return Math.floor(Math.random() * (SEED_MAX + 1));
}
