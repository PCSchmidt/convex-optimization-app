import { useMemo } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { SolveResponse } from "../api";

/**
 * Plots the returned history tail. Preferred view: |objective - ground truth|
 * (the objective gap) on a LOG scale. If any gap is zero/non-positive (or the
 * ground truth makes gaps meaningless) it falls back to the raw objective
 * value on a linear scale - stated honestly in the caption, never faked.
 */
export default function ConvergenceChart({ result }: { result: SolveResponse }) {
  const { data, useLog, caption } = useMemo(() => {
    const gt = result.ground_truth_objective;
    const gaps = result.history_tail.map((r) => Math.abs(r.objective - gt));
    const allPositive = gaps.length > 0 && gaps.every((g) => Number.isFinite(g) && g > 0);
    if (allPositive) {
      return {
        data: result.history_tail.map((r, i) => ({ iteration: r.iteration, value: gaps[i] })),
        useLog: true,
        caption: "Objective gap |f(x_k) - f*| vs iteration, log scale (last rows of the run only).",
      };
    }
    return {
      data: result.history_tail.map((r) => ({ iteration: r.iteration, value: r.objective })),
      useLog: false,
      caption:
        "Objective value f(x_k) vs iteration, linear scale (the gap is not plottable on a log scale for this run).",
    };
  }, [result]);

  if (data.length === 0) {
    return <p className="chart-empty">The response carried no history rows (tail was 0); nothing to plot.</p>;
  }

  return (
    <figure className="convergence-figure">
      <ResponsiveContainer width="100%" height={260}>
        <LineChart data={data} margin={{ top: 8, right: 16, bottom: 8, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="iteration" tick={{ fontSize: 12 }} />
          <YAxis
            scale={useLog ? "log" : "linear"}
            domain={useLog ? ["auto", "auto"] : undefined}
            allowDataOverflow={useLog}
            tick={{ fontSize: 12 }}
            width={88}
          />
          <Tooltip />
          <Line type="monotone" dataKey="value" isAnimationActive={false} dot={true} stroke="var(--accent)" />
        </LineChart>
      </ResponsiveContainer>
      <figcaption>{caption}</figcaption>
    </figure>
  );
}
