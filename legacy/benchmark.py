# Benchmark solvers on sample problems

from __future__ import annotations

import argparse
import csv
import time
import random
from typing import Any, Callable, Dict, List, Tuple

import cvxpy as cp

from solvers import parse_expression
from routes import load_problems
from jinja2 import Environment, FileSystemLoader


# -------------------------
# Solver implementations
# -------------------------

def _lp_with_pulp(objective: str, constraints: str) -> Tuple[str, float, None | int]:
    """Solve a linear program using PuLP and measure runtime."""
    from solvers import solve_lp

    start = time.perf_counter()
    solve_lp(objective, constraints)
    runtime = time.perf_counter() - start
    return "pulp", runtime, None


def _lp_with_cvxpy(objective: str, constraints: str, solver: str = "ECOS_BB") -> Tuple[str, float, int]:
    """Solve a linear program using CVXPY with the given solver."""
    terms = parse_expression(objective)
    vars: Dict[str, cp.Variable] = {}
    for _, var in terms:
        vars.setdefault(var, cp.Variable())
    constr_list: List[cp.Constraint] = []
    for line in constraints.splitlines():
        if not line.strip():
            continue
        if "<=" in line:
            lhs, rhs = line.split("<=")
            lhs_terms = parse_expression(lhs)
            for _, var in lhs_terms:
                vars.setdefault(var, cp.Variable())
            expr = sum(coef * vars[v] for coef, v in lhs_terms)
            constr_list.append(expr <= float(rhs.strip()))
        elif ">=" in line:
            lhs, rhs = line.split(">=")
            lhs_terms = parse_expression(lhs)
            for _, var in lhs_terms:
                vars.setdefault(var, cp.Variable())
            expr = sum(coef * vars[v] for coef, v in lhs_terms)
            constr_list.append(expr >= float(rhs.strip()))
    obj_expr = sum(coef * vars[v] for coef, v in terms)
    prob = cp.Problem(cp.Minimize(obj_expr), constr_list)
    start = time.perf_counter()
    prob.solve(solver=getattr(cp, solver))
    runtime = time.perf_counter() - start
    iters = prob.solver_stats.num_iters
    return f"cvxpy_{solver}", runtime, iters


def _qp_with_cvxpy(objective: str, constraints: str, solver: str) -> Tuple[str, float, int]:
    """Solve a quadratic program with CVXPY using the specified solver."""
    terms = parse_expression(objective)
    vars: Dict[str, cp.Variable] = {}
    quad_coeffs: Dict[str, float] = {}
    lin_coeffs: Dict[str, float] = {}
    for coef, term in terms:
        if term.endswith("^2"):
            name = term[:-2]
            vars.setdefault(name, cp.Variable())
            quad_coeffs[name] = quad_coeffs.get(name, 0.0) + coef
        else:
            vars.setdefault(term, cp.Variable())
            lin_coeffs[term] = lin_coeffs.get(term, 0.0) + coef
    obj = sum(c * vars[v] ** 2 for v, c in quad_coeffs.items())
    obj += sum(c * vars[v] for v, c in lin_coeffs.items())
    constr_list: List[cp.Constraint] = []
    for line in constraints.splitlines():
        if not line.strip():
            continue
        if "<=" in line:
            lhs, rhs = line.split("<=")
            lhs_terms = parse_expression(lhs)
            for _, var in lhs_terms:
                vars.setdefault(var, cp.Variable())
            expr = sum(coef * vars[v] for coef, v in lhs_terms)
            constr_list.append(expr <= float(rhs.strip()))
        elif ">=" in line:
            lhs, rhs = line.split(">=")
            lhs_terms = parse_expression(lhs)
            for _, var in lhs_terms:
                vars.setdefault(var, cp.Variable())
            expr = sum(coef * vars[v] for coef, v in lhs_terms)
            constr_list.append(expr >= float(rhs.strip()))
    prob = cp.Problem(cp.Minimize(obj), constr_list)
    start = time.perf_counter()
    prob.solve(solver=getattr(cp, solver))
    runtime = time.perf_counter() - start
    iters = prob.solver_stats.num_iters
    return f"cvxpy_{solver}", runtime, iters


def _generic_cvxpy(
    objective: str,
    constraints: str,
    setup_fn: Callable[[str, str], cp.Problem],
    solver: str,
    *,
    gp: bool = False,
) -> Tuple[str, float, int]:
    """Helper for SDP, conic, and geometric problems."""
    prob = setup_fn(objective, constraints)
    start = time.perf_counter()
    prob.solve(solver=getattr(cp, solver), gp=gp)
    runtime = time.perf_counter() - start
    return f"cvxpy_{solver}", runtime, prob.solver_stats.num_iters


# -------------------------
# Problem setup functions
# -------------------------

# These functions create cvxpy Problems without solving them.

def _sdp_problem(objective: str, constraints: str) -> cp.Problem:
    from parser import parse_matrix
    C = parse_matrix(objective)
    n = C.shape[0]
    X = cp.Variable((n, n), PSD=True)
    constr: List[cp.Constraint] = []
    for line in constraints.splitlines():
        if not line.strip():
            continue
        if "<=" in line:
            A_str, b = line.split("<=")
            from parser import parse_matrix
            A = parse_matrix(A_str)
            constr.append(cp.trace(A @ X) <= float(b))
        elif ">=" in line:
            A_str, b = line.split(">=")
            from parser import parse_matrix
            A = parse_matrix(A_str)
            constr.append(cp.trace(A @ X) >= float(b))
        elif "==" in line:
            A_str, b = line.split("==")
            from parser import parse_matrix
            A = parse_matrix(A_str)
            constr.append(cp.trace(A @ X) == float(b))
    return cp.Problem(cp.Minimize(cp.trace(C @ X)), constr)


def _conic_problem(objective: str, constraints: str) -> cp.Problem:
    from parser import parse_vector, parse_matrix
    c = parse_vector(objective)
    x = cp.Variable(len(c))
    constr: List[cp.Constraint] = []
    for line in constraints.splitlines():
        if not line.strip():
            continue
        if line.startswith("soc:"):
            _, rest = line.split(":", 1)
            A_str, b_str, c_str = rest.split("|")
            A = parse_matrix(A_str)
            b = parse_vector(b_str)
            t = float(c_str)
            constr.append(cp.norm(A @ x + b) <= t)
        elif "<=" in line:
            a_str, b_val = line.split("<=")
            a = parse_vector(a_str)
            constr.append(a @ x <= float(b_val))
        elif ">=" in line:
            a_str, b_val = line.split(">=")
            a = parse_vector(a_str)
            constr.append(a @ x >= float(b_val))
    return cp.Problem(cp.Minimize(c @ x), constr)


def _gp_problem(objective: str, constraints: str) -> cp.Problem:
    from parser import parse_posynomial
    var_names = set(cp.Variable().name for _ in range(0))  # placeholder
    var_names.update(v for _, v in parse_expression(objective))
    for line in constraints.splitlines():
        var_names.update(v for _, v in parse_expression(line))
    vars = {name: cp.Variable(pos=True) for name in var_names}
    obj = parse_posynomial(objective, vars)
    constr: List[cp.Constraint] = []
    for line in constraints.splitlines():
        if not line.strip():
            continue
        if "<=" in line:
            lhs, rhs = line.split("<=")
            constr.append(parse_posynomial(lhs, vars) <= float(rhs))
        elif ">=" in line:
            lhs, rhs = line.split(">=")
            constr.append(parse_posynomial(lhs, vars) >= float(rhs))
        elif "==" in line:
            lhs, rhs = line.split("==")
            constr.append(parse_posynomial(lhs, vars) == float(rhs))
    return cp.Problem(cp.Minimize(obj), constr)


# -------------------------
# Problem generation helpers
# -------------------------

def _generate_lp(n: int) -> tuple[str, str]:
    """Generate a random dense linear program with ``n`` variables."""
    objective = " + ".join(f"{random.randint(1,5)}x{i}" for i in range(n))
    constraints = []
    for j in range(n):
        lhs = " + ".join(f"{random.randint(1,5)}x{i}" for i in range(n))
        rhs = random.randint(n, 2 * n)
        constraints.append(f"{lhs} >= {rhs}")
    return objective, "\n".join(constraints)


def _generate_qp(n: int) -> tuple[str, str]:
    """Generate a random quadratic program with ``n`` variables."""
    objective_terms = [f"{random.randint(1,5)}x{i}^2" for i in range(n)]
    objective = " + ".join(objective_terms)
    constraints = []
    for j in range(n):
        lhs = " + ".join(f"{random.randint(1,5)}x{i}" for i in range(n))
        rhs = random.randint(n, 2 * n)
        constraints.append(f"{lhs} >= {rhs}")
    return objective, "\n".join(constraints)


# Mapping problem types to named solver functions
SOLVERS: Dict[str, Dict[str, Callable[[str, str], Tuple[str, float, Any]]]] = {
    "linear_program": {
        "pulp": _lp_with_pulp,
        "cvxpy_ECOS_BB": lambda o, c: _lp_with_cvxpy(o, c, "ECOS_BB"),
    },
    "quadratic_program": {
        "cvxpy_OSQP": lambda o, c: _qp_with_cvxpy(o, c, "OSQP"),
        "cvxpy_ECOS": lambda o, c: _qp_with_cvxpy(o, c, "ECOS"),
    },
    "semidefinite_program": {
        "cvxpy_SCS": lambda o, c: _generic_cvxpy(o, c, _sdp_problem, "SCS"),
    },
    "conic_program": {
        "cvxpy_SCS": lambda o, c: _generic_cvxpy(o, c, _conic_problem, "SCS"),
    },
    "geometric_program": {
        "cvxpy_SCS": lambda o, c: _generic_cvxpy(o, c, _gp_problem, "SCS", gp=True),
    },
}


def run_benchmarks(
    output: str | None = None,
    *,
    return_results: bool = False,
    size: int | None = None,
    problem_type: str = "linear_program",
    solver_filter: List[str] | None = None,
    html_output: str | None = None,
) -> List[dict] | None:
    """Run benchmarks and optionally write results to files.

    If ``size`` is provided, a random problem of ``problem_type`` is generated
    with that many variables. Otherwise problems are loaded from the
    ``problems`` directory.
    """
    if size is not None:
        if problem_type == "linear_program":
            obj, cons = _generate_lp(size)
        elif problem_type == "quadratic_program":
            obj, cons = _generate_qp(size)
        else:
            raise ValueError("Generated problems supported only for LP or QP")
        problems = [
            {
                "name": f"generated_{problem_type}_{size}",
                "type": problem_type,
                "objective": obj,
                "constraints": cons.splitlines(),
            }
        ]
    else:
        problems = load_problems()
    results = []
    for prob in problems:
        ptype = prob.get("type")
        objective = prob.get("objective", "")
        constraints = "\n".join(prob.get("constraints", []))
        solvers = SOLVERS.get(ptype, {})
        for sname, solver_fn in solvers.items():
            if solver_filter and sname not in solver_filter:
                continue
            solver_name, runtime, iters = solver_fn(objective, constraints)
            results.append({
                "problem": prob.get("name"),
                "type": ptype,
                "solver": solver_name,
                "runtime": runtime,
                "iterations": iters,
            })
    if output:
        with open(output, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["problem", "type", "solver", "runtime", "iterations"])
            writer.writeheader()
            writer.writerows(results)
    else:
        print("problem,type,solver,runtime,iterations")
        for row in results:
            print(",".join(str(row[h]) for h in ["problem", "type", "solver", "runtime", "iterations"]))
    if html_output:
        env = Environment(loader=FileSystemLoader("templates"))
        template = env.get_template("benchmark.html")
        html = template.render(results=results)
        with open(html_output, "w", encoding="utf-8") as f:
            f.write(html)
    if return_results:
        return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark sample or generated optimization problems"
    )
    parser.add_argument("--output", "-o", help="Write results to CSV file")
    parser.add_argument(
        "--size",
        type=int,
        help="Generate a random problem of this size instead of using samples",
    )
    parser.add_argument(
        "--type",
        choices=list(SOLVERS.keys()),
        default="linear_program",
        help="Problem type when generating problems",
    )
    parser.add_argument(
        "--solvers",
        help="Comma separated solver names to run",
    )
    parser.add_argument(
        "--html",
        help="Write HTML table to the specified file",
    )
    args = parser.parse_args()

    solver_list = args.solvers.split(",") if args.solvers else None

    run_benchmarks(
        args.output,
        size=args.size,
        problem_type=args.type,
        solver_filter=solver_list,
        html_output=args.html,
    )


if __name__ == "__main__":
    main()
