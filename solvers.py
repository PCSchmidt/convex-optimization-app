from __future__ import annotations

from typing import Any, Dict, List, Tuple
from typing import Dict, List, Tuple, Optional

# main

import cvxpy as cp
import pulp
import numpy as np
import re

from parser import parse_matrix, parse_vector, parse_posynomial


def parse_expression(expr: str) -> List[Tuple[float, str]]:
    """Parse a simple algebraic expression into coefficient/variable terms."""
    expr = expr.replace(" ", "")
    terms = re.findall(r"([+-]?(?:\d*\.)?\d*)([a-zA-Z]\w*(?:\^2)?)", expr)
    return [
        (
            float(coef)
            if coef and coef not in ["+", "-"]
            else (1.0 if coef != "-" else -1.0),
            var,
        )
        for coef, var in terms
        if var
    ]


def solve_lp(
    objective: str,
    constraints: str,
# codex/add-json-endpoints-with-fastapi
    *,
    return_dict: bool = False,
) -> str | Dict[str, Any]:
    method: Optional[str] = None,
    max_iter: Optional[int] = None,
    tolerance: Optional[float] = None,
) -> str:

#   main
    """Solve a linear program using PuLP.

    Args:
        objective: Objective function as a string, e.g. ``"3x + 2y"``.
        constraints: Newline separated constraint expressions.

    Returns:
        Human readable string describing solver status and variable values or a
        structured dictionary when ``return_dict`` is True.
    """
    prob = pulp.LpProblem("Linear Program", pulp.LpMinimize)

    obj_terms = parse_expression(objective)
    variables = {var: pulp.LpVariable(var) for _, var in obj_terms}

    # Scan constraints for additional variables
    for constraint in constraints.splitlines():
        if "<=" in constraint:
            lhs, _ = constraint.split("<=")
        elif ">=" in constraint:
            lhs, _ = constraint.split(">=")
        else:
            continue
        for _, var in parse_expression(lhs):
            if var not in variables:
                variables[var] = pulp.LpVariable(var)

    prob += pulp.lpSum(coef * variables[var] for coef, var in obj_terms)

    for constraint in constraints.splitlines():
        if not constraint.strip():
            continue
        if "<=" in constraint:
            lhs, rhs = constraint.split("<=")
            lhs_terms = parse_expression(lhs)
            prob += (
                pulp.lpSum(coef * variables[var] for coef, var in lhs_terms)
                <= float(rhs.strip())
            )
        elif ">=" in constraint:
            lhs, rhs = constraint.split(">=")
            lhs_terms = parse_expression(lhs)
            prob += (
                pulp.lpSum(coef * variables[var] for coef, var in lhs_terms)
                >= float(rhs.strip())
            )

    # Determine solver
    solver_name = (method or "cbc").lower()
    solvers = {"cbc": pulp.PULP_CBC_CMD, "glpk": pulp.GLPK_CMD}
    if solver_name not in solvers:
        raise ValueError(f"Unsupported method '{method}'")
    solver_kwargs = {"msg": False}
    if max_iter is not None:
        solver_kwargs["timeLimit"] = int(max_iter)
    if tolerance is not None:
        solver_kwargs["gapRel"] = float(tolerance)
    prob.solve(solvers[solver_name](**solver_kwargs))

    status = pulp.LpStatus[prob.status]
    if status == "Optimal":
        objective_value = pulp.value(prob.objective)
        variables = {var.name: var.varValue for var in prob.variables()}
        result = f"Status: {status}\n"
        for name, val in variables.items():
            result += f"{name} = {val}\n"
        result += f"Objective value: {objective_value}"
    else:
        objective_value = None
        variables = None
        result = f"Status: {status}"

    if return_dict:
        return {
            "status": status,
            "objective_value": objective_value,
            "variables": variables,
        }
    return result


def solve_qp(
    objective: str,
    constraints: str,
# codex/add-json-endpoints-with-fastapi
    *,
    return_dict: bool = False,
) -> str | Dict[str, Any]:
# 
    method: Optional[str] = None,
    max_iter: Optional[int] = None,
    tolerance: Optional[float] = None,
) -> str:
# main
    """Solve a quadratic program using CVXPY.

    Args:
        objective: Quadratic objective as a string with ``^2`` for squared terms.
        constraints: Newline separated constraint expressions.

    Returns:
        Human readable string describing solver status and solution values or a
        structured dictionary when ``return_dict`` is True.
    """
    obj_terms = parse_expression(objective)
    variables: Dict[str, cp.Variable] = {}
    quad_coeffs: Dict[str, float] = {}
    linear_coeffs: Dict[str, float] = {}

    for coef, term in obj_terms:
        if "^2" in term:
            var_name = term.split("^")[0]
            if var_name not in variables:
                variables[var_name] = cp.Variable()
            quad_coeffs[var_name] = coef
        else:
            if term not in variables:
                variables[term] = cp.Variable()
            linear_coeffs[term] = coef

    objective_expr = sum(
        coef * variables[var] ** 2 for var, coef in quad_coeffs.items()
    )
    objective_expr += sum(coef * variables[var] for var, coef in linear_coeffs.items())

    constraints_list: List[cp.Constraint] = []
    for constraint in constraints.splitlines():
        if not constraint.strip():
            continue
        if "<=" in constraint:
            lhs, rhs = constraint.split("<=")
            lhs_terms = parse_expression(lhs)
            lhs_expr = sum(coef * variables[var] for coef, var in lhs_terms)
            constraints_list.append(lhs_expr <= float(rhs.strip()))
        elif ">=" in constraint:
            lhs, rhs = constraint.split(">=")
            lhs_terms = parse_expression(lhs)
            lhs_expr = sum(coef * variables[var] for coef, var in lhs_terms)
            constraints_list.append(lhs_expr >= float(rhs.strip()))

    prob = cp.Problem(cp.Minimize(objective_expr), constraints_list)

    solver_name = (method or "ECOS").upper()
    valid_methods = {"ECOS", "OSQP", "SCS"}
    if solver_name not in valid_methods:
        raise ValueError(f"Unsupported method '{method}'")
    solve_kwargs = {}
    if max_iter is not None:
        if solver_name == "OSQP":
            solve_kwargs["max_iter"] = int(max_iter)
        else:
            solve_kwargs["max_iters"] = int(max_iter)
    if tolerance is not None:
        if solver_name == "SCS":
            solve_kwargs["eps"] = float(tolerance)
        elif solver_name == "OSQP":
            solve_kwargs["eps_abs"] = float(tolerance)
            solve_kwargs["eps_rel"] = float(tolerance)
        else:
            solve_kwargs["abstol"] = float(tolerance)
            solve_kwargs["reltol"] = float(tolerance)

    prob.solve(solver=solver_name, **solve_kwargs)

    if prob.status == cp.OPTIMAL:
        objective_value = float(prob.value)
        variables_val = {name: float(var.value) for name, var in variables.items()}
        result = f"Status: {prob.status}\n"
        for name, val in variables_val.items():
            result += f"{name} = {val}\n"
        result += f"Objective value: {objective_value}"
    else:
        objective_value = None
        variables_val = None
        result = f"Status: {prob.status}"

    if return_dict:
        return {
            "status": prob.status,
            "objective_value": objective_value,
            "variables": variables_val,
        }
    return result


def solve_sdp(
    objective: str,
    constraints: str,
    method: Optional[str] = None,
    max_iter: Optional[int] = None,
    tolerance: Optional[float] = None,
) -> str:
    """Solve a small semidefinite program.

    Parameters in ``objective`` and ``constraints`` should be provided as matrix
    strings using comma separated values with ``;`` separating rows.  Each
    constraint line is parsed as ``A <= b`` or ``A >= b`` meaning
    ``trace(A @ X) <= b`` etc.
    """

    C = parse_matrix(objective)
    n = C.shape[0]
    X = cp.Variable((n, n), PSD=True)

    constr: List[cp.Constraint] = []
    for line in constraints.splitlines():
        if not line.strip():
            continue
        if "<=" in line:
            A_str, b_str = line.split("<=")
            A = parse_matrix(A_str)
            constr.append(cp.trace(A @ X) <= float(b_str.strip()))
        elif ">=" in line:
            A_str, b_str = line.split(">=")
            A = parse_matrix(A_str)
            constr.append(cp.trace(A @ X) >= float(b_str.strip()))
        elif "==" in line:
            A_str, b_str = line.split("==")
            A = parse_matrix(A_str)
            constr.append(cp.trace(A @ X) == float(b_str.strip()))

    prob = cp.Problem(cp.Minimize(cp.trace(C @ X)), constr)

    solver_name = (method or "SCS").upper()
    if solver_name not in {"SCS"}:
        raise ValueError(f"Unsupported method '{method}'")
    solve_kwargs = {}
    if max_iter is not None:
        solve_kwargs["max_iters"] = int(max_iter)
    if tolerance is not None:
        solve_kwargs["eps"] = float(tolerance)
    prob.solve(solver=solver_name, **solve_kwargs)

    if prob.status == cp.OPTIMAL:
        result = f"Status: {prob.status}\n"
        result += f"X = {X.value}\n"
        result += f"Objective value: {prob.value}"
    else:
        result = f"Status: {prob.status}"
    return result


def solve_conic(
    objective: str,
    constraints: str,
    method: Optional[str] = None,
    max_iter: Optional[int] = None,
    tolerance: Optional[float] = None,
) -> str:
    """Solve a basic conic program using second-order cone constraints."""

    c = parse_vector(objective)
    n = len(c)
    x = cp.Variable(n)

    constr: List[cp.Constraint] = []
    for line in constraints.splitlines():
        if not line.strip():
            continue
        if line.startswith("soc:"):
            # Format: soc:A|b|c -> norm(A@x + b) <= c
            _, rest = line.split(":", 1)
            A_str, b_str, c_str = rest.split("|")
            A = parse_matrix(A_str)
            b = parse_vector(b_str)
            t = float(c_str)
            constr.append(cp.norm(A @ x + b) <= t)
        elif "<=" in line:
            a_str, b_val = line.split("<=")
            a = parse_vector(a_str)
            constr.append(a @ x <= float(b_val.strip()))
        elif ">=" in line:
            a_str, b_val = line.split(">=")
            a = parse_vector(a_str)
            constr.append(a @ x >= float(b_val.strip()))

    prob = cp.Problem(cp.Minimize(c @ x), constr)

    solver_name = (method or "SCS").upper()
    if solver_name not in {"SCS"}:
        raise ValueError(f"Unsupported method '{method}'")
    solve_kwargs = {}
    if max_iter is not None:
        solve_kwargs["max_iters"] = int(max_iter)
    if tolerance is not None:
        solve_kwargs["eps"] = float(tolerance)
    prob.solve(solver=solver_name, **solve_kwargs)

    if prob.status == cp.OPTIMAL:
        result = f"Status: {prob.status}\n"
        result += "\n".join(f"x{i} = {val}" for i, val in enumerate(x.value))
        result += f"\nObjective value: {prob.value}"
    else:
        result = f"Status: {prob.status}"
    return result


def solve_geometric(
    objective: str,
    constraints: str,
    method: Optional[str] = None,
    max_iter: Optional[int] = None,
    tolerance: Optional[float] = None,
) -> str:
    """Solve a geometric program using CVXPY."""

    var_names = set(re.findall(r"[a-zA-Z]\w*", objective))
    for line in constraints.splitlines():
        var_names.update(re.findall(r"[a-zA-Z]\w*", line))

    variables = {name: cp.Variable(pos=True) for name in var_names}

    objective_expr = parse_posynomial(objective, variables)

    constr: List[cp.Constraint] = []
    for line in constraints.splitlines():
        if not line.strip():
            continue
        if "<=" in line:
            lhs, rhs = line.split("<=")
            lhs_expr = parse_posynomial(lhs, variables)
            constr.append(lhs_expr <= float(rhs.strip()))
        elif ">=" in line:
            lhs, rhs = line.split(">=")
            lhs_expr = parse_posynomial(lhs, variables)
            constr.append(lhs_expr >= float(rhs.strip()))
        elif "==" in line:
            lhs, rhs = line.split("==")
            lhs_expr = parse_posynomial(lhs, variables)
            constr.append(lhs_expr == float(rhs.strip()))

    prob = cp.Problem(cp.Minimize(objective_expr), constr)

    solver_name = (method or "ECOS").upper()
    if solver_name not in {"ECOS", "SCS"}:
        raise ValueError(f"Unsupported method '{method}'")
    solve_kwargs = {}
    if max_iter is not None:
        solve_kwargs["max_iters"] = int(max_iter)
    if tolerance is not None:
        if solver_name == "SCS":
            solve_kwargs["eps"] = float(tolerance)
        else:
            solve_kwargs["abstol"] = float(tolerance)
            solve_kwargs["reltol"] = float(tolerance)

    prob.solve(gp=True, solver=solver_name, **solve_kwargs)

    if prob.status == cp.OPTIMAL:
        result = f"Status: {prob.status}\n"
        for name, var in variables.items():
            result += f"{name} = {var.value}\n"
        result += f"Objective value: {prob.value}"
    else:
        result = f"Status: {prob.status}"
    return result
