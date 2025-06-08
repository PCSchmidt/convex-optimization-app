from __future__ import annotations

from typing import Dict, List, Tuple

import cvxpy as cp
import pulp
import re


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


def solve_lp(objective: str, constraints: str) -> str:
    """Solve a linear program using PuLP.

    Args:
        objective: Objective function as a string, e.g. ``"3x + 2y"``.
        constraints: Newline separated constraint expressions.

    Returns:
        Human readable string describing solver status and variable values.
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

    prob.solve()

    status = pulp.LpStatus[prob.status]
    if status == "Optimal":
        result = f"Status: {status}\n"
        for var in prob.variables():
            result += f"{var.name} = {var.varValue}\n"
        result += f"Objective value: {pulp.value(prob.objective)}"
    else:
        result = f"Status: {status}"
    return result


def solve_qp(objective: str, constraints: str) -> str:
    """Solve a quadratic program using CVXPY.

    Args:
        objective: Quadratic objective as a string with ``^2`` for squared terms.
        constraints: Newline separated constraint expressions.

    Returns:
        Human readable string describing solver status and solution values.
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
    prob.solve()

    if prob.status == cp.OPTIMAL:
        result = f"Status: {prob.status}\n"
        for var_name, var in variables.items():
            result += f"{var_name} = {var.value}\n"
        result += f"Objective value: {prob.value}"
    else:
        result = f"Status: {prob.status}"
    return result
