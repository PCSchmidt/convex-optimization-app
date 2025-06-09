"""Utility functions for parsing polynomial and linear expressions using sympy."""

from __future__ import annotations

from typing import Dict, Tuple, List

import cvxpy as cp
import numpy as np

import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
)

TRANSFORMS = standard_transformations + (implicit_multiplication_application,)


def parse_polynomial(expr: str) -> Tuple[List[sp.Symbol], sp.Poly]:
    """Return sympy symbols and polynomial for a given expression string."""
    expr = expr.replace("^", "**")
    sym_expr = parse_expr(expr, transformations=TRANSFORMS)
    symbols = sorted(sym_expr.free_symbols, key=lambda s: s.name)
    poly = sp.Poly(sym_expr, *symbols)
    return symbols, poly


def extract_linear_coeffs(poly: sp.Poly) -> Dict[str, float]:
    """Extract linear coefficients from a polynomial."""
    coeffs: Dict[str, float] = {}
    for exps, coef in poly.as_dict().items():
        if sum(exps) == 1:
            for sym, exp in zip(poly.gens, exps):
                if exp:
                    coeffs[str(sym)] = coeffs.get(str(sym), 0.0) + float(coef)
        elif sum(exps) > 1:
            raise ValueError("Nonlinear term found in linear expression")
    return coeffs


def extract_quadratic_terms(poly: sp.Poly) -> Tuple[Dict[Tuple[str, str], float], Dict[str, float]]:
    """Return quadratic and linear coefficients from a polynomial.

    Quadratic terms are returned as a dict mapping variable pairs to coefficients
    (e.g. ("x", "y") -> coef for x*y). Linear coefficients are returned in a
    separate dict.
    """
    quad: Dict[Tuple[str, str], float] = {}
    lin: Dict[str, float] = {}
    for exps, coef in poly.as_dict().items():
        deg = sum(exps)
        if deg == 1:
            for sym, exp in zip(poly.gens, exps):
                if exp:
                    lin[str(sym)] = lin.get(str(sym), 0.0) + float(coef)
        elif deg == 2:
            vars_in_term = [str(sym) for sym, exp in zip(poly.gens, exps) if exp]
            if len(vars_in_term) == 1:
                pair = (vars_in_term[0], vars_in_term[0])
            else:
                pair = tuple(sorted(vars_in_term))  # symmetric pair
            quad[pair] = quad.get(pair, 0.0) + float(coef)
        elif deg == 0:
            continue
        else:
            raise ValueError("Only quadratic expressions are supported")
    return quad, lin


def parse_matrix(text: str) -> np.ndarray:
    """Parse a semicolon-separated matrix string into a numpy array.

    Each row in ``text`` should be separated by ``;`` and values within a row
    should be comma separated, for example ``"1,0;0,1"`` for the 2x2 identity
    matrix.
    """
    rows = [row.strip() for row in text.strip().split(";") if row.strip()]
    return np.array([[float(v) for v in row.split(",") if v] for row in rows])


def parse_vector(text: str) -> np.ndarray:
    """Parse a comma separated vector string into a numpy array."""
    return np.array([float(v) for v in text.strip().split(",") if v])


def parse_posynomial(expr: str, variables: Dict[str, cp.Variable]) -> cp.Expression:
    """Parse a (po)nomial expression string using CVXPY variables.

    The expression may contain ``^`` for exponentiation and products using ``*``.
    Variables present in ``variables`` can be used directly. The resulting
    expression is created with ``eval`` in a restricted namespace.
    """
    expr_py = expr.replace("^", "**")
    return eval(expr_py, {"__builtins__": {}}, variables)
