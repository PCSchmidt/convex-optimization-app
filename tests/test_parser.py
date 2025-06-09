pytest.importorskip("cvxpy")

# codex/create-test-client-cases-for-fastapi
import os
import sys
import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# main
from solvers import parse_expression

def test_positive_negative_coefficients():
    assert parse_expression("3x - 2y") == [(3.0, 'x'), (-2.0, 'y')]

def test_variable_suffix_numbers():
    assert parse_expression("4x1 + 5x2 - x3") == [(4.0, 'x1'), (5.0, 'x2'), (-1.0, 'x3')]

def test_expression_in_constraint():
    expr = "x + 2y >= 5"
    assert parse_expression(expr) == [(1.0, 'x'), (2.0, 'y')]
