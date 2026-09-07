import pytest
pytest.importorskip("cvxpy")

import re
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from solvers import solve_lp, solve_qp


def test_solve_lp_creates_missing_variables():
    result = solve_lp("x", "x + y <= 4\ny >= 1\nx >= 0")
    assert "Status: Optimal" in result
    values = {line.split("=")[0].strip(): float(line.split("=")[1])
              for line in re.findall(r"^\w+ = [\d.-]+", result, re.MULTILINE)}
    assert "y" in values

def test_missing_var_multiple_constraints():
    result = solve_lp("x", "y >= 2\nx + y >= 5\nx >= 0")
    assert "Status: Optimal" in result
    values = {line.split("=")[0].strip(): float(line.split("=")[1])
              for line in re.findall(r"^\w+ = [\d.-]+", result, re.MULTILINE)}
    assert "y" in values

from solvers import solve_sdp, solve_conic, solve_geometric


def test_solve_qp_with_cross_term():
    result = solve_qp("x^2 + 2*x*y + y^2", "x >= 0\ny >= 0")
    assert "Status: optimal" in result


def test_solve_sdp_basic():
    result = solve_sdp("1,0;0,1", "1,0;0,1 >= 1")
    assert "Status: optimal" in result


def test_solve_conic_basic():
    result = solve_conic("1,1", "soc:1,0;0,1|0,0|1")
    assert "Status: optimal" in result


def test_solve_geometric_basic():
    result = solve_geometric("x*y", "x*y >= 1\nx >= 1\ny >= 1")
    assert "Status: optimal" in result
