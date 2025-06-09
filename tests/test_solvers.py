import re
import os
import sys
import pytest

pytest.importorskip("cvxpy")

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from solvers import solve_lp


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
