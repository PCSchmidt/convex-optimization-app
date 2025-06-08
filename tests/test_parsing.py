import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import pytest
from app import parse_expression


def test_parse_linear_expression():
    expr = '3x + 2y - z'
    terms = parse_expression(expr)
    assert terms == [(3.0, 'x'), (2.0, 'y'), (-1.0, 'z')]


def test_parse_quadratic_expression():
    expr = '2x^2 + 3y^2 + 4x + 5y - z^2'
    terms = parse_expression(expr)
    expected = [(2.0, 'x^2'), (3.0, 'y^2'), (4.0, 'x'), (5.0, 'y'), (-1.0, 'z^2')]
    assert terms == expected
