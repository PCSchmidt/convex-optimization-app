import os
import sys
import pytest

pytest.importorskip("httpx")
pytest.importorskip("cvxpy")
pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from app import create_app

app = create_app()
client = TestClient(app)


def test_api_linear_program():
    payload = {
        "objective": "x",
        "constraints": "x >= 1",
    }
    response = client.post("/api/linear_program", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "result" in data


def test_api_quadratic_program():
    payload = {
        "objective": "x^2 + 1",
        "constraints": "x >= 0",
    }
    response = client.post("/api/quadratic_program", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "result" in data
