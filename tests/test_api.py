# codex/add-json-endpoints-with-fastapi
from fastapi.testclient import TestClient
from app import create_app

client = TestClient(create_app())

def test_api_linear_program():
    response = client.post("/api/linear_program", json={
        "objective": "x + y",
        "constraints": "x + y >= 1\nx >= 0\ny >= 0"
    })
    assert response.status_code == 200
    data = response.json()
    assert data["status"].lower() == "optimal"
    assert "objective_value" in data
    assert "variables" in data


def test_api_quadratic_program():
    response = client.post("/api/quadratic_program", json={
        "objective": "x^2 + y^2",
        "constraints": "x + y >= 1"
    })
    assert response.status_code == 200
    data = response.json()
    assert data["status"].lower() == "optimal"
    assert "objective_value" in data
    assert "variables" in data

import pytest
pytest.importorskip("httpx")
pytest.importorskip("cvxpy")
pytest.importorskip("fastapi")

import os
import sys
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
    assert data.get("status") == "ok"
    assert "result" in data


def test_api_quadratic_program():
    payload = {
        "objective": "x^2 + 1",
        "constraints": "x >= 0",
    }
    response = client.post("/api/quadratic_program", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data.get("status") == "ok"
    assert "result" in data
# main
