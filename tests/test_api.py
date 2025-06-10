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


def test_api_semidefinite_program():
    payload = {
        "objective": "1,0;0,1",
        "constraints": "1,0;0,1 >= 1",
    }
    response = client.post("/api/semidefinite_program", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data.get("status") == "ok"
    assert "result" in data


def test_api_conic_program():
    payload = {
        "objective": "1,1",
        "constraints": "soc:1,0;0,1|0,0|1",
    }
    response = client.post("/api/conic_program", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data.get("status") == "ok"
    assert "result" in data


def test_api_geometric_program():
    payload = {
        "objective": "x*y",
        "constraints": "x*y >= 1\nx >= 1\ny >= 1",
    }
    response = client.post("/api/geometric_program", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data.get("status") == "ok"
    assert "result" in data

from app import app as full_app
full_client = TestClient(full_app)

def test_visualize_route_generates_plot():
    response = full_client.post(
        "/visualize",
        data={"objective": "x", "constraints": "x >= 0"},
    )
    assert response.status_code == 200
    assert "data:image/png;base64" in response.text

def test_benchmark_route_displays_table():
    response = client.get("/benchmark")
    assert response.status_code == 200
    assert "<table" in response.text
