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
