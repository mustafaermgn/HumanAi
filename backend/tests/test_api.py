import pytest

from backend.app import app as flask_app


@pytest.fixture
def client():
    flask_app.config["TESTING"] = True
    with flask_app.test_client() as client_fixture:
        yield client_fixture


def test_predict_endpoint_success(client):
    response = client.post("/api/predict", json={"code": "def hi():\n    return 1"})
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert "predictions" in data


def test_predict_endpoint_validation(client):
    response = client.post("/api/predict", json={})
    assert response.status_code == 400
    data = response.get_json()
    assert data["success"] is False
