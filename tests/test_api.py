import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_loaded": True}


@pytest.mark.parametrize(
    "title,desc,expected",
    [
        ("Lakers beat Celtics in overtime thriller", "LeBron scored 40 points in the win.", "Sports"),
        ("Oil prices drop as markets rally", "Wall Street stocks closed higher on Friday.", "Business"),
        ("Scientists unveil new quantum computer chip", "The processor uses fewer qubits.", "Sci/Tech"),
        ("UN Security Council meets over border conflict", "Diplomats from several nations called for a ceasefire.", "World"),
    ],
)
def test_predict_categories(client, title, desc, expected):
    r = client.post("/predict", json={"title": title, "description": desc})
    assert r.status_code == 200
    body = r.json()
    assert body["category"] == expected
    assert 0 < body["confidence"] <= 1
    assert abs(sum(body["probabilities"].values()) - 1) < 1e-2


def test_predict_title_only(client):
    assert client.post("/predict", json={"title": "Football club signs new striker"}).status_code == 200


def test_validation_error(client):
    assert client.post("/predict", json={"title": ""}).status_code == 422
    assert client.post("/predict", json={}).status_code == 422


def test_batch(client):
    payload = {"articles": [{"title": "Team wins championship"}, {"title": "Intel launches new processor"}]}
    r = client.post("/predict/batch", json=payload)
    assert r.status_code == 200
    assert len(r.json()["predictions"]) == 2
