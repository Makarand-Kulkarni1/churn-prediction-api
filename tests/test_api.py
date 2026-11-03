"""API tests for the churn prediction service.

Run from the repo root:  python -m pytest -q
The valid example payload is read from the API's own OpenAPI schema, so these
tests stay correct if you rename fields or change allowed values.
"""
import copy

import pytest
from fastapi.testclient import TestClient

from app import app

client = TestClient(app)
THRESHOLD = 0.35


@pytest.fixture(scope="module")
def valid_payload():
    schema = client.get("/openapi.json").json()
    model = schema["components"]["schemas"]["CustomerData"]
    example = model.get("example") or (model.get("examples") or [None])[0]
    if not example:
        pytest.skip("CustomerData has no example in its schema")
    return example


def test_health_check_returns_ok():
    response = client.get("/")
    assert response.status_code == 200


def test_predict_returns_expected_fields(valid_payload):
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200
    body = response.json()
    for key in ("churn_probability", "churn_prediction", "decision_threshold", "recommendation"):
        assert key in body
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["churn_prediction"] in ("Yes", "No")
    assert body["decision_threshold"] == pytest.approx(THRESHOLD)


def test_prediction_matches_threshold(valid_payload):
    body = client.post("/predict", json=valid_payload).json()
    p = body["churn_probability"]
    if p > THRESHOLD + 1e-6:
        assert body["churn_prediction"] == "Yes"
    elif p < THRESHOLD - 1e-6:
        assert body["churn_prediction"] == "No"


def test_rejects_invalid_category(valid_payload):
    bad = copy.deepcopy(valid_payload)
    bad["Contract"] = "Lifetime"  # not a real Telco contract type
    assert client.post("/predict", json=bad).status_code == 422


def test_rejects_negative_tenure(valid_payload):
    bad = copy.deepcopy(valid_payload)
    bad["tenure"] = -5
    assert client.post("/predict", json=bad).status_code == 422


def test_rejects_missing_field(valid_payload):
    bad = copy.deepcopy(valid_payload)
    bad.pop("MonthlyCharges")
    assert client.post("/predict", json=bad).status_code == 422
