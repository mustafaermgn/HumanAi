import pytest

from backend.services.prediction_service import PredictionService


def test_prediction_service_rejects_invalid_code(tmp_path):
    service = PredictionService(model_storage=tmp_path)
    with pytest.raises(ValueError):
        service.predict_code("short")


def test_prediction_service_returns_predictions(tmp_path):
    service = PredictionService(model_storage=tmp_path)
    result = service.predict_code("def hi():\n    return 1")
    assert isinstance(result, dict)
    assert "random_forest" in result
