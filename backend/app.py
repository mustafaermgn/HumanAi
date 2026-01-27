from pathlib import Path
from typing import Dict

from flask import Flask, request, jsonify
from flask_cors import CORS
from pydantic import BaseModel, ValidationError

from config import Config
from backend.services.prediction_service import PredictionService


class PredictRequest(BaseModel):
    code: str


class PredictResponse(BaseModel):
    success: bool
    predictions: Dict[str, float]
    code_length: int


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    CORS(app, resources={r"/api/*": {"origins": app.config.get("CORS_ORIGINS")}})

    @app.route("/api/health", methods=["GET"])
    def health_check():
        return jsonify({"status": "healthy", "message": "API is running"})

    @app.route("/api/predict", methods=["POST"])
    def predict_code():
        payload = request.get_json() or {}
        try:
            request_model = PredictRequest(**payload)
        except ValidationError as exc:
            return jsonify({"success": False, "errors": exc.errors()}), 400

        try:
            results = prediction_service.predict_code(request_model.code)
        except ValueError as exc:
            return jsonify({"success": False, "errors": [str(exc)]}), 422

        response = PredictResponse(
            success=True,
            predictions=results,
            code_length=len(request_model.code.strip()),
        )
        return jsonify(response.dict())

    return app


model_storage_path = Path(__file__).parent / "models" / "saved_models"
prediction_service = PredictionService(model_storage=model_storage_path)

app = create_app()


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
