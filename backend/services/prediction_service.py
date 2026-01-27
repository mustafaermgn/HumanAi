from pathlib import Path
from typing import Dict

from backend.models.ml_models import MLModelManager
from backend.utils.data_cleaner import DataCleaner
from backend.utils.feature_extractor import FeatureExtractor


class PredictionService:
    def __init__(self, model_storage: Path | None = None):
        self.cleaner = DataCleaner()
        self.extractor = FeatureExtractor()
        self.manager = MLModelManager(storage_dir=model_storage)
        loaded = self.manager.load_all()
        if not any(loaded.values()):
            self.manager.train_models()
            self.manager.save_all()

    def predict_code(self, code: str) -> Dict[str, float]:
        if not isinstance(code, str) or not code.strip():
            raise ValueError("Kod gereklidir.")

        if not self.cleaner.validate_code(code):
            raise ValueError("Kod gecerli degil.")

        cleaned = self.cleaner.clean_code(code)
        features = self.extractor.extract_features(cleaned)
        vector = self.extractor.vectorize(features)
        raw_predictions = self.manager.predict_all([vector])
        return {name: float(pred[0]) for name, pred in raw_predictions.items()}
