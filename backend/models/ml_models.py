from pathlib import Path
from typing import Dict

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


class MLModel:
    def __init__(self, name: str, estimator):
        self.name = name
        self.estimator = estimator
        self.trained = False

    def train(self, features, targets):
        array = self._ensure_2d(features)
        self.estimator.fit(array, targets)
        self.trained = True

    def predict(self, features):
        array = self._ensure_2d(features)
        return self.estimator.predict(array)

    def _ensure_2d(self, data):
        array = np.asarray(data, dtype=float)
        return np.atleast_2d(array)


class MLModelManager:
    def __init__(self, storage_dir: Path = None):
        self.storage_dir = storage_dir or Path(__file__).parent / "saved_models"
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.models: Dict[str, MLModel] = {}
        self._register_defaults()

    def _register_defaults(self):
        self.register_model("random_forest", RandomForestClassifier(n_estimators=50, random_state=42))
        self.register_model("svm", SVC(probability=True, gamma="scale", random_state=42))
        self.register_model("logistic_regression", LogisticRegression(max_iter=500))

    def register_model(self, name: str, estimator):
        self.models[name] = MLModel(name, estimator)

    def get_model(self, name: str) -> MLModel:
        return self.models[name]

    def predict_all(self, features):
        return {
            name: model.predict(features)
            for name, model in self.models.items()
        }
