from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
from sklearn.datasets import make_classification
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
        self.training_history = []

    def _register_defaults(self):
        self.register_model("random_forest", RandomForestClassifier(n_estimators=50, random_state=42))
        self.register_model("svm", SVC(probability=True, gamma="scale", random_state=42))
        self.register_model("logistic_regression", LogisticRegression(max_iter=500))

    def register_model(self, name: str, estimator):
        self.models[name] = MLModel(name, estimator)

    def train_models(self, features=None, targets=None, n_samples=400) -> Dict[str, Tuple[int, int]]:
        features, targets = self._prepare_training_data(features, targets, n_samples)
        summaries = {}
        for name, model in self.models.items():
            model.train(features, targets)
            summaries[name] = (features.shape[0], features.shape[1])
            self.training_history.append(name)
            self.save_model(name)
        return summaries

    def _prepare_training_data(self, features, targets, n_samples):
        if features is None or targets is None:
            features, targets = self._generate_baseline_dataset(n_samples)
        return np.asarray(features, dtype=float), np.asarray(targets)

    def _generate_baseline_dataset(self, n_samples):
        return make_classification(
            n_samples=n_samples,
            n_features=12,
            n_informative=6,
            n_redundant=2,
            random_state=42,
        )

    def get_model(self, name: str) -> MLModel:
        return self.models[name]

    def predict_all(self, features):
        return {
            name: model.predict(features)
            for name, model in self.models.items()
        }

    def save_model(self, name: str) -> Path:
        model = self.get_model(name)
        destination = self.storage_dir / f"{name}.joblib"
        joblib.dump(model.estimator, destination)
        return destination

    def load_model(self, name: str) -> bool:
        destination = self.storage_dir / f"{name}.joblib"
        if not destination.exists():
            return False
        loaded = joblib.load(destination)
        self.models[name].estimator = loaded
        self.models[name].trained = True
        return True

    def save_all(self) -> Dict[str, Path]:
        return {name: self.save_model(name) for name in self.models}

    def load_all(self) -> Dict[str, bool]:
        return {name: self.load_model(name) for name in self.models}
