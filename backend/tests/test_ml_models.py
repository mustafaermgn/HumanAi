from models.ml_models import MLModelManager


def test_train_models_records_history(tmp_path):
    manager = MLModelManager(storage_dir=tmp_path)
    summary = manager.train_models(n_samples=30)
    assert summary
    assert len(manager.training_history) == len(manager.models)


def test_load_model_restores_trained_flag(tmp_path):
    manager = MLModelManager(storage_dir=tmp_path)
    manager.train_models(n_samples=20)
    manager.models["random_forest"].trained = False
    assert manager.load_model("random_forest")
    assert manager.models["random_forest"].trained


def test_predict_all_returns_arrays():
    manager = MLModelManager()
    manager.train_models(n_samples=20)
    output = manager.predict_all([[1.0] * 12])
    assert isinstance(output, dict)
    assert "random_forest" in output
