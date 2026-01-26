import argparse
from pathlib import Path

from models.ml_models import MLModelManager
from utils.data_cleaner import DataCleaner
from utils.feature_extractor import FeatureExtractor
from utils.data_loader import DataLoader


def run_training(data_dir: Path, model_storage: Path):
    loader = DataLoader(data_dir)
    cleaner = DataCleaner()
    extractor = FeatureExtractor()
    manager = MLModelManager(storage_dir=model_storage)

    codes, labels = loader.load_dataset()
    cleaned = [cleaner.clean_code(code) for code in codes if cleaner.validate_code(code)]
    features = [extractor.extract_features(code) for code in cleaned]
    manager.train_models(features, labels[: len(features)])
    manager.save_all()
    return manager.training_history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    args = parser.parse_args()
    run_training(args.data_dir, args.model_dir)


if __name__ == "__main__":
    main()
