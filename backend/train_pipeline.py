import argparse
from pathlib import Path

from backend.models.ml_models import MLModelManager
from backend.utils.data_cleaner import DataCleaner
from backend.utils.feature_extractor import FeatureExtractor
from backend.utils.data_loader import DataLoader


def run_training(data_dir: Path, model_storage: Path):
    loader = DataLoader(data_dir)
    cleaner = DataCleaner()
    extractor = FeatureExtractor()
    manager = MLModelManager(storage_dir=model_storage)

    codes, labels = loader.load_dataset()
    pairs = []
    for code, label in zip(codes, labels):
        if cleaner.validate_code(code):
            pairs.append((cleaner.clean_code(code), label))

    if not pairs:
        raise ValueError("Gecerli kod bulunamadi.")

    vectors = [
        extractor.vectorize(extractor.extract_features(clean_code)) for clean_code, _ in pairs
    ]
    manager.train_models(vectors, [label for _, label in pairs])
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
