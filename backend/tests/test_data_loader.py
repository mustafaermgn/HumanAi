from pathlib import Path

from backend.utils.data_loader import DataLoader


def test_load_dataset(tmp_path):
    sample = tmp_path / "sample.json"
    sample.write_text('{"code": "def hi():\\n    return 1", "label": 1}', encoding="utf-8")
    loader = DataLoader(tmp_path)
    codes, labels = loader.load_dataset()
    assert codes
    assert labels == [1]
