import json
from pathlib import Path
from typing import Iterable, Tuple


class DataLoader:
    def __init__(self, root: Path):
        self.root = root

    def load_json_files(self) -> Iterable[str]:
        for path in self.root.glob("*.json"):
            yield path.read_text(encoding="utf-8")

    def load_dataset(self) -> Tuple[list[str], list[int]]:
        codes = []
        labels = []
        for text in self.load_json_files():
            entry = json.loads(text)
            codes.append(entry.get("code", ""))
            labels.append(entry.get("label", 0))
        return codes, labels
