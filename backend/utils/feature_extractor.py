class FeatureExtractor:
    FEATURE_KEYS = [
        "length",
        "line_count",
        "avg_line_length",
        "alpha_ratio",
        "digit_ratio",
        "symbol_ratio",
    ]

    def extract_features(self, code: str) -> dict:
        if not isinstance(code, str):
            return {}
        features = {"length": len(code)}
        features.update(self.text_features(code))
        features.update(self.numeric_features(code))
        return features

    def text_features(self, code: str) -> dict:
        lines = code.splitlines()
        line_count = max(len(lines), 1)
        avg_line_length = sum(len(line) for line in lines) / line_count
        alpha_chars = sum(char.isalpha() for char in code)
        alpha_ratio = alpha_chars / max(len(code), 1)
        return {
            "line_count": line_count,
            "avg_line_length": avg_line_length,
            "alpha_ratio": alpha_ratio,
        }

    def numeric_features(self, code: str) -> dict:
        digit_count = sum(char.isdigit() for char in code)
        symbol_count = sum(not char.isalnum() and not char.isspace() for char in code)
        length = max(len(code), 1)
        return {
            "digit_ratio": digit_count / length,
            "symbol_ratio": symbol_count / length,
        }

    def vectorize(self, feature_map: dict) -> list[float]:
        return [float(feature_map.get(key, 0.0)) for key in self.FEATURE_KEYS]
