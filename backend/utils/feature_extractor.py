class FeatureExtractor:
    def extract_features(self, code: str) -> dict:
        if not isinstance(code, str):
            return {}
        return {"length": len(code)}
