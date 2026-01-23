from utils.feature_extractor import FeatureExtractor


def test_extract_features_includes_text_and_numeric():
    extractor = FeatureExtractor()
    features = extractor.extract_features("a1+\nBB")
    assert "line_count" in features
    assert "avg_line_length" in features
    assert "alpha_ratio" in features
    assert "digit_ratio" in features
    assert "symbol_ratio" in features


def test_extract_features_length_matches_input():
    extractor = FeatureExtractor()
    code = "abc"
    assert extractor.extract_features(code)["length"] == len(code)
