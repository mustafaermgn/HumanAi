from utils.data_cleaner import DataCleaner


def test_clean_code_strips_and_normalizes():
    cleaner = DataCleaner()
    code = "  a\t\tb  \n\n  c "
    assert cleaner.clean_code(code) == "a b\nc"


def test_validate_code_rejects_short_input():
    cleaner = DataCleaner()
    assert cleaner.validate_code("short") is False


def test_validate_code_accepts_code_like_input():
    cleaner = DataCleaner()
    assert cleaner.validate_code("def x():\n    return 1") is True
