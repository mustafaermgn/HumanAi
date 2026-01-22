import re


class DataCleaner:
    def clean_code(self, code: str) -> str:
        if not isinstance(code, str):
            return ""
        cleaned = code.strip()
        cleaned = self.normalize_whitespace(cleaned)
        cleaned = self.compact_blank_lines(cleaned)
        return cleaned

    def normalize_whitespace(self, code: str) -> str:
        return re.sub(r"[ \t]+", " ", code)

    def compact_blank_lines(self, code: str) -> str:
        return re.sub(r"\n\s*\n+", "\n", code)

    def validate_code(self, code: str) -> bool:
        if not isinstance(code, str):
            return False
        if len(code.strip()) < 10:
            return False
        tokens = {"def", "class", "import", "return", "if", "for", "while", "{"}
        return any(token in code for token in tokens)
