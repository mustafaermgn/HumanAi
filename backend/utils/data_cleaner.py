class DataCleaner:
    def clean_code(self, code: str) -> str:
        if not isinstance(code, str):
            return ""
        return code.strip()
