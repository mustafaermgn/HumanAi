import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///code_analysis.db")
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*")
