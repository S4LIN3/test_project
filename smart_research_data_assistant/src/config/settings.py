from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class AppSettings:
    app_name: str = "Smart Research and Data Analysis Assistant"
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    vector_db: str = os.getenv("VECTOR_DB", "faiss")


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
VECTORSTORE_DIR = DATA_DIR / "vectorstore"
DB_PATH = DATA_DIR / "app.db"


def ensure_directories() -> None:
    for path in (DATA_DIR, UPLOAD_DIR, VECTORSTORE_DIR):
        path.mkdir(parents=True, exist_ok=True)
