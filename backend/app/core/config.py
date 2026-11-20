"""
Backend Core Configuration Settings.
"""

import os
from pydantic import BaseModel

class Settings(BaseModel):
    PROJECT_NAME: str = "Brain Tumor Segmentation & Research Platform"
    API_V1_PREFIX: str = "/api"
    BASE_DIR: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    ARTIFACTS_DIR: str = os.path.join(BASE_DIR, "artifacts")
    DATA_DIR: str = os.path.join(BASE_DIR, "cv", "data", "brisc2025")
    DB_PATH: str = os.path.join(ARTIFACTS_DIR, "analysis_history.db")
    MAX_UPLOAD_SIZE_MB: int = 25
    ALLOWED_EXTENSIONS: tuple = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

settings = Settings()
