"""
History Endpoints.
"""

from fastapi import APIRouter
from typing import List

from backend.app.schemas.schemas import AnalysisHistoryItem
from backend.app.database.db import get_history, clear_history

router = APIRouter()

@router.get("/history", response_model=List[AnalysisHistoryItem])
async def fetch_history(limit: int = 50):
    return get_history(limit=limit)

@router.delete("/history")
async def delete_history():
    clear_history()
    return {"status": "success", "message": "History cleared"}
