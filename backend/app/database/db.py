"""
SQLite Database Operations for Analysis History and Experiment Records.
"""

import sqlite3
import json
import os
from typing import List, Dict, Any, Optional
from datetime import datetime

from backend.app.core.config import settings

def get_db_connection() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(settings.DB_PATH), exist_ok=True)
    conn = sqlite3.connect(settings.DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                model_used TEXT NOT NULL,
                tumor_detected BOOLEAN NOT NULL,
                tumor_pixel_count INTEGER NOT NULL,
                tumor_area_percentage REAL NOT NULL,
                classification_label TEXT,
                confidence REAL,
                has_uncertainty BOOLEAN DEFAULT 0,
                mean_uncertainty REAL DEFAULT 0.0,
                degradation_type TEXT DEFAULT 'none',
                severity INTEGER DEFAULT 0,
                details_json TEXT
            )
        """)
    conn.close()

def save_analysis(
    filename: str,
    model_used: str,
    tumor_detected: bool,
    tumor_pixel_count: int,
    tumor_area_percentage: float,
    classification_label: Optional[str] = None,
    confidence: Optional[float] = None,
    has_uncertainty: bool = False,
    mean_uncertainty: float = 0.0,
    degradation_type: str = "none",
    severity: int = 0,
    details: Optional[Dict[str, Any]] = None
) -> int:
    conn = get_db_connection()
    timestamp = datetime.utcnow().isoformat() + "Z"
    details_str = json.dumps(details or {})
    
    with conn:
        cursor = conn.execute("""
            INSERT INTO analyses (
                filename, timestamp, model_used, tumor_detected, tumor_pixel_count,
                tumor_area_percentage, classification_label, confidence,
                has_uncertainty, mean_uncertainty, degradation_type, severity, details_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            filename, timestamp, model_used, int(tumor_detected), tumor_pixel_count,
            tumor_area_percentage, classification_label, confidence,
            int(has_uncertainty), mean_uncertainty, degradation_type, severity, details_str
        ))
        analysis_id = cursor.lastrowid
    conn.close()
    return analysis_id

def get_history(limit: int = 50) -> List[Dict[str, Any]]:
    conn = get_db_connection()
    rows = conn.execute("""
        SELECT * FROM analyses ORDER BY id DESC LIMIT ?
    """, (limit,)).fetchall()
    conn.close()

    results = []
    for r in rows:
        item = dict(r)
        item["tumor_detected"] = bool(item["tumor_detected"])
        item["has_uncertainty"] = bool(item["has_uncertainty"])
        if item.get("details_json"):
            try:
                item["details"] = json.loads(item["details_json"])
            except Exception:
                item["details"] = {}
        results.append(item)
    return results

def clear_history():
    conn = get_db_connection()
    with conn:
        conn.execute("DELETE FROM analyses")
    conn.close()
