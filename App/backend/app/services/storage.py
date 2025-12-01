from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime
from typing import List

from app.core.config import get_settings
from app.schemas import ProcessImageResponse

settings = get_settings()


def _get_connection() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(settings.db_path), exist_ok=True)
    conn = sqlite3.connect(settings.db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS results (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                animal_type TEXT,
                body_length REAL,
                withers_height REAL,
                girth REAL,
                body_score REAL,
                estimated_weight REAL,
                confidence REAL,
                processing_status TEXT,
                notes TEXT,
                warnings TEXT,
                raw_measurements TEXT,
                classification TEXT
            )
            """
        )
        conn.commit()


def save_result(result: ProcessImageResponse) -> None:
    with _get_connection() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO results (
                id, timestamp, animal_type, body_length, withers_height, girth,
                body_score, estimated_weight, confidence, processing_status,
                notes, warnings, raw_measurements, classification
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                result.id,
                result.timestamp.isoformat(),
                result.animalType,
                result.bodyLength,
                result.withersHeight,
                result.girth,
                result.bodyScore,
                result.estimatedWeight,
                result.confidence,
                result.processingStatus,
                result.notes,
                json.dumps(result.warnings or []),
                json.dumps(result.rawMeasurements or {}),
                json.dumps(result.classification.dict() if result.classification else {}),
            ),
        )
        conn.commit()


def list_results(limit: int = 50) -> List[ProcessImageResponse]:
    with _get_connection() as conn:
        rows = conn.execute(
            """
            SELECT * FROM results
            ORDER BY datetime(timestamp) DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    results: List[ProcessImageResponse] = []
    for row in rows:
        results.append(
            ProcessImageResponse(
                id=row["id"],
                timestamp=datetime.fromisoformat(row["timestamp"]),
                animalType=row["animal_type"],
                bodyLength=row["body_length"],
                withersHeight=row["withers_height"],
                girth=row["girth"],
                bodyScore=row["body_score"],
                estimatedWeight=row["estimated_weight"],
                confidence=row["confidence"],
                processingStatus=row["processing_status"],
                notes=row["notes"],
                warnings=json.loads(row["warnings"] or "[]"),
                rawMeasurements=json.loads(row["raw_measurements"] or "{}"),
                classification=json.loads(row["classification"] or "{}"),
            )
        )
    return results

