
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ProcessAnimalRequest(BaseModel):
    sideImageData: str = Field(..., description="Base64-encoded side view image (may include data URL prefix)")
    backImageData: Optional[str] = Field(None, description="Optional back view image (base64)")
    animalType: Optional[str] = Field(None, description="Optional animal type hint")
    notes: Optional[str] = Field(None, description="Optional notes from user")


class ClassificationResult(BaseModel):
    animal_type: Optional[str] = None
    breed: Optional[str] = None
    classification_confidence: Optional[float] = None
    breed_confidence: Optional[float] = None
    atc_scores: Optional[Dict[str, float]] = None
    overall_score: Optional[float] = None
    elite_status: Optional[str] = None


class ProcessImageResponse(BaseModel):
    id: str
    bodyLength: float
    withersHeight: float
    girth: float
    bodyScore: float
    estimatedWeight: Optional[float]
    confidence: float
    processingStatus: str
    timestamp: datetime
    animalType: Optional[str] = None
    notes: Optional[str] = None
    warnings: Optional[List[str]] = None
    rawMeasurements: Optional[Dict[str, float]] = None
    classification: Optional[ClassificationResult] = None


class ResultListResponse(BaseModel):
    results: List[ProcessImageResponse]
    total: int
