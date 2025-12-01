from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import HTTPException, status

from app.models.vetra_core import vetra_core
from app.schemas import ClassificationResult, ProcessImageResponse


class InferenceService:
    def __init__(self) -> None:
        self._loaded = False

    def load(self) -> None:
        vetra_core.load()
        self._loaded = True

    def _map_result(
        self,
        result: Dict[str, Any],
        animal_type: Optional[str],
        notes: Optional[str],
    ) -> ProcessImageResponse:
        cm = result.get("measurements_cm") or {}
        warnings = result.get("warnings") or []
        classification_raw = result.get("classification") or {}

        # Extract values with fallbacks
        body_length = cm.get("oblique_body_length") or cm.get("body_length") or 0.0
        withers_height = cm.get("withers_height") or 0.0
        girth = cm.get("girth_length") or cm.get("girth") or 0.0
        body_score = cm.get("body_score") or classification_raw.get("overall_score") or 0.0
        estimated_weight = cm.get("estimated_weight") or classification_raw.get("estimated_weight")
        confidence = classification_raw.get("classification_confidence") or 90.0

        classification = None
        if classification_raw:
            classification = ClassificationResult(
                animal_type=classification_raw.get("animal_type"),
                breed=classification_raw.get("breed"),
                classification_confidence=classification_raw.get("classification_confidence"),
                breed_confidence=classification_raw.get("breed_confidence"),
                atc_scores=classification_raw.get("atc_scores"),
                overall_score=classification_raw.get("overall_score"),
                elite_status=classification_raw.get("elite_status"),
            )

        return ProcessImageResponse(
            id=str(uuid.uuid4()),
            bodyLength=float(body_length),
            withersHeight=float(withers_height),
            girth=float(girth),
            bodyScore=float(body_score),
            estimatedWeight=float(estimated_weight) if estimated_weight is not None else None,
            confidence=float(confidence),
            processingStatus="completed",
            timestamp=datetime.utcnow(),
            animalType=animal_type,
            notes=notes,
            warnings=warnings,
            rawMeasurements=cm,
            classification=classification,
        )

    def process_single_image(
        self,
        image_b64: str,
        view_type: str = "side",
        animal_type: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> ProcessImageResponse:
        if not self._loaded:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Inference service not initialized",
            )

        image_path = vetra_core.decode_and_save(image_b64)
        try:
            result = vetra_core.process_image_file(image_path, view_type=view_type, animal_type=animal_type)
            return self._map_result(result, animal_type, notes)
        except Exception as exc:  # pragma: no cover - defensive
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Failed to process image: {exc}",
            ) from exc


inference_service = InferenceService()

