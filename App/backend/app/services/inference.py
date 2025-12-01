
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
        vetra_core.load_models_and_scaling()
        self._loaded = True

    def _map_result(
        self,
        result: Dict[str, Any],
        animal_type: Optional[str],
        notes: Optional[str],
    ) -> ProcessImageResponse:
        cm = result.get("cm") or result.get("measurements_cm") or {}
        warnings = result.get("warnings") or []
        classification_raw = result.get("classification") or {}

        body_length = cm.get("oblique_body_length") or cm.get("body_length") or 0.0
        withers_height = cm.get("withers_height") or 0.0
        girth = cm.get("girth_length") or cm.get("girth") or 0.0
        body_score = cm.get("body_score") or classification_raw.get("overall_score") or 0.0
        estimated_weight = (
            cm.get("estimated_weight")
            or classification_raw.get("estimated_weight")
            or cm.get("weight")
        )

        confidence = (
            classification_raw.get("classification_confidence")
            or classification_raw.get("breed_confidence")
            or cm.get("confidence")
            or 90.0
        )

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
            animalType=animal_type or classification_raw.get("animal_type"),
            notes=notes,
            warnings=warnings,
            rawMeasurements=cm,
            classification=classification,
        )

    def process_animal(
        self,
        side_image_b64: str,
        back_image_b64: Optional[str] = None,
        animal_type: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> ProcessImageResponse:
        if not self._loaded:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Inference service not initialized",
            )

        side_path = vetra_core.decode_and_save(side_image_b64)
        back_path = vetra_core.decode_and_save(back_image_b64) if back_image_b64 else None

        try:
            if back_path:
                result = vetra_core.run_full_vetra(side_path, back_path)
                # Normalize structure from complete pipeline
                side_cm = (result.get("side_view_analysis") or {}).get("measurements_cm") or {}
                back_cm = (result.get("back_view_analysis") or {}).get("measurements_cm") or {}
                classification = result.get("classification") or {}
                warnings = []
                merged_cm = {**side_cm, **back_cm}
                merged_result = {
                    "cm": merged_cm,
                    "warnings": warnings,
                    "classification": classification,
                }
                return self._map_result(merged_result, animal_type, notes)

            # Side-only path
            result = vetra_core.process_image_file(side_path, view_type="side", animal_type=animal_type)
            return self._map_result(result, animal_type, notes)

        except HTTPException:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Failed to process image(s): {exc}",
            ) from exc


inference_service = InferenceService()
