"""
Lightweight wrapper around the notebook pipeline.

This module is intentionally minimal: it tries to import the existing notebook
logic if it has been exported to a Python module, and otherwise falls back to a
deterministic mock so the API stays available during development. The goal is
to preserve the original measurement logic without rewriting it in this file.
"""
from __future__ import annotations

import base64
import os
import random
import tempfile
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np

# Try to import any existing exported module from the notebook (optional)
try:
    from app_notebook_export import (  # type: ignore
        calibrated_factors,
        final_measurement_pipeline,
        run_high_accuracy_demo,
        side_model,
        back_model,
    )
except Exception:  # pragma: no cover - fallback if notebook export is absent
    calibrated_factors = None
    final_measurement_pipeline = None
    run_high_accuracy_demo = None
    side_model = None
    back_model = None


class VetraCore:
    """
    Wrapper to run the measurement pipeline. If the real notebook code is
    available (exported as a module), it will be used. Otherwise, a controlled
    mock is returned so the API and UI can still be exercised.
    """

    def __init__(self) -> None:
        self._models_loaded = False

    def load(self) -> None:
        # If models from notebook are available, mark as loaded
        if side_model is not None and back_model is not None:
            self._models_loaded = True
        else:
            # Nothing to load for mock path
            self._models_loaded = True

    def process_image_file(
        self, image_path: str, view_type: str = "side", animal_type: Optional[str] = None
    ) -> Dict[str, Any]:
        if final_measurement_pipeline:
            pixel, cm_vals, warnings = final_measurement_pipeline(
                image_path, view_type, calibrated_factors
            )
            classification = None
            if run_high_accuracy_demo and cm_vals:
                side_meas = cm_vals if view_type == "side" else {}
                back_meas = cm_vals if view_type == "back" else {}
                classification = run_high_accuracy_demo(side_meas, back_meas)

            return {
                "measurements_cm": cm_vals or {},
                "pixel_measurements": pixel or {},
                "warnings": warnings or [],
                "classification": classification,
            }

        # Fallback deterministic mock for development
        rng = random.Random(os.path.basename(image_path))
        body_score = rng.randint(4, 7)
        body_length = 140 + rng.random() * 20
        withers_height = 120 + rng.random() * 15
        girth = 165 + rng.random() * 20
        estimated_weight = 350 + rng.random() * 80
        confidence = 85 + rng.random() * 10

        return {
            "measurements_cm": {
                "oblique_body_length": round(body_length, 2),
                "withers_height": round(withers_height, 2),
                "girth_length": round(girth, 2),
                "body_score": body_score,
            },
            "pixel_measurements": {},
            "warnings": [],
            "classification": {
                "animal_type": animal_type or "cattle",
                "breed": "mock-breed",
                "classification_confidence": confidence,
                "breed_confidence": confidence - 2,
                "overall_score": body_score,
                "elite_status": "monitor",
            },
        }

    def decode_and_save(self, image_b64: str, suffix: str = ".jpg") -> str:
        # Strip data URL prefix if present
        if "," in image_b64:
            image_b64 = image_b64.split(",", 1)[1]
        data = base64.b64decode(image_b64)
        fd, path = tempfile.mkstemp(suffix=suffix)
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        return path


vetra_core = VetraCore()

