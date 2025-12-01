
"""
Wrapper around the notebook-exported pipeline with a mock fallback.

If `app_notebook_export.py` exists (generated from the notebook), we load and
use it. Otherwise, we return deterministic mock measurements so the app stays
usable.
"""
from __future__ import annotations

import base64
import builtins
import logging
import os
import random
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _ensure_fake_get_ipython() -> None:
    """Provide a no-op get_ipython so notebook magics don't crash imports."""
    if not hasattr(builtins, "get_ipython"):
        logger.info("Defining fake get_ipython stub for notebook import...")

        class _DummyShell:
            def run_line_magic(self, *args, **kwargs):
                return None

            def run_cell_magic(self, *args, **kwargs):
                return None

            def run_line(self, *args, **kwargs):
                return None

        def _fake_get_ipython():  # pragma: no cover - simple stub
            return _DummyShell()

        builtins.get_ipython = _fake_get_ipython  # type: ignore[attr-defined]


_ensure_fake_get_ipython()

# Try to import the exported notebook module (placed alongside this file)
try:  # pragma: no cover - import guarded
    from . import app_notebook_export as nb  # type: ignore
    NOTEBOOK_AVAILABLE = True
except Exception as e:  # pragma: no cover - fallback path
    nb = None
    NOTEBOOK_AVAILABLE = False
    logger.warning("Notebook export not available, using mock pipeline. Error: %s", e)

PROJECT_ROOT = Path(__file__).resolve().parents[3]  # .../App


class VetraCore:
    """Wrapper to run measurement/classification via the notebook or a mock."""

    def __init__(self) -> None:
        self._models_loaded = False
        self.scaling_factors = None

    def load_models_and_scaling(self) -> None:
        if not NOTEBOOK_AVAILABLE:
            logger.warning("Notebook not available; using mock models.")
            self._models_loaded = True
            return

        # Set working dir so notebook relative paths resolve (runs/, data/, etc.)
        try:
            os.chdir(PROJECT_ROOT)
        except Exception as exc:  # pragma: no cover
            logger.warning("Could not change cwd to project root: %s", exc)

        # If notebook provided an explicit loader, call it
        loader = getattr(nb, "load_models_and_scaling", None)
        if callable(loader):
            try:
                loader()
            except Exception as exc:  # pragma: no cover
                logger.warning("load_models_and_scaling in notebook failed: %s", exc)

        self.scaling_factors = getattr(nb, "calibrated_factors", None)
        self._models_loaded = True
        logger.info("Notebook models/scaling ready (or already loaded via import).")

    # -------- Mock helpers --------
    def _make_mock_measurements(self, image_path: str, view_type: str) -> Dict[str, Any]:
        rng = random.Random(os.path.basename(image_path) + view_type)
        body_score = rng.randint(4, 7)
        body_length = 140 + rng.random() * 20
        withers_height = 120 + rng.random() * 15
        girth = 165 + rng.random() * 20
        estimated_weight = 350 + rng.random() * 80
        confidence = 85 + rng.random() * 10

        cm_vals = {
            "oblique_body_length": round(body_length, 2),
            "withers_height": round(withers_height, 2),
            "girth_length": round(girth, 2),
            "body_score": body_score,
            "estimated_weight": round(estimated_weight, 2),
            "confidence": round(confidence, 2),
        }

        classification = {
            "animal_type": "cattle",
            "breed": "mock-breed",
            "classification_confidence": confidence,
            "breed_confidence": confidence - 2,
            "overall_score": body_score,
            "elite_status": "monitor",
        }

        return {
            "cm": cm_vals,
            "pixel": {},
            "warnings": [],
            "classification": classification,
        }

    def _make_mock_full_result(self, side_image_path: str, back_image_path: Optional[str]) -> Dict[str, Any]:
        side = self._make_mock_measurements(side_image_path, "side")
        back = self._make_mock_measurements(back_image_path or side_image_path, "back") if back_image_path else None
        return {
            "side_view_analysis": {"measurements_cm": side.get("cm", {})},
            "back_view_analysis": {"measurements_cm": back.get("cm", {})} if back else None,
            "classification": side.get("classification"),
            "timestamp": None,
        }

    # -------- Real pipeline wrappers --------
    def run_measurement(self, image_path: str, view_type: str = "side") -> Dict[str, Any]:
        if NOTEBOOK_AVAILABLE and hasattr(nb, "final_measurement_pipeline"):
            scaling = self.scaling_factors or getattr(nb, "calibrated_factors", None)
            pixel, cm_vals, warnings = nb.final_measurement_pipeline(
                image_path=image_path,
                view_type=view_type,
                scaling_factors=scaling,
            )
            classification = None
            if hasattr(nb, "run_high_accuracy_demo") and cm_vals:
                side_meas = cm_vals if view_type == "side" else {}
                back_meas = cm_vals if view_type == "back" else {}
                classification = nb.run_high_accuracy_demo(side_meas, back_meas)

            return {
                "cm": cm_vals or {},
                "pixel": pixel or {},
                "warnings": warnings or [],
                "classification": classification,
            }

        return self._make_mock_measurements(image_path, view_type)

    def run_full_vetra(self, side_image_path: str, back_image_path: Optional[str] = None) -> Dict[str, Any]:
        if NOTEBOOK_AVAILABLE and hasattr(nb, "complete_vetra_pipeline"):
            return nb.complete_vetra_pipeline(side_image_path, back_image_path)
        return self._make_mock_full_result(side_image_path, back_image_path)

    def process_image_file(
        self, image_path: str, view_type: str = "side", animal_type: Optional[str] = None
    ) -> Dict[str, Any]:
        result = self.run_measurement(image_path=image_path, view_type=view_type)
        classification = result.get("classification") or {}
        if animal_type and not classification.get("animal_type"):
            classification["animal_type"] = animal_type
            result["classification"] = classification
        return result

    def decode_and_save(self, image_b64: str, suffix: str = ".jpg") -> str:
        """Decode a base64 string (optionally with data URL prefix) to a temp file."""
        if "," in image_b64:
            image_b64 = image_b64.split(",", 1)[1]
        data = base64.b64decode(image_b64)
        fd, path = tempfile.mkstemp(suffix=suffix)
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        return path


vetra_core = VetraCore()
