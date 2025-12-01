from __future__ import annotations

from fastapi import APIRouter

from app.schemas import ProcessImageRequest, ProcessImageResponse
from app.services.inference import inference_service
from app.services.storage import save_result

router = APIRouter()


@router.post("/process-image", response_model=ProcessImageResponse)
def process_image(payload: ProcessImageRequest) -> ProcessImageResponse:
    result = inference_service.process_single_image(
        image_b64=payload.imageData,
        view_type=payload.viewType,
        animal_type=payload.animalType,
        notes=payload.notes,
    )
    save_result(result)
    return result

