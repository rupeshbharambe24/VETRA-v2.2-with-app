
from __future__ import annotations

from fastapi import APIRouter

from app.schemas import ProcessAnimalRequest, ProcessImageResponse
from app.services.inference import inference_service
from app.services.storage import save_result

router = APIRouter()


@router.post("/process-image", response_model=ProcessImageResponse)
def process_image(payload: ProcessAnimalRequest) -> ProcessImageResponse:
    result = inference_service.process_animal(
        side_image_b64=payload.sideImageData,
        back_image_b64=payload.backImageData,
        animal_type=payload.animalType,
        notes=payload.notes,
    )
    save_result(result)
    return result
