from __future__ import annotations

from fastapi import APIRouter, Query

from app.schemas import ProcessImageResponse
from app.services.storage import list_results

router = APIRouter()


@router.get("/results", response_model=list[ProcessImageResponse])
def get_results(limit: int = Query(50, ge=1, le=200)) -> list[ProcessImageResponse]:
    return list_results(limit=limit)

