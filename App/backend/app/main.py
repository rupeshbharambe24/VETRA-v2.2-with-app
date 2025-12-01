from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1 import records, vetra
from app.core.config import get_settings
from app.services.inference import inference_service
from app.services.storage import init_db


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="VETRA Backend", version="1.0.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.backend_cors_origins + ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    def health_check() -> dict[str, str]:
        return {
            "status": "ok",
            "service": "vetra-backend",
            "message": "VETRA FastAPI backend is running",
        }

    @app.on_event("startup")
    def startup_event() -> None:
        init_db()
        inference_service.load()

    app.include_router(vetra.router, prefix=settings.api_v1_prefix, tags=["vetra"])
    app.include_router(records.router, prefix=settings.api_v1_prefix, tags=["records"])
    return app


app = create_app()
