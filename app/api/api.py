from fastapi import APIRouter

from app.api.endpoints import transcription

router = APIRouter()

router.include_router(
    transcription.router, prefix="/transcription", tags=["transcription"]
)
