import os
import tempfile

from fastapi import APIRouter, Depends, File, Query, UploadFile
from starlette.concurrency import run_in_threadpool

from app.controllers.transcription import TranscriptionController
from app.core.factory import Factory
from app.schemas.responses.transcription import TranscriptionResponse

router = APIRouter()


@router.get("/health")
async def health(
    controller: TranscriptionController = Depends(Factory().get_transcription_controller),
):
    return {"status": "ok", "ollama_available": controller.ollama.health()}


@router.post("/", response_model=TranscriptionResponse)
async def transcribe_audio(
    file: UploadFile = File(...),
    downstream: bool = Query(True, description="Run Ollama summary + cleanup"),
    controller: TranscriptionController = Depends(Factory().get_transcription_controller),
):
    suffix = os.path.splitext(file.filename or "")[1].lower() or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        result = await run_in_threadpool(controller.transcribe, tmp_path, downstream)
    finally:
        os.remove(tmp_path)

    return result
