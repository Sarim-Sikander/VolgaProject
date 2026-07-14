from app.controllers.transcription import TranscriptionController
from app.integrations.ollama_client import OllamaClient
from app.integrations.whisper_engine import WhisperEngine


class Factory:
    whisper_engine = WhisperEngine
    ollama_client = OllamaClient

    def get_transcription_controller(self) -> TranscriptionController:
        return TranscriptionController(
            whisper_engine=self.whisper_engine(),
            ollama_client=self.ollama_client(),
        )
