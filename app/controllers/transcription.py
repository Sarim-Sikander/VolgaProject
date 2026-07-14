import sys

from app.core.config import config
from app.core.exceptions import TranscriptionException
from app.core.utils.audio import validate_audio
from app.integrations.ollama_client import OllamaClient
from app.integrations.whisper_engine import WhisperEngine


class TranscriptionController:
    def __init__(self, whisper_engine: WhisperEngine, ollama_client: OllamaClient):
        self.whisper = whisper_engine
        self.ollama = ollama_client

    def transcribe(self, audio_path: str, downstream: bool = True) -> dict:
        validate_audio(audio_path)

        try:
            result = self.whisper.transcribe(audio_path)
        except Exception as e:
            raise TranscriptionException(detail=str(e))

        if downstream and config.ENABLE_DOWNSTREAM and result["full_text"]:
            result.update(self._downstream(result["full_text"]))

        return result

    def _downstream(self, text: str) -> dict:
        try:
            return {
                "summary": self.ollama.summarize(text),
                "cleaned_text": self.ollama.clean_transcript(text),
            }
        except Exception as e:
            # best-effort: keep the transcript even if the LLM step fails
            print(f"[downstream skipped] {e}", file=sys.stderr)
            return {"summary": None, "cleaned_text": None}
