import threading

from faster_whisper import WhisperModel

from app.core.config import config


class WhisperEngine:
    _model = None
    _lock = threading.Lock()

    def __init__(self):
        self.model_size = config.WHISPER_MODEL
        self.device = config.WHISPER_DEVICE
        self.compute_type = config.WHISPER_COMPUTE_TYPE

    def _load(self):
        if WhisperEngine._model is None:
            with WhisperEngine._lock:
                if WhisperEngine._model is None:
                    WhisperEngine._model = self._build()
        return WhisperEngine._model

    def _build(self, force_cpu: bool = False):
        if force_cpu or self.device == "cpu":
            return WhisperModel(self.model_size, device="cpu", compute_type="int8")

        device = "cuda" if self.device in ("auto", "cuda") else "cpu"
        compute = self.compute_type
        if compute == "auto":
            compute = "float16" if device == "cuda" else "int8"
        return WhisperModel(self.model_size, device=device, compute_type=compute)

    def transcribe(self, audio_path: str) -> dict:
        try:
            return self._run(self._load(), audio_path)
        except Exception:
            WhisperEngine._model = self._build(force_cpu=True)
            return self._run(WhisperEngine._model, audio_path)

    def _run(self, model, audio_path: str) -> dict:
        segments, info = model.transcribe(
            audio_path,
            beam_size=config.WHISPER_BEAM_SIZE,
            vad_filter=config.WHISPER_VAD_FILTER,
        )

        seg_list = []
        text_parts = []
        for s in segments:
            text = s.text.strip()
            seg_list.append({"start": round(s.start, 2), "end": round(s.end, 2), "text": text})
            text_parts.append(text)

        return {
            "language": info.language,
            "duration": round(info.duration, 2),
            "segments": seg_list,
            "full_text": " ".join(text_parts).strip(),
        }
