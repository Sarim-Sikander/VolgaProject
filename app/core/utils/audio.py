import os

from app.core.config import config
from app.core.exceptions import UnsupportedAudioException


def validate_audio(path: str) -> str:
    # faster-whisper decodes any container via PyAV, so we only guard the
    # extension here and let the decoder handle the actual sample-rate/channel
    # normalization (it resamples to 16 kHz mono internally).
    if not os.path.exists(path):
        raise UnsupportedAudioException(f"File not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext not in config.SUPPORTED_FORMATS:
        raise UnsupportedAudioException(
            f"Unsupported format '{ext}'. Supported: {', '.join(config.SUPPORTED_FORMATS)}"
        )
    return ext
