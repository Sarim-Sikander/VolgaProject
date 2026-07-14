from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", case_sensitive=True, extra="ignore"
    )

    HOST: str = "0.0.0.0"
    PORT: int = 8000
    ENVIRONMENT: str = "development"

    WHISPER_MODEL: str = "small"
    WHISPER_DEVICE: str = "auto" 
    WHISPER_COMPUTE_TYPE: str = "auto"
    WHISPER_BEAM_SIZE: int = 5
    WHISPER_VAD_FILTER: bool = True
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.2:3b"
    OLLAMA_NUM_CTX: int = 4096
    ENABLE_DOWNSTREAM: bool = True

    MAX_UPLOAD_MB: int = 100
    SUPPORTED_FORMATS: tuple[str, ...] = (
        ".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm", ".aac",
    )


config = Config()
