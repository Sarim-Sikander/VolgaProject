from pydantic import BaseModel


class Segment(BaseModel):
    start: float
    end: float
    text: str


class TranscriptionResponse(BaseModel):
    language: str
    duration: float
    segments: list[Segment]
    full_text: str
    summary: str | None = None
    cleaned_text: str | None = None
