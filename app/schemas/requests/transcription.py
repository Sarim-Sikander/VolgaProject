from pydantic import BaseModel


class TranscriptionOptions(BaseModel):
    downstream: bool = True
