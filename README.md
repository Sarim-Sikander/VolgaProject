# Transcription Pipeline — Design & Decisions

A small but complete transcription pipeline: it takes an audio file, converts it
to text with **per-segment timestamps** (Whisper), and then runs a **downstream
processing** step (summary + cleanup) with a local LLM (Ollama).

It ships as both a **CLI** and a **FastAPI service**, following a
Factory / Controller / Integration layout.

---

## Architecture

```
audio file
   -> validate format             (app/core/utils/audio.py)
   -> WhisperEngine.transcribe     (app/integrations/whisper_engine.py)  -> segments + timestamps
   -> OllamaClient (downstream)    (app/integrations/ollama_client.py)   -> summary + cleaned text
   -> JSON response
```

| Layer | File(s) | Responsibility |
|-------|---------|----------------|
| Entry — API | `main.py`, `app/core/server.py` | Uvicorn + app factory |
| Entry — CLI | `cli.py` | Same pipeline from the command line |
| Routing | `app/api/api.py`, `app/api/endpoints/transcription.py` | HTTP endpoints |
| Orchestration | `app/controllers/transcription.py` | Ties the steps together |
| Wiring (DI) | `app/core/factory/factory.py` | Builds controllers + integrations |
| Integrations | `app/integrations/whisper_engine.py`, `app/integrations/ollama_client.py` | External engines |
| Schemas | `app/schemas/responses/transcription.py` | Pydantic response models |
| Config | `app/core/config.py`, `.env` | Typed settings |

---

## Design decisions

### 1. A two-stage pipeline: Whisper for STT, Ollama for downstream
The prompt has two halves — *"convert audio to text"* **and** *"process the
result for downstream use."* Those are different problems, so they use different
tools. Ollama runs text LLMs and **cannot transcribe audio**, so it can't be the
transcriber; Whisper is the right tool for speech-to-text. Ollama is then a
natural fit for the *downstream* half — it takes the transcript and produces a
summary and a cleaned-up version. This keeps a local, no-API-key stack end to end.

### 2. Why `faster-whisper`
Over the alternatives: it's a CTranslate2 reimplementation of OpenAI Whisper —
roughly **4× faster** and lighter on memory than the reference `openai-whisper`,
runs on **CPU or GPU**, and (unlike `whisper.cpp`) is pure-Python to integrate.
Crucially, it returns **segments with timestamps built in**, so the "timestamps
per segment" requirement needs no extra work. The model is loaded **once and
shared** (a module-level singleton) rather than per request, since model load is
the expensive part.

### 3. Layered architecture (Factory / Controller / Integration)
Even for a small task, separating concerns keeps it testable and swappable:
integrations wrap the external engines (Whisper, Ollama), the controller
orchestrates them, and the Factory wires dependencies so the CLI and API share
the exact same code path. Swapping Whisper for a cloud STT, or Ollama for
another LLM, is a one-file change behind a stable interface.

### 4. Accepting audio (CLI + API)
The CLI takes a file path; the API accepts a **multipart upload**, writes it to a
temp file, processes it, and deletes it in a `finally`. The extension is checked
against an allowlist up front so bad input fails fast with a clean `400` rather
than deep inside the decoder.

### 5. Timestamps per segment
Whisper natively segments speech and assigns each segment a `start`/`end`. We map
those straight into a JSON `segments[]` of `{start, end, text}` (rounded to 2 dp)
plus a concatenated `full_text`.

### 6. Handling different audio formats
faster-whisper decodes through **PyAV** (the ffmpeg libraries shipped as Python
wheels), so WAV / MP3 / M4A / FLAC / OGG all work **with no separate ffmpeg
install**, and everything is resampled to 16 kHz mono internally — exactly what
Whisper expects. So there's no per-format branching; the allowlist exists only to
give a clean early error.

### 7. Long audio files
Whisper processes audio in 30-second windows, and faster-whisper yields segments
as a **generator**. We stream and accumulate segment by segment instead of
holding the whole thing in memory, so the **memory footprint stays flat**
regardless of length. A **VAD (voice-activity-detection) filter** skips silence
so we don't waste compute on non-speech. The one scaling caveat is the downstream
LLM step: a very long transcript would exceed the context window, so at scale
that step moves to chunked (map-reduce) summarization.

### 8. Hardware / GPU with automatic fallback
`WHISPER_DEVICE=auto` tries CUDA first and **falls back to CPU (int8)** if the
GPU or its cuDNN libraries aren't available — at build time *and* at runtime — so
the same code runs on any machine. Model size and compute type are configurable.

### 9. Downstream model choice, VRAM, and `num_ctx`
This was the one real gotcha. On a 4 GB GPU, `qwen2.5:7b` (4.7 GB) **OOMs**, and
any model initially tried to allocate a **128k-token context** whose KV cache
won't fit in memory. Two decisions came out of that:
- Default to **`llama3.2:3b`**, which fits a 4 GB GPU comfortably. `qwen2.5:7b`
  gives better output and is available via config for machines with more VRAM.
- **Pin `num_ctx`** (`OLLAMA_NUM_CTX`, default 4096) so the KV cache is bounded
  and the model actually loads on modest hardware.

### 10. Downstream is best-effort
The transcript is the primary product. If Ollama is down or errors, we still
return the full transcript + timestamps; `summary`/`cleaned_text` come back
`null` and a warning is written to stderr. A failed summary never fails the job.

### 11. Config-driven
Everything tunable — model sizes, device, compute type, Ollama URL/model,
context length, downstream on/off — lives in typed settings (`.env`), so no code
change is needed to move between a laptop and a GPU box.

---

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate  
pip install -r requirements.txt
cp .env.example .env       
```

Ollama must be running locally with the configured model pulled:

```bash
ollama pull llama3.2:3b
```

## Usage

Generate a mock audio sample (uses offline TTS, no external calls):

```bash
python scripts/generate_mock_audio.py       # writes sample/sample.wav
```

**CLI:**

```bash
python cli.py sample/sample.wav                  # transcript + downstream
python cli.py sample/sample.wav --no-downstream  # transcript only
python cli.py sample/sample.wav -o result.json   # write JSON to a file
```

**API:**

```bash
python main.py             # http://localhost:8000/docs
curl -F "file=@sample/sample.wav" "http://localhost:8000/transcription/?downstream=true"
```

Response shape:

```json
{
  "language": "en",
  "duration": 25.18,
  "segments": [{ "start": 0.0, "end": 4.0, "text": "Hello and welcome..." }],
  "full_text": "Hello and welcome ...",
  "summary": "A short demo transcript ...",
  "cleaned_text": "Hello and welcome ..."
}
```

---

## Known limitations / next steps

This is intentionally scoped to the task. For a production system I'd add: a
job queue + worker pool so uploads are processed asynchronously (and GPU
concurrency is bounded to avoid the OOM above); object storage for audio and a
database for transcripts; a job state machine with retries/backoff and a
dead-letter queue; and auth + rate limiting on the API. Accuracy scales with the
Whisper model size (`tiny` → `large-v3`) via a single config value.
