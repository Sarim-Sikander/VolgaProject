import requests

from app.core.config import config


class OllamaClient:
    def __init__(self):
        self.base_url = config.OLLAMA_BASE_URL.rstrip("/")
        self.model = config.OLLAMA_MODEL

    def _generate(self, prompt: str, system: str | None = None) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            # cap context so the KV cache fits in VRAM (models default to 128k)
            "options": {"num_ctx": config.OLLAMA_NUM_CTX},
        }
        if system:
            payload["system"] = system
        resp = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=180)
        resp.raise_for_status()
        return resp.json().get("response", "").strip()

    def summarize(self, text: str) -> str:
        prompt = (
            "Summarize the following transcript in 2-3 sentences. "
            "Respond with the summary only, no preamble:\n\n" + text
        )
        return self._generate(prompt, system="You are a concise assistant.")

    def clean_transcript(self, text: str) -> str:
        prompt = (
            "Clean up this raw transcript. Fix punctuation and capitalization and remove "
            "filler words, but keep the meaning identical. Respond with the cleaned text "
            "only, no preamble:\n\n" + text
        )
        return self._generate(prompt, system="You are a transcript editor.")

    def health(self) -> bool:
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            resp.raise_for_status()
            return True
        except Exception:
            return False
