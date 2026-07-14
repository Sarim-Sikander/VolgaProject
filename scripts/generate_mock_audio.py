import os
import subprocess
import sys

SAMPLE_TEXT = (
    "Hello and welcome to the transcription pipeline demo. "
    "This is a short sample of spoken audio generated for testing. "
    "It contains a few sentences so we can verify the segment timestamps. "
    "The pipeline uses Whisper for speech to text and Ollama for downstream processing. "
    "Thank you for listening and have a great day."
)

OUT_PATH = os.path.join("sample", "sample.wav")


def _via_pyttsx3(text: str, out_path: str) -> bool:
    try:
        import pyttsx3
    except ImportError:
        return False
    engine = pyttsx3.init()
    engine.setProperty("rate", 165)
    engine.save_to_file(text, out_path)
    engine.runAndWait()
    return os.path.exists(out_path) and os.path.getsize(out_path) > 1024


def _via_system_speech(text: str, out_path: str) -> bool:
    if sys.platform != "win32":
        return False
    safe = text.replace("'", "''")
    ps = (
        "Add-Type -AssemblyName System.Speech; "
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        f"$s.SetOutputToWaveFile('{os.path.abspath(out_path)}'); "
        f"$s.Speak('{safe}'); $s.Dispose()"
    )
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True)
    return os.path.exists(out_path) and os.path.getsize(out_path) > 1024


def main():
    os.makedirs("sample", exist_ok=True)
    if _via_pyttsx3(SAMPLE_TEXT, OUT_PATH) or _via_system_speech(SAMPLE_TEXT, OUT_PATH):
        print(f"Generated mock audio at {OUT_PATH}")
    else:
        raise RuntimeError("Could not generate mock audio with pyttsx3 or System.Speech")


if __name__ == "__main__":
    main()
