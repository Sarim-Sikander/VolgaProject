import argparse
import json

from app.core.factory import Factory


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe an audio file to text with per-segment timestamps."
    )
    parser.add_argument("audio", help="Path to the audio file (wav, mp3, m4a, flac, ...)")
    parser.add_argument(
        "--no-downstream", action="store_true", help="Skip the Ollama summary/cleanup step"
    )
    parser.add_argument("-o", "--output", help="Write the JSON result to this path")
    args = parser.parse_args()

    controller = Factory().get_transcription_controller()
    result = controller.transcribe(args.audio, downstream=not args.no_downstream)

    output = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"Saved transcription to {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
