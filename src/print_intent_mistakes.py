import json
from pathlib import Path


# stable path from project root - J.z
PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = PROJECT_ROOT / "results" / "intent_classifier_predictions.jsonl"


def main():
    mistakes = []

    with INPUT_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)

            if not row["correct"]:
                mistakes.append(row)

    print(f"Found {len(mistakes)} intent mistakes")
    print("=" * 60)

    for i, mistake in enumerate(mistakes, start=1):
        print()
        print(f"Mistake #{i}")
        print("-" * 60)
        print("Text:")
        print(mistake["text"])
        print()
        print(f"gold:      {mistake['gold_intent']}")
        print(f"predicted: {mistake['predicted_intent']}")


if __name__ == "__main__":
    main()