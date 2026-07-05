import json
from pathlib import Path


# project root so this works even if I run it from src - J.z
PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = PROJECT_ROOT / "results" / "rule_baseline_test_metrics.json"


def main():
    with INPUT_PATH.open("r", encoding="utf-8") as file:
        data = json.load(file)

    mistakes = data["mistakes"]

    print(f"Found {len(mistakes)} mistakes")
    print("=" * 60)

    for i, mistake in enumerate(mistakes, start=1):
        print()
        print(f"Mistake #{i}")
        print("-" * 60)

        print("Text:")
        print(mistake["text"])

        print()
        print("Intent:")
        print(f"gold:      {mistake['gold_intent']}")
        print(f"predicted: {mistake['predicted_intent']}")

        print()
        print("Missing fields:")
        print(f"gold:      {mistake['gold_missing_fields']}")
        print(f"predicted: {mistake['predicted_missing_fields']}")

        print()
        print("Slot differences:")

        gold_slots = mistake["gold"]
        predicted_slots = mistake["prediction"]

        for field in gold_slots:
            gold_value = gold_slots[field]
            predicted_value = predicted_slots[field]

            if gold_value != predicted_value:
                print(f"{field}: gold={gold_value} | predicted={predicted_value}")


if __name__ == "__main__":
    main()