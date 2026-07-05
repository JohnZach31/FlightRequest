import json
from pathlib import Path
from collections import Counter


# project root so it works from anywhere - J.z
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "synthetic_examples.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "results" / "dataset_eda.json"


SLOT_FIELDS = [
    "origin",
    "destination",
    "departure_date",
    "return_date",
    "trip_type",
    "passengers",
    "cabin_class",
    "budget",
    "layover_preference",
    "time_preference",
    "baggage",
    "airline_preference",
    "booking_reference",
]


def read_jsonl(path):
    examples = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            examples.append(json.loads(line))

    return examples


def main():
    examples = read_jsonl(DATA_PATH)

    intent_counter = Counter()
    missing_field_counter = Counter()
    filled_slot_counter = Counter()
    text_lengths = []

    for example in examples:
        intent_counter[example["intent"]] += 1

        text_lengths.append(len(example["text"].split()))

        for field in example.get("missing_fields", []):
            missing_field_counter[field] += 1

        for field in SLOT_FIELDS:
            if example.get(field) is not None:
                filled_slot_counter[field] += 1

    eda = {
        "total_examples": len(examples),
        "intent_distribution": dict(intent_counter),
        "missing_field_distribution": dict(missing_field_counter),
        "filled_slot_distribution": dict(filled_slot_counter),
        "average_text_length_words": sum(text_lengths) / len(text_lengths),
        "min_text_length_words": min(text_lengths),
        "max_text_length_words": max(text_lengths),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(eda, file, indent=2, ensure_ascii=False)

    print("Dataset EDA")
    print("-----------")
    print(f"total examples: {eda['total_examples']}")
    print(f"average text length: {eda['average_text_length_words']:.2f} words")
    print()
    print("intent distribution:")
    for intent, count in eda["intent_distribution"].items():
        print(f"{intent}: {count}")

    print()
    print(f"Saved EDA to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()