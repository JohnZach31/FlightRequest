import json
from pathlib import Path
from schema import get_missing_fields

##Imports and sources -J.z


INPUT_PATH = Path("data/manual_examples.jsonl")
OUTPUT_PATH = Path("data/manual_examples_prepared.jsonl")


def make_question(missing_fields):
    if not missing_fields:
        return ""

    readable = {
        "origin": "where you are flying from",
        "destination": "where you want to fly",
        "departure_date": "when you want to fly",
        "trip_type": "whether this is one-way or round-trip",
        "passengers": "how many passengers are flying",
        "booking_reference": "your booking reference"
    }

    parts = [readable.get(field, field) for field in missing_fields]

    if len(parts) == 1:
        return f"Can you tell me {parts[0]}?"

    return "Can you tell me " + ", ".join(parts[:-1]) + " and " + parts[-1] + "?"


def main():
    prepared_examples = []

    with INPUT_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            example = json.loads(line)

            missing_fields = get_missing_fields(
                intent=example["intent"],
                example=example
            )

            example["missing_fields"] = missing_fields
            example["next_question"] = make_question(missing_fields)

            prepared_examples.append(example)

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        for example in prepared_examples:
            file.write(json.dumps(example, ensure_ascii=False) + "\n")

    print(f"Saved {len(prepared_examples)} examples to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()