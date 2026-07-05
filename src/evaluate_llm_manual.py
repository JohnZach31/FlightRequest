import json
from pathlib import Path


# stable paths - J.z
PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROMPTS_PATH = PROJECT_ROOT / "results" / "llm_prompts.jsonl"
PREDICTIONS_PATH = PROJECT_ROOT / "results" / "llm_predictions_manual_20.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "results" / "llm_manual_20_metrics.json"


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
    rows = []

    with path.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            # skip accidental blank lines
            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                print()
                print(f"Bad JSON on line {line_number} in {path}")
                print("Line content:")
                print(repr(line))
                raise

    return rows


def normalize(value):
    # small cleanup before comparing values
    if value is None:
        return None

    if isinstance(value, str):
        return value.strip().lower()

    return value


def compare_slots(gold, prediction):
    correct = 0
    total = 0

    for field in SLOT_FIELDS:
        gold_value = normalize(gold.get(field))
        predicted_value = normalize(prediction.get(field))

        if gold_value == predicted_value:
            correct += 1

        total += 1

    return correct, total


def main():
    prompt_rows = read_jsonl(PROMPTS_PATH)
    prediction_rows = read_jsonl(PREDICTIONS_PATH)

    # only evaluate ids that exist in the manual prediction file
    gold_by_id = {
        row["id"]: row["gold"]
        for row in prompt_rows
    }

    total_examples = len(prediction_rows)

    intent_correct = 0
    slot_correct = 0
    slot_total = 0
    missing_fields_correct = 0
    full_example_correct = 0

    mistakes = []

    for row in prediction_rows:
        example_id = row["id"]
        prediction = row["prediction"]
        gold = gold_by_id[example_id]

        intent_is_correct = gold["intent"] == prediction["intent"]

        if intent_is_correct:
            intent_correct += 1

        current_slot_correct, current_slot_total = compare_slots(gold, prediction)

        slot_correct += current_slot_correct
        slot_total += current_slot_total

        slots_are_all_correct = current_slot_correct == current_slot_total

        gold_missing = set(gold.get("missing_fields", []))
        predicted_missing = set(prediction.get("missing_fields", []))

        missing_is_correct = gold_missing == predicted_missing

        if missing_is_correct:
            missing_fields_correct += 1

        full_is_correct = (
            intent_is_correct
            and slots_are_all_correct
            and missing_is_correct
        )

        if full_is_correct:
            full_example_correct += 1
        else:
            mistakes.append({
                "id": example_id,
                "gold_intent": gold["intent"],
                "predicted_intent": prediction["intent"],
                "gold_missing_fields": gold.get("missing_fields", []),
                "predicted_missing_fields": prediction.get("missing_fields", []),
                "gold_slots": {field: gold.get(field) for field in SLOT_FIELDS},
                "predicted_slots": {field: prediction.get(field) for field in SLOT_FIELDS},
            })

    metrics = {
        "model": "manual_off_the_shelf_llm_20_examples",
        "total_examples": total_examples,
        "intent_accuracy": intent_correct / total_examples,
        "slot_accuracy": slot_correct / slot_total,
        "missing_fields_accuracy": missing_fields_correct / total_examples,
        "full_example_accuracy": full_example_correct / total_examples,
        "number_of_mistakes": len(mistakes),
    }

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "metrics": metrics,
                "mistakes": mistakes,
            },
            file,
            indent=2,
            ensure_ascii=False
        )

    print("Manual LLM evaluation")
    print("---------------------")

    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")

    print()
    print(f"Saved full evaluation to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()