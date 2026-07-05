import json
from pathlib import Path

from schema import get_missing_fields


# stable paths - J.z
PROJECT_ROOT = Path(__file__).resolve().parents[1]

TEST_PATH = PROJECT_ROOT / "data" / "test.jsonl"

INTENT_PREDICTIONS_PATH = PROJECT_ROOT / "results" / "intent_classifier_predictions.jsonl"
SLOT_PREDICTIONS_PATH = PROJECT_ROOT / "results" / "slot_classifier_predictions.jsonl"

OUTPUT_PATH = PROJECT_ROOT / "results" / "ml_pipeline_metrics.json"


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

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            rows.append(json.loads(line))

    return rows


def normalize(value):
    # small cleanup so "Rome" and "rome" count the same
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


def missing_fields_match(gold_missing, predicted_missing):
    return set(gold_missing) == set(predicted_missing)


def main():
    test_examples = read_jsonl(TEST_PATH)
    intent_predictions = read_jsonl(INTENT_PREDICTIONS_PATH)
    slot_predictions = read_jsonl(SLOT_PREDICTIONS_PATH)

    total_examples = len(test_examples)

    intent_correct = 0
    slot_correct = 0
    slot_total = 0
    missing_fields_correct = 0
    full_example_correct = 0

    mistakes = []

    for i in range(total_examples):
        gold = test_examples[i]

        # intent comes from the intent classifier
        predicted_intent = intent_predictions[i]["predicted_intent"]

        # slots come from the slot classifiers
        predicted_slots = slot_predictions[i]["prediction"]

        # missing fields are calculated from the predicted intent + predicted slots
        predicted_missing_fields = get_missing_fields(
            predicted_intent,
            predicted_slots
        )

        intent_is_correct = gold["intent"] == predicted_intent

        if intent_is_correct:
            intent_correct += 1

        current_slot_correct, current_slot_total = compare_slots(
            gold,
            predicted_slots
        )

        slot_correct += current_slot_correct
        slot_total += current_slot_total

        slots_are_all_correct = current_slot_correct == current_slot_total

        missing_is_correct = missing_fields_match(
            gold.get("missing_fields", []),
            predicted_missing_fields
        )

        if missing_is_correct:
            missing_fields_correct += 1

        # full example = intent + all slots + missing fields are all correct
        full_is_correct = (
            intent_is_correct
            and slots_are_all_correct
            and missing_is_correct
        )

        if full_is_correct:
            full_example_correct += 1
        else:
            mistakes.append({
                "text": gold["text"],
                "gold_intent": gold["intent"],
                "predicted_intent": predicted_intent,
                "gold_missing_fields": gold.get("missing_fields", []),
                "predicted_missing_fields": predicted_missing_fields,
                "gold_slots": {field: gold.get(field) for field in SLOT_FIELDS},
                "predicted_slots": predicted_slots,
            })

    metrics = {
        "model": "tfidf_intent_plus_slot_classifiers",
        "total_examples": total_examples,
        "intent_accuracy": intent_correct / total_examples,
        "slot_accuracy": slot_correct / slot_total,
        "missing_fields_accuracy": missing_fields_correct / total_examples,
        "full_example_accuracy": full_example_correct / total_examples,
        "number_of_mistakes": len(mistakes),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

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

    print("ML pipeline evaluation")
    print("----------------------")

    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")

    print()
    print(f"Saved full evaluation to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()