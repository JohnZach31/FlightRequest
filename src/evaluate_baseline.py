import json
from pathlib import Path
#libraries
PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = PROJECT_ROOT / "results" / "rule_baseline_test_predictions.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "results" / "rule_baseline_test_metrics.json"

# add slot fields
# these are the values that the model/baseline is supposed to extract from the text
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


def normalize(value):
    # normalizes the values - J.z
    # this is mostly so "Tel Aviv" and "tel aviv" won't be counted as different answers

    if value is None:
        return None

    if isinstance(value, str):
        return value.strip().lower()

    return value


def compare_slots(gold, prediction):
    # compare the slot fields one by one
    # gold = the correct/manual answer
    # prediction = what my rule baseline guessed

    correct = 0
    total = 0

    for field in SLOT_FIELDS:
        gold_value = normalize(gold.get(field))
        predicted_value = normalize(prediction.get(field))

        if gold_value == predicted_value:
            correct += 1

        total += 1

    return correct, total


def missing_fields_match(gold, prediction):
    # check if the baseline found the same missing fields as the manual label

    gold_missing = set(gold.get("missing_fields", []))
    predicted_missing = set(prediction.get("missing_fields", []))

    return gold_missing == predicted_missing


def main():
    # load the baseline results from the jsonl file

    rows = []

    with INPUT_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            rows.append(json.loads(line))

    total_examples = len(rows)

    intent_correct = 0
    slot_correct = 0
    slot_total = 0
    missing_fields_correct = 0
    full_example_correct = 0

    mistakes = []

    for row in rows:
        gold = row["gold"]
        prediction = row["prediction"]

        # check intent
        intent_is_correct = gold["intent"] == prediction["intent"]

        if intent_is_correct:
            intent_correct += 1

        # check all slot values
        current_slot_correct, current_slot_total = compare_slots(gold, prediction)

        slot_correct += current_slot_correct
        slot_total += current_slot_total

        slots_are_all_correct = current_slot_correct == current_slot_total

        # check missing fields
        missing_is_correct = missing_fields_match(gold, prediction)

        if missing_is_correct:
            missing_fields_correct += 1

        # full example only counts as correct if everything matched
        full_is_correct = (
            intent_is_correct
            and slots_are_all_correct
            and missing_is_correct
        )

        if full_is_correct:
            full_example_correct += 1
        else:
            # save mistakes so I can inspect them later instead of just looking at numbers
            mistakes.append({
                "text": row["text"],
                "gold_intent": gold["intent"],
                "predicted_intent": prediction["intent"],
                "gold_missing_fields": gold.get("missing_fields", []),
                "predicted_missing_fields": prediction.get("missing_fields", []),
                "gold": {field: gold.get(field) for field in SLOT_FIELDS},
                "prediction": {field: prediction.get(field) for field in SLOT_FIELDS},
            })

    # calculate simple metrics for now
    # later I can replace some of this with precision/recall/f1
    # we will use this to compare the rule baseline to the model baseline and see if the model is actually better than a simple rule-based approach
    metrics = {
        "total_examples": total_examples,
        "intent_accuracy": intent_correct / total_examples,
        "slot_accuracy": slot_correct / slot_total,
        "missing_fields_accuracy": missing_fields_correct / total_examples,
        "full_example_accuracy": full_example_correct / total_examples,
        "number_of_mistakes": len(mistakes),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # save both the metrics and the actual mistakes
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

    print("Evaluation results:")
    print("-------------------")

    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")

    print()
    print(f"Saved full evaluation to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()