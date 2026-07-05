import json
from pathlib import Path
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline


# stable paths - J.z
PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = PROJECT_ROOT / "data" / "train.jsonl"
TEST_PATH = PROJECT_ROOT / "data" / "test.jsonl"

METRICS_PATH = PROJECT_ROOT / "results" / "slot_classifier_metrics.json"
PREDICTIONS_PATH = PROJECT_ROOT / "results" / "slot_classifier_predictions.jsonl"


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


NONE_LABEL = "__NONE__"


def read_jsonl(path):
    examples = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            examples.append(json.loads(line))

    return examples


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def encode_value(value):
    # sklearn wants labels as strings, so None gets a special label
    if value is None:
        return NONE_LABEL

    return str(value)


def decode_value(value):
    # convert predictions back to normal project format
    if value == NONE_LABEL:
        return None

    if value.isdigit():
        return int(value)

    return value


def train_one_slot_model(slot_name, x_train, train_examples):
    y_train = [encode_value(example.get(slot_name)) for example in train_examples]

    label_counts = Counter(y_train)

    # if the slot has only one value in training, there is nothing to learn
    if len(label_counts) == 1:
        only_label = list(label_counts.keys())[0]

        return {
            "type": "constant",
            "slot": slot_name,
            "label": only_label,
            "label_counts": dict(label_counts),
        }

    # one small classifier per slot
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
        ("classifier", LogisticRegression(max_iter=1000)),
    ])

    model.fit(x_train, y_train)

    return {
        "type": "model",
        "slot": slot_name,
        "model": model,
        "label_counts": dict(label_counts),
    }


def predict_slot(slot_model, x_test):
    if slot_model["type"] == "constant":
        return [slot_model["label"] for _ in x_test]

    return slot_model["model"].predict(x_test)


def main():
    train_examples = read_jsonl(TRAIN_PATH)
    test_examples = read_jsonl(TEST_PATH)

    x_train = [example["text"] for example in train_examples]
    x_test = [example["text"] for example in test_examples]

    print("Training slot classifiers")
    print("-------------------------")
    print(f"train examples: {len(train_examples)}")
    print(f"test examples: {len(test_examples)}")
    print()

    slot_models = {}

    # train one classifier for each slot field
    for slot in SLOT_FIELDS:
        print(f"training slot: {slot}")
        slot_models[slot] = train_one_slot_model(slot, x_train, train_examples)

    print()
    print("Evaluating slot classifiers")
    print("---------------------------")

    slot_metrics = {}
    all_predictions_by_slot = {}

    for slot in SLOT_FIELDS:
        y_test = [encode_value(example.get(slot)) for example in test_examples]
        y_pred = list(predict_slot(slot_models[slot], x_test))

        accuracy = accuracy_score(y_test, y_pred)

        # output_dict is for saving, regular report is printed below
        report = classification_report(
            y_test,
            y_pred,
            output_dict=True,
            zero_division=0
        )

        slot_metrics[slot] = {
            "accuracy": accuracy,
            "label_counts_train": slot_models[slot]["label_counts"],
            "classification_report": report,
        }

        all_predictions_by_slot[slot] = y_pred

        print(f"{slot}: {accuracy:.3f}")

    prediction_rows = []

    # rebuild one predicted slot dictionary per test example
    for i, example in enumerate(test_examples):
        predicted_slots = {}

        for slot in SLOT_FIELDS:
            raw_prediction = all_predictions_by_slot[slot][i]
            predicted_slots[slot] = decode_value(raw_prediction)

        prediction_rows.append({
            "text": example["text"],
            "gold": {slot: example.get(slot) for slot in SLOT_FIELDS},
            "prediction": predicted_slots,
        })

    overall_slot_accuracy = sum(
        metric["accuracy"] for metric in slot_metrics.values()
    ) / len(slot_metrics)

    metrics = {
        "model": "one_classifier_per_slot_tfidf_logistic_regression",
        "overall_average_slot_accuracy": overall_slot_accuracy,
        "slot_metrics": slot_metrics,
    }

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

    with METRICS_PATH.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)

    write_jsonl(PREDICTIONS_PATH, prediction_rows)

    print()
    print("Overall average slot accuracy:")
    print(f"{overall_slot_accuracy:.3f}")
    print()
    print(f"Saved metrics to {METRICS_PATH}")
    print(f"Saved predictions to {PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()