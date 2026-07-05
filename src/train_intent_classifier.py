import json
from pathlib import Path
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline


# keeping paths stable even if I run the script from /src - J.z
PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = PROJECT_ROOT / "data" / "train.jsonl"
TEST_PATH = PROJECT_ROOT / "data" / "test.jsonl"

METRICS_PATH = PROJECT_ROOT / "results" / "intent_classifier_metrics.json"
PREDICTIONS_PATH = PROJECT_ROOT / "results" / "intent_classifier_predictions.jsonl"


def read_jsonl(path):
    # jsonl = each line is a separate example
    examples = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            examples.append(json.loads(line))

    return examples


def write_jsonl(path, rows):
    # saving predictions this way makes it easier to inspect mistakes later
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    # load the already-split dataset
    train_examples = read_jsonl(TRAIN_PATH)
    test_examples = read_jsonl(TEST_PATH)

    # x = user text, y = intent label
    x_train = [example["text"] for example in train_examples]
    y_train = [example["intent"] for example in train_examples]

    x_test = [example["text"] for example in test_examples]
    y_test = [example["intent"] for example in test_examples]

    print("Training intent classifier")
    print("--------------------------")
    print(f"train examples: {len(train_examples)}")
    print(f"test examples: {len(test_examples)}")
    print()

    # quick check to see if the train data is super unbalanced
    print("train intent distribution:")
    print(dict(Counter(y_train)))
    print()

    # first real ML baseline:
    # TF-IDF converts text into numeric features, LogisticRegression predicts the intent
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
        ("classifier", LogisticRegression(max_iter=1000)),
    ])

    # train only on the train split
    model.fit(x_train, y_train)

    # test on unseen examples
    predictions = model.predict(x_test)

    # simple overall score
    accuracy = accuracy_score(y_test, predictions)

    # gives precision / recall / f1 per intent, better than just accuracy
    report = classification_report(
        y_test,
        predictions,
        output_dict=True,
        zero_division=0
    )

    # useful for seeing which intents get confused
    labels = sorted(set(y_test) | set(predictions))
    matrix = confusion_matrix(y_test, predictions, labels=labels)

    prediction_rows = []

    # save every prediction, not just the final score
    for text, gold_intent, predicted_intent in zip(x_test, y_test, predictions):
        prediction_rows.append({
            "text": text,
            "gold_intent": gold_intent,
            "predicted_intent": predicted_intent,
            "correct": gold_intent == predicted_intent,
        })

    # one json file for summary metrics
    metrics = {
        "model": "tfidf_logistic_regression",
        "accuracy": accuracy,
        "labels": labels,
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
    }

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

    with METRICS_PATH.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)

    # separate file for manually checking prediction errors
    write_jsonl(PREDICTIONS_PATH, prediction_rows)

    print("Results")
    print("-------")
    print(f"accuracy: {accuracy:.3f}")
    print()

    # readable terminal version of the report
    print(classification_report(y_test, predictions, zero_division=0))

    print(f"Saved metrics to {METRICS_PATH}")
    print(f"Saved predictions to {PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()