import json
from pathlib import Path

import matplotlib.pyplot as plt


# stable paths - J.z
PROJECT_ROOT = Path(__file__).resolve().parents[1]

COMPARISON_PATH = PROJECT_ROOT / "results" / "model_comparison.json"
EDA_PATH = PROJECT_ROOT / "results" / "dataset_eda.json"

VISUALS_DIR = PROJECT_ROOT / "visuals"


def load_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def clean_method_name(method):
    names = {
        "rule_based_baseline": "Rule baseline",
        "tfidf_logistic_regression": "TF-IDF intent",
        "fine_tuned_distilbert": "Fine-tuned DistilBERT",
        "one_classifier_per_slot": "Slot classifiers",
        "tfidf_intent_plus_slot_classifiers": "TF-IDF full pipeline",
        "manual_off_the_shelf_llm": "Manual LLM sample",
    }

    return names.get(method, method)


def plot_intent_accuracy(rows):
    intent_rows = [
        row for row in rows
        if row["intent_accuracy"] != ""
    ]

    methods = [clean_method_name(row["method"]) for row in intent_rows]
    scores = [row["intent_accuracy"] for row in intent_rows]

    x = range(len(methods))

    plt.figure(figsize=(10, 5))
    plt.bar(x, scores)
    plt.xticks(x, methods, rotation=25, ha="right")
    plt.ylim(0, 1.05)
    plt.ylabel("Accuracy")
    plt.title("Intent Accuracy by Method")
    plt.tight_layout()

    output_path = VISUALS_DIR / "intent_accuracy_by_method.png"
    plt.savefig(output_path)
    plt.close()

    print(f"Saved {output_path}")


def plot_full_schema_accuracy(rows):
    full_schema_rows = [
        row for row in rows
        if row["full_example_accuracy"] != ""
    ]

    methods = [clean_method_name(row["method"]) for row in full_schema_rows]
    scores = [row["full_example_accuracy"] for row in full_schema_rows]

    x = range(len(methods))

    plt.figure(figsize=(10, 5))
    plt.bar(x, scores)
    plt.xticks(x, methods, rotation=25, ha="right")
    plt.ylim(0, 1.05)
    plt.ylabel("Accuracy")
    plt.title("Full Example Accuracy by Method")
    plt.tight_layout()

    output_path = VISUALS_DIR / "full_example_accuracy.png"
    plt.savefig(output_path)
    plt.close()

    print(f"Saved {output_path}")


def plot_slot_missing_accuracy(rows):
    full_schema_rows = [
        row for row in rows
        if row["slot_accuracy"] != "" and row["missing_fields_accuracy"] != ""
    ]

    methods = [clean_method_name(row["method"]) for row in full_schema_rows]
    slot_scores = [row["slot_accuracy"] for row in full_schema_rows]
    missing_scores = [row["missing_fields_accuracy"] for row in full_schema_rows]

    width = 0.35
    x_positions = list(range(len(methods)))

    plt.figure(figsize=(10, 5))

    plt.bar(
        [x - width / 2 for x in x_positions],
        slot_scores,
        width=width,
        label="Slot accuracy"
    )

    plt.bar(
        [x + width / 2 for x in x_positions],
        missing_scores,
        width=width,
        label="Missing fields accuracy"
    )

    plt.xticks(x_positions, methods, rotation=25, ha="right")
    plt.ylim(0, 1.05)
    plt.ylabel("Accuracy")
    plt.title("Slot and Missing-Field Accuracy")
    plt.legend()
    plt.tight_layout()

    output_path = VISUALS_DIR / "slot_missing_accuracy.png"
    plt.savefig(output_path)
    plt.close()

    print(f"Saved {output_path}")


def plot_dataset_intent_distribution(eda):
    intent_distribution = eda["intent_distribution"]

    labels = list(intent_distribution.keys())
    counts = list(intent_distribution.values())

    plt.figure(figsize=(8, 5))
    plt.bar(labels, counts)
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Number of examples")
    plt.title("Synthetic Dataset Intent Distribution")
    plt.tight_layout()

    output_path = VISUALS_DIR / "intent_distribution.png"
    plt.savefig(output_path)
    plt.close()

    print(f"Saved {output_path}")


def main():
    VISUALS_DIR.mkdir(parents=True, exist_ok=True)

    comparison_rows = load_json(COMPARISON_PATH)
    eda = load_json(EDA_PATH)

    plot_intent_accuracy(comparison_rows)
    plot_full_schema_accuracy(comparison_rows)
    plot_slot_missing_accuracy(comparison_rows)
    plot_dataset_intent_distribution(eda)

    print()
    print("Done creating updated result graphs.")


if __name__ == "__main__":
    main()