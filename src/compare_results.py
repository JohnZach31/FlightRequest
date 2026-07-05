import json
import csv
from pathlib import Path


# stable project path - J.z
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RULE_METRICS_PATH = PROJECT_ROOT / "results" / "rule_baseline_test_metrics.json"
INTENT_METRICS_PATH = PROJECT_ROOT / "results" / "intent_classifier_metrics.json"
SLOT_METRICS_PATH = PROJECT_ROOT / "results" / "slot_classifier_metrics.json"
ML_PIPELINE_METRICS_PATH = PROJECT_ROOT / "results" / "ml_pipeline_metrics.json"
LLM_MANUAL_METRICS_PATH = PROJECT_ROOT / "results" / "llm_manual_20_metrics.json"
HF_INTENT_METRICS_PATH = PROJECT_ROOT / "results" / "hf_intent_classifier_metrics.json"

CSV_OUTPUT_PATH = PROJECT_ROOT / "results" / "model_comparison.csv"
JSON_OUTPUT_PATH = PROJECT_ROOT / "results" / "model_comparison.json"


def load_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def main():
    rule_data = load_json(RULE_METRICS_PATH)
    intent_data = load_json(INTENT_METRICS_PATH)
    slot_data = load_json(SLOT_METRICS_PATH)
    ml_pipeline_data = load_json(ML_PIPELINE_METRICS_PATH)
    llm_manual_data = load_json(LLM_MANUAL_METRICS_PATH)
    hf_intent_data = load_json(HF_INTENT_METRICS_PATH)

    rule_metrics = rule_data["metrics"]
    ml_metrics = ml_pipeline_data["metrics"]
    llm_metrics = llm_manual_data["metrics"]

    # the classical models were tested on the same 150-example test set
    test_total = rule_metrics["total_examples"]

    rows = [
        {
            "method": "rule_based_baseline",
            "task": "full_schema_extraction",
            "total_examples": rule_metrics["total_examples"],
            "intent_accuracy": rule_metrics["intent_accuracy"],
            "slot_accuracy": rule_metrics["slot_accuracy"],
            "missing_fields_accuracy": rule_metrics["missing_fields_accuracy"],
            "full_example_accuracy": rule_metrics["full_example_accuracy"],
        },
        {
            "method": "tfidf_logistic_regression",
            "task": "intent_classification_only",
            "total_examples": test_total,
            "intent_accuracy": intent_data["accuracy"],
            "slot_accuracy": "",
            "missing_fields_accuracy": "",
            "full_example_accuracy": "",
        },
        {
            "method": "fine_tuned_distilbert",
            "task": "intent_classification_only",
            "total_examples": hf_intent_data["total_examples"],
            "intent_accuracy": hf_intent_data["accuracy"],
            "slot_accuracy": "",
            "missing_fields_accuracy": "",
            "full_example_accuracy": "",
        },
        {
            "method": "one_classifier_per_slot",
            "task": "slot_classification_only",
            "total_examples": test_total,
            "intent_accuracy": "",
            "slot_accuracy": slot_data["overall_average_slot_accuracy"],
            "missing_fields_accuracy": "",
            "full_example_accuracy": "",
        },
        {
            "method": "tfidf_intent_plus_slot_classifiers",
            "task": "full_schema_extraction",
            "total_examples": ml_metrics["total_examples"],
            "intent_accuracy": ml_metrics["intent_accuracy"],
            "slot_accuracy": ml_metrics["slot_accuracy"],
            "missing_fields_accuracy": ml_metrics["missing_fields_accuracy"],
            "full_example_accuracy": ml_metrics["full_example_accuracy"],
        },
        {
            "method": "manual_off_the_shelf_llm",
            "task": "full_schema_extraction_20_example_sample",
            "total_examples": llm_metrics["total_examples"],
            "intent_accuracy": llm_metrics["intent_accuracy"],
            "slot_accuracy": llm_metrics["slot_accuracy"],
            "missing_fields_accuracy": llm_metrics["missing_fields_accuracy"],
            "full_example_accuracy": llm_metrics["full_example_accuracy"],
        },
    ]

    JSON_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with JSON_OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(rows, file, indent=2, ensure_ascii=False)

    with CSV_OUTPUT_PATH.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print("Model comparison")
    print("----------------")

    for row in rows:
        print()
        print(f"method: {row['method']}")
        print(f"task: {row['task']}")
        print(f"total examples: {row['total_examples']}")

        if row["intent_accuracy"] != "":
            print(f"intent accuracy: {row['intent_accuracy']}")

        if row["slot_accuracy"] != "":
            print(f"slot accuracy: {row['slot_accuracy']}")

        if row["missing_fields_accuracy"] != "":
            print(f"missing fields accuracy: {row['missing_fields_accuracy']}")

        if row["full_example_accuracy"] != "":
            print(f"full example accuracy: {row['full_example_accuracy']}")

    print()
    print(f"Saved CSV to {CSV_OUTPUT_PATH}")
    print(f"Saved JSON to {JSON_OUTPUT_PATH}")


if __name__ == "__main__":
    main()