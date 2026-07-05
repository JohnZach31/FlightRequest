import json
import inspect
from pathlib import Path

import numpy as np
from datasets import Dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


# stable project path - J.z
PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = PROJECT_ROOT / "data" / "train.jsonl"
VALIDATION_PATH = PROJECT_ROOT / "data" / "validation.jsonl"
TEST_PATH = PROJECT_ROOT / "data" / "test.jsonl"

MODEL_OUTPUT_DIR = PROJECT_ROOT / "models" / "hf_intent_distilbert"

METRICS_OUTPUT_PATH = PROJECT_ROOT / "results" / "hf_intent_classifier_metrics.json"
PREDICTIONS_OUTPUT_PATH = PROJECT_ROOT / "results" / "hf_intent_classifier_predictions.jsonl"

MODEL_NAME = "distilbert-base-uncased"


def read_jsonl(path):
    rows = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            rows.append(json.loads(line))

    return rows


def convert_to_dataset(rows, label_to_id):
    simple_rows = []

    for row in rows:
        simple_rows.append({
            "text": row["text"],
            "label": label_to_id[row["intent"]],
        })

    return Dataset.from_list(simple_rows)


def build_training_args():
    kwargs = {
        "output_dir": str(MODEL_OUTPUT_DIR),
        "learning_rate": 2e-5,
        "per_device_train_batch_size": 8,
        "per_device_eval_batch_size": 8,
        "num_train_epochs": 3,
        "weight_decay": 0.01,
        "logging_steps": 20,
        "save_strategy": "no",
        "report_to": "none",
    }

    # transformers changed this argument name between versions
    params = inspect.signature(TrainingArguments.__init__).parameters

    if "eval_strategy" in params:
        kwargs["eval_strategy"] = "epoch"
    else:
        kwargs["evaluation_strategy"] = "epoch"

    return TrainingArguments(**kwargs)


def build_trainer(model, training_args, train_dataset, validation_dataset, tokenizer, data_collator):
    def compute_metrics(eval_prediction):
        # supports both older and newer HF prediction formats
        if hasattr(eval_prediction, "predictions"):
            logits = eval_prediction.predictions
            labels_array = eval_prediction.label_ids
        else:
            logits, labels_array = eval_prediction

        if isinstance(logits, tuple):
            logits = logits[0]

        predictions = np.argmax(logits, axis=-1)

        return {
            "accuracy": accuracy_score(labels_array, predictions)
        }

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": validation_dataset,
        "data_collator": data_collator,
        "compute_metrics": compute_metrics,
    }

    # HF changed tokenizer= to processing_class= in some versions
    trainer_params = inspect.signature(Trainer.__init__).parameters

    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer

    return Trainer(**trainer_kwargs)


def main():
    train_rows = read_jsonl(TRAIN_PATH)
    validation_rows = read_jsonl(VALIDATION_PATH)
    test_rows = read_jsonl(TEST_PATH)

    labels = sorted({row["intent"] for row in train_rows})

    label_to_id = {
        label: i
        for i, label in enumerate(labels)
    }

    id_to_label = {
        i: label
        for label, i in label_to_id.items()
    }

    train_dataset = convert_to_dataset(train_rows, label_to_id)
    validation_dataset = convert_to_dataset(validation_rows, label_to_id)
    test_dataset = convert_to_dataset(test_rows, label_to_id)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=128,
        )

    tokenized_train = train_dataset.map(tokenize, batched=True)
    tokenized_validation = validation_dataset.map(tokenize, batched=True)
    tokenized_test = test_dataset.map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(labels),
        id2label=id_to_label,
        label2id=label_to_id,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    training_args = build_training_args()

    trainer = build_trainer(
        model=model,
        training_args=training_args,
        train_dataset=tokenized_train,
        validation_dataset=tokenized_validation,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("Training fine-tuned DistilBERT intent classifier...")
    trainer.train()

    print("Evaluating on test set...")
    test_output = trainer.predict(tokenized_test)

    logits = test_output.predictions

    if isinstance(logits, tuple):
        logits = logits[0]

    predicted_ids = np.argmax(logits, axis=-1)
    gold_ids = test_output.label_ids

    predicted_labels = [
        id_to_label[int(label_id)]
        for label_id in predicted_ids
    ]

    gold_labels = [
        id_to_label[int(label_id)]
        for label_id in gold_ids
    ]

    accuracy = accuracy_score(gold_labels, predicted_labels)

    report = classification_report(
        gold_labels,
        predicted_labels,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        gold_labels,
        predicted_labels,
        labels=labels,
    ).tolist()

    mistakes = []

    PREDICTIONS_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with PREDICTIONS_OUTPUT_PATH.open("w", encoding="utf-8") as file:
        for row, gold, prediction in zip(test_rows, gold_labels, predicted_labels):
            prediction_row = {
                "text": row["text"],
                "gold_intent": gold,
                "predicted_intent": prediction,
            }

            file.write(json.dumps(prediction_row, ensure_ascii=False) + "\n")

            if gold != prediction:
                mistakes.append(prediction_row)

    metrics = {
        "model": "fine_tuned_distilbert_intent_classifier",
        "base_model": MODEL_NAME,
        "task": "intent_classification_only",
        "total_examples": len(test_rows),
        "accuracy": accuracy,
        "labels": labels,
        "classification_report": report,
        "confusion_matrix": matrix,
        "number_of_mistakes": len(mistakes),
        "mistakes": mistakes,
    }

    METRICS_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with METRICS_OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)

    trainer.save_model(MODEL_OUTPUT_DIR)
    tokenizer.save_pretrained(MODEL_OUTPUT_DIR)

    print()
    print("HF DistilBERT intent classifier")
    print("--------------------------------")
    print(f"base model: {MODEL_NAME}")
    print(f"test examples: {len(test_rows)}")
    print(f"accuracy: {accuracy:.3f}")
    print(f"mistakes: {len(mistakes)}")
    print()
    print(f"Saved metrics to {METRICS_OUTPUT_PATH}")
    print(f"Saved predictions to {PREDICTIONS_OUTPUT_PATH}")
    print(f"Saved model to {MODEL_OUTPUT_DIR}")


if __name__ == "__main__":
    main()