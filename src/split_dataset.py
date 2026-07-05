import json
import random
from pathlib import Path


# always save/read from the real project folder, not inside src - J.z
PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = PROJECT_ROOT / "data" / "synthetic_examples.jsonl"

TRAIN_PATH = PROJECT_ROOT / "data" / "train.jsonl"
VALIDATION_PATH = PROJECT_ROOT / "data" / "validation.jsonl"
TEST_PATH = PROJECT_ROOT / "data" / "test.jsonl"


def read_jsonl(path):
    # jsonl = one json object per line
    examples = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            examples.append(json.loads(line))

    return examples


def write_jsonl(path, examples):
    # write the examples back as jsonl
    with path.open("w", encoding="utf-8") as file:
        for example in examples:
            file.write(json.dumps(example, ensure_ascii=False) + "\n")


def main():
    examples = read_jsonl(INPUT_PATH)

    # keep the split the same every time I run it
    random.seed(42)
    random.shuffle(examples)

    total = len(examples)

    train_end = int(total * 0.70)
    validation_end = int(total * 0.85)

    train_examples = examples[:train_end]
    validation_examples = examples[train_end:validation_end]
    test_examples = examples[validation_end:]

    write_jsonl(TRAIN_PATH, train_examples)
    write_jsonl(VALIDATION_PATH, validation_examples)
    write_jsonl(TEST_PATH, test_examples)

    print("Dataset split complete")
    print("----------------------")
    print(f"total examples: {total}")
    print(f"train examples: {len(train_examples)}")
    print(f"validation examples: {len(validation_examples)}")
    print(f"test examples: {len(test_examples)}")


if __name__ == "__main__":
    main()