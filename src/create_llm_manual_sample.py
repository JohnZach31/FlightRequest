import json
from pathlib import Path


# stable paths - J.z
PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = PROJECT_ROOT / "results" / "llm_prompts.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "results" / "llm_manual_sample_20.txt"


def main():
    prompts = []

    with INPUT_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            prompts.append(json.loads(line))

    sample = prompts[:20]

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        for row in sample:
            file.write("=" * 80 + "\n")
            file.write(f"ID: {row['id']}\n")
            file.write("=" * 80 + "\n\n")
            file.write(row["prompt"])
            file.write("\n\n\n")

    print(f"Saved {len(sample)} prompts to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()