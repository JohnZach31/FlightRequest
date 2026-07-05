import json
from pathlib import Path


# stable paths - J.z
PROJECT_ROOT = Path(__file__).resolve().parents[1]

TEST_PATH = PROJECT_ROOT / "data" / "test.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "results" / "llm_prompts.jsonl"


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


def build_prompt(text):
    # strict prompt because we need JSON, not a friendly chatbot answer
    return f"""
You are an NLP information extraction system for flight booking requests.

Extract the following user request into valid JSON.

Allowed intents:
- search_flight
- book_flight
- change_flight
- cancel_flight
- baggage_question

Return exactly these fields:
- intent
- origin
- destination
- departure_date
- return_date
- trip_type
- passengers
- cabin_class
- budget
- layover_preference
- time_preference
- baggage
- airline_preference
- booking_reference
- missing_fields

Rules:
- Use null for missing values.
- passengers should be a number or null.
- trip_type should be "one-way", "round-trip", or null.
- missing_fields should include required fields that are missing.
- For search_flight and book_flight, required fields are:
  origin, destination, departure_date, trip_type, passengers.
- For change_flight and cancel_flight, required field is:
  booking_reference.
- For baggage_question, missing_fields should usually be empty.
- Return only JSON. Do not explain.

User request:
{text}
""".strip()


def main():
    test_examples = read_jsonl(TEST_PATH)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        for i, example in enumerate(test_examples):
            row = {
                "id": i,
                "text": example["text"],
                "prompt": build_prompt(example["text"]),
                "gold": {
                    "intent": example["intent"],
                    **{field: example.get(field) for field in SLOT_FIELDS},
                    "missing_fields": example.get("missing_fields", []),
                },
            }

            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Saved {len(test_examples)} LLM prompts to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()