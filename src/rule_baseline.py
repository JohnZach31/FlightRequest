import json
import re
from pathlib import Path
from schema import get_missing_fields

#imports - both libraries and local.

# use project root so it does not create folders inside src by mistake - J.z
PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "test.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "results" / "rule_baseline_test_predictions.jsonl"

CITIES = [
    "Tel Aviv",
    "Rome",
    "Paris",
    "Berlin",
    "Amsterdam",
    "London",
    "Tokyo",
    "Athens",
    "Prague",
    "New York",
    "Barcelona",
    "Madrid",
    "Vienna",
    "Budapest",
    "Warsaw",
    "Lisbon",
    "Istanbul",
    "Copenhagen",
    "Stockholm",
    "Dublin",
]

# Cities list, can be changed


DATES = [
    "tomorrow",
    "next Friday",
    "next month",
    "August 12",
    "August 20",
    "this weekend",
    "September 3",
    "in two weeks",
    "around the 20th",
    "October 5",
    "November 15",
    "December 1",
    "January 10",
]


RETURN_DATES = [
    "August 20",
    "September 10",
    "next Sunday",
    "one week later",
    "three days later",
    "in two weeks",
    "around the 25th",
    "October 5",
    "November 15",
    "December 1",
    "January 10",
]

#Also return dates - fixed after first run

# Dates list, can be changed

def find_city_after_word(text, word):
    """
    Tries to find a city after words like 'from' or 'to'.
    Example:
    'from Tel Aviv to Rome' -> after 'from' gives Tel Aviv
    """
    text_lower = text.lower()
    for city in CITIES:
        pattern = word.lower() + " " + city.lower()

        if pattern in text_lower:
            return city

    return None

##regex to find a city after a specific word in the text. It checks for each city in the CITIES list and returns the first match found.


def find_date(text):
    text_lower = text.lower()

    for date in DATES:
        if date.lower() in text_lower:
            return date

    return None

##Same for dates

def find_return_date(text):
    # looks for return date only after the word returning
    # otherwise the code can confuse departure date and return date

    text_lower = text.lower()

    if "returning" not in text_lower:
        return None

    after_returning = text_lower.split("returning", 1)[1]

    for date in RETURN_DATES:
        if date.lower() in after_returning:
            return date

    return None

def find_passengers(text):
    text_lower = text.lower()

    if "two people" in text_lower:
        return 2

    if "one passenger" in text_lower:
        return 1

    match = re.search(r"for (\d+) passengers?", text_lower)

    if match:
        return int(match.group(1))

    return None


def find_booking_reference(text):
    # booking ref can be like AB1234 or TLV7788
    # made it 2-4 letters + 4 digits so it is less fragile - J.z

    match = re.search(r"\b[A-Z]{2,4}\d{4}\b", text)

    if match:
        return match.group(0)

    return None


def predict_intent(text):
    text_lower = text.lower()

    if "cancel" in text_lower:
        return "cancel_flight"

    if "change" in text_lower:
        return "change_flight"

    if "baggage" in text_lower or "checked bag" in text_lower or "carry-on" in text_lower:
        return "baggage_question"

    if "book" in text_lower or "reserve" in text_lower:
        return "book_flight"

    return "search_flight"

#some prediction of intents

def make_question(missing_fields):
    if not missing_fields:
        return ""

    readable = {
        "origin": "where you are flying from",
        "destination": "where you want to fly",
        "departure_date": "when you want to fly",
        "trip_type": "whether this is one-way or round-trip",
        "passengers": "how many passengers are flying",
        "booking_reference": "your booking reference",
    }

    parts = [readable.get(field, field) for field in missing_fields]

    if len(parts) == 1:
        return f"Can you tell me {parts[0]}?"

    return "Can you tell me " + ", ".join(parts[:-1]) + " and " + parts[-1] + "?"


def predict_example(text):
    intent = predict_intent(text)

    prediction = {
        "text": text,
        "intent": intent,
        "origin": find_city_after_word(text, "from"),
        "destination": find_city_after_word(text, "to"),
        "departure_date": find_date(text),
        "return_date": find_return_date(text),
        "trip_type": None,
        "passengers": find_passengers(text),
        "cabin_class": None,
        "budget": None,
        "layover_preference": None,
        "time_preference": None,
        "baggage": None,
        "airline_preference": None,
        "booking_reference": find_booking_reference(text),
    }

    text_lower = text.lower()

    if "one-way" in text_lower or "one way" in text_lower:
        prediction["trip_type"] = "one-way"

    if "round-trip" in text_lower or "round trip" in text_lower:
        prediction["trip_type"] = "round-trip"

    if "cheap" in text_lower:
        prediction["budget"] = "cheap"

    if "low budget" in text_lower:
        prediction["budget"] = "low budget"

    if "under 300 dollars" in text_lower:
        prediction["budget"] = "under 300 dollars"

    if "under 500 dollars" in text_lower:
        prediction["budget"] = "under 500 dollars"

    if "no layovers" in text_lower:
        prediction["layover_preference"] = "no layovers"

    if "morning" in text_lower:
        prediction["time_preference"] = "morning"

    if "afternoon" in text_lower:
        prediction["time_preference"] = "afternoon"

    if "late night" in text_lower:
        prediction["time_preference"] = "late night"

    if "economy" in text_lower:
        prediction["cabin_class"] = "economy"

    if "direct" in text_lower:
        prediction["layover_preference"] = "direct"

    if "evening" in text_lower:
        prediction["time_preference"] = "evening"

    if "business" in text_lower:
        prediction["cabin_class"] = "business"

    if "checked bag" in text_lower:
        prediction["baggage"] = "checked bag"

    if "lufthansa" in text_lower:
        prediction["airline_preference"] = "Lufthansa"

    missing_fields = get_missing_fields(intent, prediction)

    prediction["missing_fields"] = missing_fields
    prediction["next_question"] = make_question(missing_fields)

    return prediction

#some more predictions


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    predictions = []

    with INPUT_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            gold_example = json.loads(line)

            predicted_example = predict_example(gold_example["text"])

            predictions.append({
                "text": gold_example["text"],
                "gold": gold_example,
                "prediction": predicted_example,
            })

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        for row in predictions:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Saved {len(predictions)} predictions to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()