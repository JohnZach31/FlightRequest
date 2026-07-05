import json
import random
from pathlib import Path

from schema import get_missing_fields


# always use project root and not src/data by accident - J.z
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "data" / "synthetic_examples.jsonl"


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
]


DATES = [
    "tomorrow",
    "next Friday",
    "next month",
    "this weekend",
    "August 12",
    "September 3",
    "in two weeks",
    "around the 20th",
    "sometime next week",
    "after my exams",
]


BUDGETS = [
    "cheap",
    "low budget",
    "under 300 dollars",
    "under 500 dollars",
    "not too expensive",
    "as cheap as possible",
]


TIME_PREFERENCES = [
    "morning",
    "evening",
    "afternoon",
    "late night",
    "after work",
]


LAYOVER_PREFERENCES = [
    "direct",
    "no layovers",
    "short layover only",
    "layovers are fine",
]


CABIN_CLASSES = [
    "economy",
    "business",
]


AIRLINES = [
    "Lufthansa",
    "El Al",
    "Ryanair",
    "Wizz Air",
]


BOOKING_REFERENCES = [
    "AB1234",
    "ZX9021",
    "TLV7788",
    "BK2026",
]


def maybe(value, probability):
    # randomly removes fields because users rarely give everything
    if random.random() < probability:
        return value

    return None


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


def render_passengers(passengers):
    if passengers is None:
        return None

    if passengers == 1:
        return random.choice([
            "for one passenger",
            "for 1 person",
            "just me",
            "solo",
        ])

    return random.choice([
        f"for {passengers} passengers",
        f"for {passengers} people",
        f"{passengers} tickets",
        f"we are {passengers}",
    ])


def render_city_part(origin, destination):
    # several styles so regex baseline won't get a free 100%
    options = []

    if origin and destination:
        options.extend([
            f"from {origin} to {destination}",
            f"{origin} to {destination}",
            f"{origin} -> {destination}",
            f"leaving {origin} and landing in {destination}",
            f"departing {origin}, arriving {destination}",
            f"to {destination} from {origin}",
        ])

    elif origin:
        options.extend([
            f"from {origin}",
            f"leaving {origin}",
            f"departing {origin}",
        ])

    elif destination:
        options.extend([
            f"to {destination}",
            f"going to {destination}",
            f"destination {destination}",
            f"I want {destination}",
        ])

    if not options:
        return ""

    return random.choice(options)


def render_date_part(departure_date):
    if departure_date is None:
        return ""

    return random.choice([
        departure_date,
        f"on {departure_date}",
        f"for {departure_date}",
        f"around {departure_date}",
    ])


def render_trip_type(trip_type):
    if trip_type is None:
        return ""

    if trip_type == "one-way":
        return random.choice([
            "one-way",
            "one way",
            "single direction",
            "no return",
        ])

    return random.choice([
        "round-trip",
        "round trip",
        "with return",
        "return flight",
    ])


def generate_search_or_book_example(intent):
    origin = maybe(random.choice(CITIES), 0.82)
    destination = maybe(random.choice(CITIES), 0.88)

    while origin is not None and destination is not None and origin == destination:
        destination = random.choice(CITIES)

    departure_date = maybe(random.choice(DATES), 0.78)
    trip_type = maybe(random.choice(["one-way", "round-trip"]), 0.68)
    passengers = maybe(random.randint(1, 4), 0.68)

    budget = maybe(random.choice(BUDGETS), 0.45)
    time_preference = maybe(random.choice(TIME_PREFERENCES), 0.35)
    layover_preference = maybe(random.choice(LAYOVER_PREFERENCES), 0.35)
    cabin_class = maybe(random.choice(CABIN_CLASSES), 0.25)

    if intent == "search_flight":
        start = random.choice([
            "Find me",
            "I need",
            "Show me",
            "Can you find",
            "I'm looking for",
            "Need help finding",
            "Any flights",
            "Looking for tickets",
        ])
    else:
        start = random.choice([
            "Book me",
            "I want to book",
            "Please book",
            "Can you reserve",
            "I need to reserve",
            "I want to buy",
            "Get me tickets",
            "I want tickets",
        ])

    text_parts = [
        start,
        random.choice(["a flight", "tickets", "a plane ticket", ""]),
        render_city_part(origin, destination),
        render_date_part(departure_date),
        render_trip_type(trip_type),
    ]

    passenger_text = render_passengers(passengers)
    if passenger_text:
        text_parts.append(passenger_text)

    if budget:
        text_parts.append(random.choice([
            f"make it {budget}",
            f"prefer {budget}",
            f"budget is {budget}",
            budget,
        ]))

    if layover_preference:
        text_parts.append(random.choice([
            f"prefer {layover_preference}",
            layover_preference,
            f"with {layover_preference}",
        ]))

    if time_preference:
        text_parts.append(random.choice([
            f"in the {time_preference}",
            f"prefer {time_preference}",
            time_preference,
        ]))

    if cabin_class:
        text_parts.append(random.choice([
            f"in {cabin_class}",
            f"{cabin_class} class",
        ]))

    text = " ".join(part for part in text_parts if part).strip()

    # add some casual noise
    if random.random() < 0.12:
        text = text.lower()

    if random.random() < 0.10:
        text += " pls"

    if random.random() < 0.08:
        text = text.replace("you", "u")

    example = {
        "text": text,
        "intent": intent,
        "origin": origin,
        "destination": destination,
        "departure_date": departure_date,
        "return_date": None,
        "trip_type": trip_type,
        "passengers": passengers,
        "cabin_class": cabin_class,
        "budget": budget,
        "layover_preference": layover_preference,
        "time_preference": time_preference,
        "baggage": None,
        "airline_preference": None,
        "booking_reference": None,
    }

    missing_fields = get_missing_fields(intent, example)

    example["missing_fields"] = missing_fields
    example["next_question"] = make_question(missing_fields)

    return example


def generate_baggage_question():
    airline = maybe(random.choice(AIRLINES), 0.60)
    baggage = maybe(random.choice(["checked bag", "carry-on", "two checked bags"]), 0.70)

    text = random.choice([
        "Can I take baggage",
        "Do I get a checked bag",
        "What is the baggage policy",
        "Can I bring a carry-on",
        "Is luggage included",
        "How much luggage do I get",
        "Do I need to pay for bags",
    ])

    if airline:
        text += random.choice([
            f" with {airline}",
            f" on {airline}",
            f" for {airline}",
        ])

    example = {
        "text": text,
        "intent": "baggage_question",
        "origin": None,
        "destination": None,
        "departure_date": None,
        "return_date": None,
        "trip_type": None,
        "passengers": None,
        "cabin_class": None,
        "budget": None,
        "layover_preference": None,
        "time_preference": None,
        "baggage": baggage,
        "airline_preference": airline,
        "booking_reference": None,
        "missing_fields": [],
        "next_question": "",
    }

    return example


def generate_cancel_or_change(intent):
    booking_reference = maybe(random.choice(BOOKING_REFERENCES), 0.72)

    if intent == "cancel_flight":
        text = random.choice([
            "Cancel my flight",
            "I want to cancel my booking",
            "Please cancel my ticket",
            "Need to cancel this flight",
            "Can you cancel it",
        ])
    else:
        text = random.choice([
            "I need to change my flight",
            "Can I change my booking",
            "I want to move my flight",
            "Please change my booking",
            "Need to update the ticket",
        ])

    if booking_reference:
        text += random.choice([
            f" {booking_reference}",
            f" booking {booking_reference}",
            f" ref {booking_reference}",
            f" reference number {booking_reference}",
        ])

    example = {
        "text": text,
        "intent": intent,
        "origin": None,
        "destination": None,
        "departure_date": None,
        "return_date": None,
        "trip_type": None,
        "passengers": None,
        "cabin_class": None,
        "budget": None,
        "layover_preference": None,
        "time_preference": None,
        "baggage": None,
        "airline_preference": None,
        "booking_reference": booking_reference,
    }

    missing_fields = get_missing_fields(intent, example)

    example["missing_fields"] = missing_fields
    example["next_question"] = make_question(missing_fields)

    return example


def generate_example():
    # search/book appear more often because that is the main use case
    intent = random.choice([
        "search_flight",
        "search_flight",
        "search_flight",
        "book_flight",
        "book_flight",
        "baggage_question",
        "cancel_flight",
        "change_flight",
    ])

    if intent in ["search_flight", "book_flight"]:
        return generate_search_or_book_example(intent)

    if intent == "baggage_question":
        return generate_baggage_question()

    return generate_cancel_or_change(intent)


def main():
    random.seed(42)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    examples = []

    # now bigger because the basic flow already works - J.z
    for _ in range(1000):
        examples.append(generate_example())

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        for example in examples:
            file.write(json.dumps(example, ensure_ascii=False) + "\n")

    print(f"Saved {len(examples)} synthetic examples to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()