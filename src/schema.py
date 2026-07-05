INTENTS = [
    "search_flight",
    "book_flight",
    "change_flight",
    "cancel_flight",
    "baggage_question"
]

## Intents for the user input - J.z

SLOTS = [
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
    "booking_reference"
]

## Classification of inputs for the slots - J.z

REQUIRED_FIELDS = {
    "search_flight": [
        "origin",
        "destination",
        "departure_date",
        "trip_type",
        "passengers"
    ],
    "book_flight": [
        "origin",
        "destination",
        "departure_date",
        "trip_type",
        "passengers"
    ],
    "change_flight": [
        "booking_reference"
    ],
    "cancel_flight": [
        "booking_reference"
    ],
    "baggage_question": []
}

## Required fiels to enter. - J.z

def get_missing_fields(intent, example):
    """
    Receives an intent and an example dictionary.
    """

    required = REQUIRED_FIELDS.get(intent, [])
    missing = []

    for field in required:
        if field not in example or example[field] in [None, "", []]:
            missing.append(field)

    return missing