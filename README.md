# FlightNLP – Natural Language Flight Request Parser

## Project overview

This project focuses on understanding natural-language flight requests and converting them into structured flight-booking information.

Users usually write flight requests in a messy or incomplete way, for example:

    Need something cheap from Tel Aviv to Rome next Friday, direct if possible

But a flight booking system needs structured fields such as:

    {
      "intent": "search_flight",
      "origin": "Tel Aviv",
      "destination": "Rome",
      "departure_date": "next Friday",
      "budget": "cheap",
      "layover_preference": "direct"
    }

The goal of this project is to compare several NLP approaches for extracting that structure from user text.

---

## Task definition

The input is a user flight-related request written in natural language.

The output is a structured schema containing:

- intent
- origin
- destination
- departure date
- return date
- trip type
- passengers
- cabin class
- budget preference
- layover preference
- time preference
- baggage
- airline preference
- booking reference
- missing fields

Example input:

    Can you reserve a flight from Paris to Vienna tomorrow round-trip make it under 300 dollars prefer direct

Example output:

    {
      "intent": "book_flight",
      "origin": "Paris",
      "destination": "Vienna",
      "departure_date": "tomorrow",
      "trip_type": "round-trip",
      "budget": "under 300 dollars",
      "layover_preference": "direct",
      "missing_fields": ["passengers"]
    }

---

## Intents

The project uses five intent labels:

| Intent | Meaning |
|---|---|
| `search_flight` | User wants to search for flight options |
| `book_flight` | User wants to book, reserve, or buy a flight |
| `change_flight` | User wants to change an existing flight |
| `cancel_flight` | User wants to cancel an existing flight |
| `baggage_question` | User asks about luggage or baggage |

---

## Slot fields

The project extracts the following slot fields:

| Slot | Meaning |
|---|---|
| `origin` | Departure city |
| `destination` | Arrival city |
| `departure_date` | When the user wants to fly |
| `return_date` | Return date, if mentioned |
| `trip_type` | One-way or round-trip |
| `passengers` | Number of passengers |
| `cabin_class` | Economy, business, etc. |
| `budget` | Budget preference |
| `layover_preference` | Direct flight / no layovers / layovers allowed |
| `time_preference` | Morning, evening, after work, etc. |
| `baggage` | Carry-on or checked bag information |
| `airline_preference` | Preferred airline |
| `booking_reference` | Existing booking reference |

---

## Dataset

The dataset was synthetically generated.

I first created a small manual dataset to test the schema and baseline logic. After that, I generated a larger synthetic dataset with more varied user phrasing.

The final synthetic dataset contains:

| Property | Value |
|---|---:|
| Total examples | 1000 |
| Average text length | 13.34 words |

Intent distribution:

| Intent | Count |
|---|---:|
| `search_flight` | 379 |
| `book_flight` | 245 |
| `baggage_question` | 133 |
| `cancel_flight` | 127 |
| `change_flight` | 116 |

The data was split into:

| Split | Examples |
|---|---:|
| Train | 700 |
| Validation | 150 |
| Test | 150 |

---

## Why synthetic data?

The task is specific to flight-booking requests and custom structured output. Instead of relying on an existing dataset, I generated examples with controlled labels.

This makes it possible to test:

- different user phrasings
- missing fields
- different intent types
- different city/date/passenger formats
- easy and hard examples

Some examples are simple:

    Book me a flight from Rome to Paris tomorrow

Some are more realistic and harder:

    Need help finding tickets leaving Athens and landing in Amsterdam with return for 2 passengers economy class

    Can you find tickets London -> Tokyo on tomorrow for 1 person with no layovers

    How much luggage do I get for Ryanair

---

## Models and baselines

### 1. Rule-based baseline

The rule baseline uses simple keyword matching and regular expressions.

It tries to extract:

- intent
- cities
- dates
- passengers
- trip type
- budget
- baggage
- booking reference

This baseline works on simple examples but struggles with more realistic phrasing.

Example difficult cases:

    London -> Tokyo
    leaving Tel Aviv and landing in Prague
    destination Berlin
    just me
    we are 4
    How much luggage do I get

---

### 2. TF-IDF + Logistic Regression intent classifier

This model predicts only the intent.

It uses:

- TF-IDF text features
- Logistic Regression classifier

This model performed very well because intent classification is easier than full slot extraction.

---

### 3. One classifier per slot

For slot extraction, I trained one classifier for each slot field.

Examples of slots:

- origin
- destination
- departure date
- trip type
- passengers
- budget
- baggage
- airline preference

This is a simple ML baseline, not a full NER model.

---

### 4. Combined ML pipeline

The combined ML pipeline uses:

- the intent classifier for intent prediction
- the slot classifiers for slot prediction
- rule-based missing-field logic after prediction

This gives a full structured output, but the task is much harder than intent classification alone.

---

## Results

| Method | Task | Examples | Intent Accuracy | Slot Accuracy | Missing Field Accuracy | Full Example Accuracy |
|---|---|---:|---:|---:|---:|---:|
| Rule-based baseline | Full schema extraction | 150 | 0.840 | 0.862 | 0.307 | 0.220 |
| TF-IDF Logistic Regression | Intent classification only | 150 | 0.993 | - | - | - |
| One classifier per slot | Slot classification only | 150 | - | 0.844 | - | - |
| TF-IDF intent + slot classifiers | Full schema extraction | 150 | 0.993 | 0.844 | 0.347 | 0.247 |
| Manual off-the-shelf LLM | Full schema extraction sample | 20 | 1.000 | 0.992 | 1.000 | 0.900 |

Note: the manual off-the-shelf LLM baseline was evaluated on a smaller 20-example sample, while the rule-based and classical ML methods were evaluated on the full 150-example test set.
---

## Main observations

The intent classification task was relatively easy. The TF-IDF Logistic Regression model reached very high intent accuracy.

Full schema extraction was much harder. Even when the intent was correct, the models often failed to extract all required fields correctly.

The hardest fields were:

- origin
- destination
- passengers

This makes sense because users can write these in many different ways:

    from Paris to Berlin
    Paris -> Berlin
    leaving Paris and landing in Berlin
    to Berlin from Paris
    we are 4
    just me
    4 tickets

The rule baseline was useful, but fragile. It performed well on patterns it was designed for, but failed on phrasing variations.

---

## Visual results

The project also creates basic result graphs in the `visuals/` folder:

| File | Meaning |
|---|---|
| `visuals/full_example_accuracy.png` | Compares full example accuracy between full-schema methods |
| `visuals/slot_missing_accuracy.png` | Compares slot accuracy and missing-field accuracy |
| `visuals/intent_distribution.png` | Shows the synthetic dataset intent distribution |

These graphs are generated by running:

    python src/plot_results.py

---

## Project structure

    FlightNLP/
    ├── data/
    │   ├── manual_examples.jsonl
    │   ├── manual_examples_prepared.jsonl
    │   ├── synthetic_examples.jsonl
    │   ├── train.jsonl
    │   ├── validation.jsonl
    │   └── test.jsonl
    │
    ├── results/
    │   ├── dataset_eda.json
    │   ├── rule_baseline_test_metrics.json
    │   ├── intent_classifier_metrics.json
    │   ├── slot_classifier_metrics.json
    │   ├── ml_pipeline_metrics.json
    │   ├── model_comparison.csv
    │   └── model_comparison.json
    │
    ├── src/
    │   ├── schema.py
    │   ├── prepare_manual_data.py
    │   ├── rule_baseline.py
    │   ├── evaluate_baseline.py
    │   ├── generate_synthetic_data.py
    │   ├── split_dataset.py
    │   ├── analyze_dataset.py
    │   ├── train_intent_classifier.py
    │   ├── print_intent_mistakes.py
    │   ├── train_slot_classifiers.py
    │   ├── evaluate_ml_pipeline.py
    │   ├── compare_results.py
    │   └── plot_results.py
    │
    ├── visuals/
    │   ├── full_example_accuracy.png
    │   ├── slot_missing_accuracy.png
    │   └── intent_distribution.png
    │
    └── README.md

---

## How to run

Generate synthetic data:

    python src/generate_synthetic_data.py

Split the dataset:

    python src/split_dataset.py

Run the rule baseline:

    python src/rule_baseline.py
    python src/evaluate_baseline.py

Analyze the dataset:

    python src/analyze_dataset.py

Train the intent classifier:

    python src/train_intent_classifier.py

Train the slot classifiers:

    python src/train_slot_classifiers.py

Evaluate the combined ML pipeline:

    python src/evaluate_ml_pipeline.py

Compare all current methods:

    python src/compare_results.py

Create graphs:

    python src/plot_results.py

---

## Current status

The project currently includes:

- synthetic dataset generation
- train / validation / test split
- dataset EDA
- rule-based baseline
- ML intent classifier
- ML slot classifiers
- combined ML pipeline
- evaluation scripts
- comparison table
- basic result graphs

---

## Next steps

The next planned steps are:

1. Add an off-the-shelf LLM prompt baseline.
2. Compare LLM JSON extraction with the rule-based and ML baselines.
3. Improve slot extraction with a stronger model.
4. Add deeper error analysis.
5. Prepare presentation slides and final report material.

---

## Notes

The main lesson so far is that intent classification is much easier than full structured extraction.

A simple TF-IDF model can predict the intent very accurately, but extracting every slot correctly is harder because users express the same information in many different ways.

This makes the project a good fit for comparing traditional NLP baselines with LLM-based extraction.