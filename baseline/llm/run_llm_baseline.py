import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import requests
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path("/home/robot/Projects/intership")

DATA_DIR = PROJECT_ROOT / "student_evaluation" / "data"

RESULTS_DIR = (
    PROJECT_ROOT
    / "student_evaluation"
    / "results"
    / "phase2_stage2"
    / "llm"
)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ANSWERS_FILE = DATA_DIR / "dsa_model_answers.json"
BENCHMARK_FILE = DATA_DIR / "provisional_human_benchmark.csv"

OUTPUT_JSON = RESULTS_DIR / "llm_results.json"
OUTPUT_CSV = RESULTS_DIR / "llm_question_scores.csv"
OUTPUT_SUMMARY = RESULTS_DIR / "llm_summary.json"


# ============================================================
# OLLAMA
# ============================================================

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen3:4b"

REQUEST_TIMEOUT = 180
MAX_RETRIES = 2


# ============================================================
# RUBRIC
# ============================================================

CRITERIA = {
    "correctness": {
        "weight": 0.40,
        "description": (
            "How factually and technically correct is the answer?"
        ),
    },
    "completeness": {
        "weight": 0.30,
        "description": (
            "How completely does the answer cover the important "
            "knowledge required by the question?"
        ),
    },
    "relevance": {
        "weight": 0.15,
        "description": (
            "How directly does the answer address the question "
            "without unnecessary unrelated material?"
        ),
    },
    "clarity": {
        "weight": 0.15,
        "description": (
            "How clearly and understandably is the answer expressed?"
        ),
    },
}


# ============================================================
# JSON SCHEMA FOR OLLAMA
# ============================================================

DISTRIBUTION_PROPERTIES = {
    str(i): {
        "type": "number",
        "minimum": 0,
        "maximum": 1,
    }
    for i in range(1, 11)
}

DISTRIBUTION_REQUIRED = [
    str(i) for i in range(1, 11)
]

DISTRIBUTION_SCHEMA = {
    "type": "object",
    "properties": DISTRIBUTION_PROPERTIES,
    "required": DISTRIBUTION_REQUIRED,
    "additionalProperties": False,
}

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        criterion: DISTRIBUTION_SCHEMA
        for criterion in CRITERIA
    },
    "required": list(CRITERIA.keys()),
    "additionalProperties": False,
}


# ============================================================
# DATA LOADING
# ============================================================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_students():
    students = []

    for path in sorted(DATA_DIR.glob("student_*.json")):
        data = load_json(path)

        if "student" not in data or "answers" not in data:
            raise ValueError(f"Invalid student file: {path}")

        students.append(data)

    if not students:
        raise RuntimeError("No student_*.json files found.")

    return students


def load_benchmark():
    benchmark = {}

    with open(BENCHMARK_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        required = {"student", "question", "score_0_10"}

        if not required.issubset(reader.fieldnames):
            raise ValueError(
                "Benchmark must contain: "
                "student, question, score_0_10"
            )

        for row in reader:
            key = (row["student"], row["question"])
            benchmark[key] = float(row["score_0_10"])

    return benchmark


def build_records(model_data, students, benchmark):
    records = []

    for student_data in students:

        student_id = student_data["student"]

        for question_id, student_answer in (
            student_data["answers"].items()
        ):

            if question_id not in model_data:
                print(
                    f"WARNING: missing model answer for "
                    f"{question_id}"
                )
                continue

            key = (student_id, question_id)

            if key not in benchmark:
                print(
                    f"WARNING: missing benchmark for "
                    f"{student_id}/{question_id}"
                )
                continue

            model_entry = model_data[question_id]

            if isinstance(model_entry, dict):
                model_answer = model_entry["model_answer"]
            else:
                model_answer = model_entry

            records.append(
                {
                    "student": student_id,
                    "question": question_id,
                    "student_answer": student_answer,
                    "model_answer": model_answer,
                    "human_score": benchmark[key],
                }
            )

    return records


# ============================================================
# PROMPT
# ============================================================

def build_prompt(record):
    return f"""
Evaluate this student answer academically.

Question:
{record["question"]}

Reference answer:
{record["model_answer"]}

Student answer:
{record["student_answer"]}

Score the student independently on:

1. correctness — factual and technical correctness
2. completeness — coverage of important required concepts
3. relevance — directness and absence of unnecessary material
4. clarity — understandable and well-organized expression

For each dimension, return a probability distribution over scores 1
through 10.

The probabilities represent your uncertainty about the appropriate
score. A probability distribution must sum to 1.

Do not compare wording. Evaluate meaning and knowledge.

Return ONLY the requested JSON structure.
""".strip()


# ============================================================
# OLLAMA CALL
# ============================================================

def call_ollama(prompt):
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "stream": False,
        "format": OUTPUT_SCHEMA,
        "options": {
            "temperature": 0,
        },
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )

    response.raise_for_status()

    data = response.json()

    if "message" not in data:
        raise RuntimeError(
            f"Unexpected Ollama response: {data}"
        )

    message = data["message"]

    content = message.get("content", "").strip()

    if not content:
        raise RuntimeError(
            "Ollama returned empty message.content"
        )

    parsed = json.loads(content)

    return parsed, data


# ============================================================
# DISTRIBUTION VALIDATION
# ============================================================

def normalize_distribution(distribution, criterion):
    if not isinstance(distribution, dict):
        raise ValueError(
            f"{criterion}: distribution is not an object"
        )

    values = {}

    for score in range(1, 11):
        key = str(score)

        if key not in distribution:
            raise ValueError(
                f"{criterion}: missing score {score}"
            )

        value = float(distribution[key])

        if not math.isfinite(value):
            raise ValueError(
                f"{criterion}: invalid probability"
            )

        if value < 0:
            raise ValueError(
                f"{criterion}: negative probability"
            )

        values[key] = value

    total = sum(values.values())

    if total <= 0:
        raise ValueError(
            f"{criterion}: probability sum is zero"
        )

    return {
        key: value / total
        for key, value in values.items()
    }


def validate_output(data):
    if not isinstance(data, dict):
        raise ValueError("LLM output is not an object")

    normalized = {}

    for criterion in CRITERIA:
        if criterion not in data:
            raise ValueError(
                f"Missing criterion: {criterion}"
            )

        normalized[criterion] = normalize_distribution(
            data[criterion],
            criterion,
        )

    return normalized


# ============================================================
# EXPECTED SCORE
# ============================================================

def expected_score(distribution):
    return sum(
        int(score) * probability
        for score, probability in distribution.items()
    )


def calculate_dimension_scores(distributions):
    return {
        criterion: expected_score(
            distributions[criterion]
        )
        for criterion in CRITERIA
    }


def calculate_overall(scores):
    return sum(
        CRITERIA[criterion]["weight"] * scores[criterion]
        for criterion in CRITERIA
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    result = {
        "n": int(len(y_true)),
        "mae": float(
            mean_absolute_error(y_true, y_pred)
        ),
        "rmse": float(
            math.sqrt(
                mean_squared_error(y_true, y_pred)
            )
        ),
        "pearson_r": None,
        "spearman_rho": None,
    }

    if (
        len(y_true) >= 2
        and np.std(y_true) > 0
        and np.std(y_pred) > 0
    ):
        result["pearson_r"] = float(
            pearsonr(y_true, y_pred).statistic
        )

    if len(y_true) >= 2 and np.std(y_true) > 0:
        result["spearman_rho"] = float(
            spearmanr(y_true, y_pred).statistic
        )

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("PHASE 2 - STAGE 2")
    print("Local Qwen3:4B LLM Evaluator")
    print("=" * 70)

    print()
    print(f"Ollama model : {OLLAMA_MODEL}")
    print(f"Ollama URL   : {OLLAMA_URL}")
    print(f"Data         : {DATA_DIR}")
    print(f"Results      : {RESULTS_DIR}")
    print()

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    model_data = load_json(MODEL_ANSWERS_FILE)
    students = load_students()
    benchmark = load_benchmark()

    records = build_records(
        model_data,
        students,
        benchmark,
    )

    print(f"Students loaded    : {len(students)}")
    print(f"Benchmark rows     : {len(benchmark)}")
    print(f"Evaluation records : {len(records)}")
    print()

    # --------------------------------------------------------
    # Verify Ollama
    # --------------------------------------------------------

    try:
        test_response = requests.get(
            "http://localhost:11434/api/tags",
            timeout=10,
        )

        test_response.raise_for_status()

        models = test_response.json().get("models", [])

        model_names = {
            model.get("name")
            for model in models
        }

        if OLLAMA_MODEL not in model_names:
            raise RuntimeError(
                f"{OLLAMA_MODEL} not found in Ollama."
            )

    except Exception as e:
        raise RuntimeError(
            "Could not connect to local Ollama. "
            "Make sure Ollama is running.\n"
            f"Error: {e}"
        )

    print("Ollama connection : OK")
    print(f"Model available   : {OLLAMA_MODEL}")
    print()

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    results = []

    y_true = []
    y_pred = []

    total = len(records)

    for index, record in enumerate(records, start=1):

        print(
            f"[{index}/{total}] "
            f"{record['student']} / "
            f"{record['question']}",
            flush=True,
        )

        prompt = build_prompt(record)

        parsed = None
        raw_response = None
        error = None

        for attempt in range(1, MAX_RETRIES + 2):

            try:
                parsed, raw_response = call_ollama(prompt)

                parsed = validate_output(parsed)

                break

            except Exception as exc:

                error = str(exc)

                print(
                    f"  attempt {attempt} failed: {error}",
                    flush=True,
                )

                if attempt <= MAX_RETRIES:
                    time.sleep(1)

        if parsed is None:

            print("  FAILED", flush=True)

            results.append(
                {
                    "student": record["student"],
                    "question": record["question"],
                    "human_benchmark_score_0_10":
                        record["human_score"],
                    "evaluation_status": "failed",
                    "error": error,
                    "raw_ollama_response": raw_response,
                }
            )

            continue

        dimension_scores = calculate_dimension_scores(
            parsed
        )

        overall = calculate_overall(
            dimension_scores
        )

        human = record["human_score"]

        result = {
            "student": record["student"],
            "question": record["question"],
            "human_benchmark_score_0_10": human,

            "dimension_probability_distributions":
                parsed,

            "dimension_expected_scores":
                dimension_scores,

            "weighted_overall_score_0_10":
                overall,

            "absolute_error":
                abs(overall - human),

            "signed_error":
                overall - human,

            "raw_ollama_response":
                raw_response,

            "evaluation_status":
                "success",
        }

        results.append(result)

        y_true.append(human)
        y_pred.append(overall)

        print(
            f"  Correctness : "
            f"{dimension_scores['correctness']:.2f}"
        )

        print(
            f"  Completeness: "
            f"{dimension_scores['completeness']:.2f}"
        )

        print(
            f"  Relevance   : "
            f"{dimension_scores['relevance']:.2f}"
        )

        print(
            f"  Clarity     : "
            f"{dimension_scores['clarity']:.2f}"
        )

        print(
            f"  Overall     : "
            f"{overall:.2f}"
        )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    metrics = calculate_metrics(
        y_true,
        y_pred,
    )

    # --------------------------------------------------------
    # Student summary
    # --------------------------------------------------------

    student_groups = {}

    for result in results:

        if result["evaluation_status"] != "success":
            continue

        student = result["student"]

        student_groups.setdefault(
            student,
            {
                "human": [],
                "predicted": [],
            },
        )

        student_groups[student]["human"].append(
            result["human_benchmark_score_0_10"]
        )

        student_groups[student]["predicted"].append(
            result["weighted_overall_score_0_10"]
        )

    student_summary = {}

    for student, values in student_groups.items():

        human_mean = float(
            np.mean(values["human"])
        )

        predicted_mean = float(
            np.mean(values["predicted"])
        )

        student_summary[student] = {
            "human_mean_score_0_10":
                human_mean,

            "llm_mean_score_0_10":
                predicted_mean,

            "absolute_error":
                abs(predicted_mean - human_mean),
        }

    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

    output = {
        "experiment":
            "phase2_stage2_local_llm_baseline",

        "model": {
            "provider": "Ollama",
            "model": OLLAMA_MODEL,
            "offline_inference": True,
            "api": OLLAMA_URL,
        },

        "rubric": CRITERIA,

        "scoring": {
            "dimension_score":
                "sum(score * probability)",

            "overall_score":
                "0.40*correctness + "
                "0.30*completeness + "
                "0.15*relevance + "
                "0.15*clarity",
        },

        "dataset": {
            "students": len(students),
            "records_attempted": len(records),
            "records_successful": len(y_true),
            "records_failed":
                len(records) - len(y_true),
        },

        "metrics": metrics,

        "student_summary":
            student_summary,

        "question_results":
            results,
    }

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    fieldnames = [
        "student",
        "question",
        "human_benchmark_score_0_10",
        "correctness_expected",
        "completeness_expected",
        "relevance_expected",
        "clarity_expected",
        "weighted_overall_score_0_10",
        "absolute_error",
        "signed_error",
        "evaluation_status",
    ]

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for result in results:

            dimensions = result.get(
                "dimension_expected_scores"
            )

            writer.writerow(
                {
                    "student":
                        result["student"],

                    "question":
                        result["question"],

                    "human_benchmark_score_0_10":
                        result[
                            "human_benchmark_score_0_10"
                        ],

                    "correctness_expected":
                        (
                            dimensions["correctness"]
                            if dimensions
                            else None
                        ),

                    "completeness_expected":
                        (
                            dimensions["completeness"]
                            if dimensions
                            else None
                        ),

                    "relevance_expected":
                        (
                            dimensions["relevance"]
                            if dimensions
                            else None
                        ),

                    "clarity_expected":
                        (
                            dimensions["clarity"]
                            if dimensions
                            else None
                        ),

                    "weighted_overall_score_0_10":
                        result.get(
                            "weighted_overall_score_0_10"
                        ),

                    "absolute_error":
                        result.get("absolute_error"),

                    "signed_error":
                        result.get("signed_error"),

                    "evaluation_status":
                        result["evaluation_status"],
                }
            )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = {
        "experiment":
            "phase2_stage2_local_llm_baseline",

        "model":
            OLLAMA_MODEL,

        "metrics":
            metrics,

        "student_summary":
            student_summary,

        "records_attempted":
            len(records),

        "records_successful":
            len(y_true),

        "records_failed":
            len(records) - len(y_true),
    }

    with open(
        OUTPUT_SUMMARY,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Terminal summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("LOCAL LLM RESULTS")
    print("=" * 70)

    print(
        f"Records attempted : {len(records)}"
    )

    print(
        f"Successful        : {len(y_true)}"
    )

    print(
        f"Failed            : "
        f"{len(records) - len(y_true)}"
    )

    if y_true:

        print(
            f"MAE               : "
            f"{metrics['mae']:.4f}"
        )

        print(
            f"RMSE              : "
            f"{metrics['rmse']:.4f}"
        )

        if metrics["pearson_r"] is not None:
            print(
                f"Pearson r         : "
                f"{metrics['pearson_r']:.4f}"
            )

        if metrics["spearman_rho"] is not None:
            print(
                f"Spearman rho      : "
                f"{metrics['spearman_rho']:.4f}"
            )

    print()
    print("Student-level results:")
    print()

    for student in sorted(student_summary):

        s = student_summary[student]

        print(
            f"{student:10s} "
            f"human="
            f"{s['human_mean_score_0_10']:.2f} "
            f"llm="
            f"{s['llm_mean_score_0_10']:.2f} "
            f"error="
            f"{s['absolute_error']:.2f}"
        )

    print()
    print("Files written:")
    print(OUTPUT_JSON)
    print(OUTPUT_CSV)
    print(OUTPUT_SUMMARY)

    print()
    print("Stage 2 local LLM baseline complete.")


if __name__ == "__main__":
    main()
