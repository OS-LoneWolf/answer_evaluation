#!/usr/bin/env python3

"""
PHASE 2 - STAGE 4
Final Evaluation / Reporting Layer

Consumes:
    Stage 1:
        - TF-IDF baseline
        - Semantic baseline

    Stage 3:
        - LLM-Rubric-inspired calibrated predictions

Produces:
    results/phase2_stage4/
        final_question_scores.csv
        final_student_scores.csv
        final_results.json
        final_summary.json

IMPORTANT:
    This stage does NOT call Ollama.
    This stage does NOT require Sentence Transformers.
    This stage does NOT retrain the calibration model.

    Stage 4 is deliberately separated from the expensive
    model-evaluation stages so that the entire project can
    be transferred between machines.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path("/home/robot/Projects/intership")

DATA_DIR = PROJECT_ROOT / "student_evaluation" / "data"

RESULTS_DIR = PROJECT_ROOT / "student_evaluation" / "results"

STAGE1_DIR = RESULTS_DIR / "phase2_stage1"
STAGE3_DIR = RESULTS_DIR / "phase2_stage3"
STAGE4_DIR = RESULTS_DIR / "phase2_stage4"


SEMANTIC_CSV = (
    STAGE1_DIR
    / "semantic"
    / "semantic_question_scores.csv"
)

TFIDF_CSV = (
    STAGE1_DIR
    / "tfidf"
    / "tfidf_question_scores.csv"
)


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None

    try:
        result = float(value)

        if math.isnan(result) or math.isinf(result):
            return None

        return result

    except (TypeError, ValueError):
        return None


def find_value(
    record: Dict[str, Any],
    candidates: List[str],
) -> Any:

    lowered = {
        str(k).lower(): v
        for k, v in record.items()
    }

    for candidate in candidates:

        if candidate.lower() in lowered:
            return lowered[candidate.lower()]

    return None


def find_stage3_files() -> List[Path]:

    if not STAGE3_DIR.exists():
        return []

    files = []

    for pattern in ("*.json", "*.csv"):
        files.extend(STAGE3_DIR.rglob(pattern))

    return sorted(files)


def load_first_matching_stage3_file() -> Optional[Path]:

    files = find_stage3_files()

    if not files:
        return None

    # Prefer files whose names strongly suggest they contain
    # final/calibrated/question predictions.

    priority_terms = [
        "calibrated",
        "calibration",
        "prediction",
        "predictions",
        "question",
        "result",
        "score",
    ]

    ranked = []

    for path in files:

        name = path.name.lower()

        priority = sum(
            1
            for term in priority_terms
            if term in name
        )

        ranked.append((priority, path))

    ranked.sort(
        key=lambda x: (-x[0], str(x[1]))
    )

    return ranked[0][1]


# ============================================================
# STAGE 1 LOADING
# ============================================================

def load_stage1_csv(path: Path) -> List[Dict[str, Any]]:

    if not path.exists():
        raise FileNotFoundError(
            f"Required Stage 1 file not found:\n{path}"
        )

    rows = load_csv(path)

    converted = []

    for row in rows:

        converted.append({
            k: (
                safe_float(v)
                if v not in ("", None)
                else None
            )
            for k, v in row.items()
        })

        # Restore identifiers as strings.
        for key in (
            "student_id",
            "student",
            "question_id",
            "question",
            "qid",
        ):
            if key in row:
                converted[-1][key] = row[key]

    return converted


def identify_key(
    record: Dict[str, Any],
    candidates: List[str],
) -> Optional[str]:

    lowered = {
        str(k).lower(): k
        for k in record.keys()
    }

    for candidate in candidates:

        if candidate.lower() in lowered:
            return lowered[candidate.lower()]

    return None


def make_record_key(
    record: Dict[str, Any],
) -> Optional[tuple]:

    student_key = identify_key(
        record,
        ["student_id", "student", "studentid"],
    )

    question_key = identify_key(
        record,
        ["question_id", "question", "qid", "questionid"],
    )

    if student_key is None or question_key is None:
        return None

    return (
        str(record[student_key]),
        str(record[question_key]),
    )


# ============================================================
# STAGE 3 LOADING
# ============================================================

def load_stage3_records(path: Path) -> List[Dict[str, Any]]:

    print(f"Stage 3 artifact: {path}")

    if path.suffix.lower() == ".csv":
        return load_csv(path)

    data = load_json(path)

    # Common possible Stage 3 structures.

    if isinstance(data, list):
        return data

    if isinstance(data, dict):

        for key in (
            "records",
            "results",
            "predictions",
            "question_scores",
            "evaluations",
            "data",
        ):

            value = data.get(key)

            if isinstance(value, list):
                return value

    raise ValueError(
        "Could not identify a list of Stage 3 records "
        f"in {path}"
    )


# ============================================================
# SCORE EXTRACTION
# ============================================================

def extract_final_score(
    record: Dict[str, Any],
) -> Optional[float]:

    candidates = [
        "calibrated_score",
        "predicted_human_score",
        "predicted_score",
        "final_score",
        "calibration_score",
        "score",
    ]

    value = find_value(record, candidates)

    return safe_float(value)


def extract_human_score(
    record: Dict[str, Any],
) -> Optional[float]:

    candidates = [
        "human_score",
        "benchmark_score",
        "human",
        "gold_score",
        "target",
        "actual_score",
    ]

    return safe_float(
        find_value(record, candidates)
    )


def extract_dimension_score(
    record: Dict[str, Any],
    dimension: str,
) -> Optional[float]:

    candidates = [
        dimension,
        f"{dimension}_score",
        f"calibrated_{dimension}",
        f"predicted_{dimension}",
    ]

    return safe_float(
        find_value(record, candidates)
    )


# ============================================================
# METRICS
# ============================================================

def mae(
    predictions: List[float],
    actuals: List[float],
) -> Optional[float]:

    if not predictions:
        return None

    return sum(
        abs(p - a)
        for p, a in zip(predictions, actuals)
    ) / len(predictions)


def rmse(
    predictions: List[float],
    actuals: List[float],
) -> Optional[float]:

    if not predictions:
        return None

    return math.sqrt(
        sum(
            (p - a) ** 2
            for p, a in zip(predictions, actuals)
        ) / len(predictions)
    )


def pearson(
    x: List[float],
    y: List[float],
) -> Optional[float]:

    if len(x) < 2:
        return None

    mean_x = statistics.mean(x)
    mean_y = statistics.mean(y)

    numerator = sum(
        (a - mean_x) * (b - mean_y)
        for a, b in zip(x, y)
    )

    denominator_x = math.sqrt(
        sum((a - mean_x) ** 2 for a in x)
    )

    denominator_y = math.sqrt(
        sum((b - mean_y) ** 2 for b in y)
    )

    denominator = denominator_x * denominator_y

    if denominator == 0:
        return None

    return numerator / denominator


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("PHASE 2 - STAGE 4")
    print("Final Evaluation / Reporting Layer")
    print("=" * 70)

    print()
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Data         : {DATA_DIR}")
    print(f"Stage 1      : {STAGE1_DIR}")
    print(f"Stage 3      : {STAGE3_DIR}")
    print(f"Output       : {STAGE4_DIR}")

    # --------------------------------------------------------
    # Check Stage 1
    # --------------------------------------------------------

    print()
    print("Checking Stage 1 artifacts...")

    semantic_rows = load_stage1_csv(
        SEMANTIC_CSV
    )

    tfidf_rows = load_stage1_csv(
        TFIDF_CSV
    )

    print(
        f"Semantic records : {len(semantic_rows)}"
    )

    print(
        f"TF-IDF records   : {len(tfidf_rows)}"
    )

    # --------------------------------------------------------
    # Check Stage 3
    # --------------------------------------------------------

    print()
    print("Checking Stage 3 artifacts...")

    stage3_path = load_first_matching_stage3_file()

    if stage3_path is None:

        print()
        print("=" * 70)
        print("STAGE 4 NOT READY YET")
        print("=" * 70)
        print()
        print(
            "No Stage 3 output was found."
        )
        print()
        print(
            "Expected Stage 3 directory:"
        )
        print(
            f"  {STAGE3_DIR}"
        )
        print()
        print(
            "This is expected if your friend has not "
            "provided the Stage 3 output yet."
        )
        print()
        print(
            "Stage 1 artifacts were found successfully."
        )
        print()
        return

    stage3_rows = load_stage3_records(
        stage3_path
    )

    print(
        f"Stage 3 records  : {len(stage3_rows)}"
    )

    # --------------------------------------------------------
    # Index Stage 1
    # --------------------------------------------------------

    semantic_index = {}

    for row in semantic_rows:

        key = make_record_key(row)

        if key:
            semantic_index[key] = row

    tfidf_index = {}

    for row in tfidf_rows:

        key = make_record_key(row)

        if key:
            tfidf_index[key] = row

    # --------------------------------------------------------
    # Build final records
    # --------------------------------------------------------

    final_records = []

    for row in stage3_rows:

        key = make_record_key(row)

        if key is None:
            continue

        student_id, question_id = key

        final_score = extract_final_score(row)

        human_score = extract_human_score(row)

        semantic_row = semantic_index.get(key)

        tfidf_row = tfidf_index.get(key)

        semantic_score = None
        tfidf_score = None

        if semantic_row:

            semantic_score = safe_float(
                find_value(
                    semantic_row,
                    [
                        "semantic_score",
                        "score",
                        "predicted_score",
                        "semantic",
                    ],
                )
            )

        if tfidf_row:

            tfidf_score = safe_float(
                find_value(
                    tfidf_row,
                    [
                        "tfidf_score",
                        "score",
                        "predicted_score",
                        "tfidf",
                    ],
                )
            )

        record = {
            "student_id": student_id,
            "question_id": question_id,

            "human_score": human_score,

            "tfidf_score": tfidf_score,

            "semantic_score": semantic_score,

            "final_score": final_score,

            "correctness": extract_dimension_score(
                row,
                "correctness",
            ),

            "completeness": extract_dimension_score(
                row,
                "completeness",
            ),

            "relevance": extract_dimension_score(
                row,
                "relevance",
            ),

            "clarity": extract_dimension_score(
                row,
                "clarity",
            ),
        }

        final_records.append(record)

    if not final_records:

        raise RuntimeError(
            "Stage 3 was found, but no records could be "
            "matched using student_id + question_id."
        )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    metric_predictions = []
    metric_actuals = []

    for row in final_records:

        prediction = row["final_score"]
        actual = row["human_score"]

        if (
            prediction is not None
            and actual is not None
        ):

            metric_predictions.append(prediction)
            metric_actuals.append(actual)

    metrics = {
        "records": len(final_records),
        "scored_records": len(metric_predictions),
        "mae": mae(
            metric_predictions,
            metric_actuals,
        ),
        "rmse": rmse(
            metric_predictions,
            metric_actuals,
        ),
        "pearson_r": pearson(
            metric_predictions,
            metric_actuals,
        ),
    }

    # --------------------------------------------------------
    # Student aggregation
    # --------------------------------------------------------

    students = {}

    for row in final_records:

        student = row["student_id"]

        students.setdefault(
            student,
            [],
        ).append(row)

    student_results = []

    for student_id, rows in students.items():

        human_scores = [
            r["human_score"]
            for r in rows
            if r["human_score"] is not None
        ]

        final_scores = [
            r["final_score"]
            for r in rows
            if r["final_score"] is not None
        ]

        semantic_scores = [
            r["semantic_score"]
            for r in rows
            if r["semantic_score"] is not None
        ]

        tfidf_scores = [
            r["tfidf_score"]
            for r in rows
            if r["tfidf_score"] is not None
        ]

        result = {
            "student_id": student_id,
            "questions": len(rows),

            "human_score": (
                statistics.mean(human_scores)
                if human_scores
                else None
            ),

            "final_score": (
                statistics.mean(final_scores)
                if final_scores
                else None
            ),

            "semantic_score": (
                statistics.mean(semantic_scores)
                if semantic_scores
                else None
            ),

            "tfidf_score": (
                statistics.mean(tfidf_scores)
                if tfidf_scores
                else None
            ),
        }

        if (
            result["human_score"] is not None
            and result["final_score"] is not None
        ):

            result["absolute_error"] = abs(
                result["final_score"]
                - result["human_score"]
            )

        else:

            result["absolute_error"] = None

        student_results.append(result)

    student_results.sort(
        key=lambda x: x["student_id"]
    )

    # --------------------------------------------------------
    # Comparison summary
    # --------------------------------------------------------

    comparison = {}

    # Semantic baseline

    semantic_predictions = []
    semantic_actuals = []

    for row in final_records:

        if (
            row["semantic_score"] is not None
            and row["human_score"] is not None
        ):

            semantic_predictions.append(
                row["semantic_score"]
            )

            semantic_actuals.append(
                row["human_score"]
            )

    comparison["semantic"] = {
        "mae": mae(
            semantic_predictions,
            semantic_actuals,
        ),
        "rmse": rmse(
            semantic_predictions,
            semantic_actuals,
        ),
        "pearson_r": pearson(
            semantic_predictions,
            semantic_actuals,
        ),
    }

    # TF-IDF baseline

    tfidf_predictions = []
    tfidf_actuals = []

    for row in final_records:

        if (
            row["tfidf_score"] is not None
            and row["human_score"] is not None
        ):

            tfidf_predictions.append(
                row["tfidf_score"]
            )

            tfidf_actuals.append(
                row["human_score"]
            )

    comparison["tfidf"] = {
        "mae": mae(
            tfidf_predictions,
            tfidf_actuals,
        ),
        "rmse": rmse(
            tfidf_predictions,
            tfidf_actuals,
        ),
        "pearson_r": pearson(
            tfidf_predictions,
            tfidf_actuals,
        ),
    }

    comparison["llm_rubric_calibrated"] = metrics

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    STAGE4_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Question CSV
    # --------------------------------------------------------

    question_csv = (
        STAGE4_DIR
        / "final_question_scores.csv"
    )

    fieldnames = [
        "student_id",
        "question_id",
        "human_score",
        "tfidf_score",
        "semantic_score",
        "final_score",
        "correctness",
        "completeness",
        "relevance",
        "clarity",
    ]

    with question_csv.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in final_records:
            writer.writerow(row)

    # --------------------------------------------------------
    # Student CSV
    # --------------------------------------------------------

    student_csv = (
        STAGE4_DIR
        / "final_student_scores.csv"
    )

    student_fields = [
        "student_id",
        "questions",
        "human_score",
        "final_score",
        "semantic_score",
        "tfidf_score",
        "absolute_error",
    ]

    with student_csv.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=student_fields,
        )

        writer.writeheader()

        for row in student_results:
            writer.writerow(row)

    # --------------------------------------------------------
    # Full JSON
    # --------------------------------------------------------

    results_json = (
        STAGE4_DIR
        / "final_results.json"
    )

    payload = {
        "stage": "phase2_stage4",
        "description": (
            "Final evaluation and reporting "
            "using LLM-Rubric-inspired calibration."
        ),
        "records": final_records,
        "student_results": student_results,
        "metrics": metrics,
        "comparison": comparison,
    }

    with results_json.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            payload,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # --------------------------------------------------------
    # Summary JSON
    # --------------------------------------------------------

    summary_json = (
        STAGE4_DIR
        / "final_summary.json"
    )

    summary = {
        "stage": "phase2_stage4",
        "records": len(final_records),
        "students": len(student_results),
        "metrics": metrics,
        "comparison": comparison,
    }

    with summary_json.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("FINAL EVALUATION RESULTS")
    print("-" * 70)

    print(
        f"Records : {metrics['records']}"
    )

    print(
        f"MAE     : {metrics['mae']:.4f}"
        if metrics["mae"] is not None
        else "MAE     : N/A"
    )

    print(
        f"RMSE    : {metrics['rmse']:.4f}"
        if metrics["rmse"] is not None
        else "RMSE    : N/A"
    )

    print(
        f"Pearson : {metrics['pearson_r']:.4f}"
        if metrics["pearson_r"] is not None
        else "Pearson : N/A"
    )

    print()
    print("Student-level results:")

    for row in student_results:

        human = row["human_score"]
        final = row["final_score"]

        if human is not None and final is not None:

            error = abs(
                human - final
            )

            print(
                f"{row['student_id']} "
                f"human={human:.2f} "
                f"final={final:.2f} "
                f"error={error:.2f}"
            )

        else:

            print(
                f"{row['student_id']} "
                f"insufficient score data"
            )

    print()
    print("Baseline comparison:")

    for name, values in comparison.items():

        print(
            f"{name:25s} "
            f"MAE={values['mae']:.4f}"
            if values["mae"] is not None
            else f"{name:25s} MAE=N/A"
        )

    print()
    print("Files written:")

    print(question_csv)
    print(student_csv)
    print(results_json)
    print(summary_json)

    print()
    print("=" * 70)
    print("PHASE 2 - STAGE 4 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
