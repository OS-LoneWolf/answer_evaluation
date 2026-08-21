import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sentence_transformers import SentenceTransformer


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path("/home/robot/Projects/intership")

DATA_DIR = PROJECT_ROOT / "student_evaluation" / "data"

RESULTS_DIR = (
    PROJECT_ROOT
    / "student_evaluation"
    / "results"
    / "phase2_stage1"
    / "semantic"
)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


MODEL_ANSWERS_FILE = DATA_DIR / "dsa_model_answers.json"
BENCHMARK_FILE = DATA_DIR / "provisional_human_benchmark.csv"


OUTPUT_JSON = RESULTS_DIR / "semantic_results.json"
OUTPUT_CSV = RESULTS_DIR / "semantic_question_scores.csv"
OUTPUT_SUMMARY = RESULTS_DIR / "semantic_summary.json"


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# ============================================================
# LOAD JSON
# ============================================================

def load_json(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


# ============================================================
# LOAD STUDENTS
# ============================================================

def load_student_files():

    students = []

    for path in sorted(
        DATA_DIR.glob("student_*.json")
    ):

        data = load_json(path)

        if (
            "student" not in data
            or "answers" not in data
        ):

            raise ValueError(
                f"Invalid student file: {path}"
            )

        students.append(data)

    if not students:

        raise RuntimeError(
            "No student JSON files found."
        )

    return students


# ============================================================
# LOAD BENCHMARK
# ============================================================

def load_benchmark(path):

    benchmark = {}

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        reader = csv.DictReader(f)

        required = {
            "student",
            "question",
            "score_0_10",
        }

        if not required.issubset(
            reader.fieldnames
        ):

            raise ValueError(
                "Benchmark must contain: "
                "student, question, score_0_10"
            )

        for row in reader:

            benchmark[
                (
                    row["student"],
                    row["question"]
                )
            ] = float(
                row["score_0_10"]
            )

    return benchmark


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    y_pred
):

    y_true = np.asarray(
        y_true,
        dtype=float
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float
    )

    metrics = {
        "n": int(len(y_true)),
        "mae": float(
            mean_absolute_error(
                y_true,
                y_pred
            )
        ),
        "rmse": float(
            math.sqrt(
                mean_squared_error(
                    y_true,
                    y_pred
                )
            )
        ),
    }

    if (
        len(y_true) >= 2
        and np.std(y_true) > 0
        and np.std(y_pred) > 0
    ):

        metrics["pearson_r"] = float(
            pearsonr(
                y_true,
                y_pred
            ).statistic
        )

    else:

        metrics["pearson_r"] = None

    if (
        len(y_true) >= 2
        and np.std(y_true) > 0
    ):

        metrics["spearman_rho"] = float(
            spearmanr(
                y_true,
                y_pred
            ).statistic
        )

    else:

        metrics["spearman_rho"] = None

    return metrics


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("PHASE 2 - STAGE 1")
    print("Sentence Transformer Semantic Baseline")
    print("=" * 70)

    print()
    print(f"Model : {MODEL_NAME}")
    print(f"Data  : {DATA_DIR}")
    print(f"Output: {RESULTS_DIR}")
    print()

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    model_data = load_json(
        MODEL_ANSWERS_FILE
    )

    students = load_student_files()

    benchmark = load_benchmark(
        BENCHMARK_FILE
    )

    print(
        f"Students loaded: {len(students)}"
    )

    print(
        f"Benchmark rows : {len(benchmark)}"
    )

    # --------------------------------------------------------
    # Construct records
    # --------------------------------------------------------

    records = []

    for student_data in students:

        student_id = student_data[
            "student"
        ]

        for (
            question_id,
            student_answer
        ) in student_data[
            "answers"
        ].items():

            if question_id not in model_data:

                print(
                    f"WARNING: no model answer "
                    f"for {question_id}"
                )

                continue

            key = (
                student_id,
                question_id
            )

            if key not in benchmark:

                print(
                    f"WARNING: no benchmark "
                    f"for {student_id}/{question_id}"
                )

                continue

            records.append(
                {
                    "student": student_id,
                    "question": question_id,
                    "student_answer": student_answer,
                    "model_answer": model_data[
                        question_id
                    ]["model_answer"],
                    "human_score": benchmark[
                        key
                    ],
                }
            )

    print(
        f"Evaluation records: {len(records)}"
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print()
    print(
        "Loading Sentence Transformer..."
    )

    model = SentenceTransformer(
        MODEL_NAME
    )

    print(
        "Model loaded."
    )

    # --------------------------------------------------------
    # Encode answers
    # --------------------------------------------------------

    student_answers = [
        r["student_answer"]
        for r in records
    ]

    model_answers = [
        r["model_answer"]
        for r in records
    ]

    print()
    print("Encoding student answers...")

    student_embeddings = model.encode(
        student_answers,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    print("Encoding model answers...")

    model_embeddings = model.encode(
        model_answers,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    # --------------------------------------------------------
    # Similarity + scores
    # --------------------------------------------------------

    results = []

    y_true = []
    y_pred = []

    for index, record in enumerate(
        records
    ):

        # Since embeddings are normalized,
        # dot product == cosine similarity.

        similarity = float(
            np.dot(
                student_embeddings[index],
                model_embeddings[index]
            )
        )

        predicted_score = (
            similarity * 10.0
        )

        predicted_score = max(
            0.0,
            min(
                10.0,
                predicted_score
            )
        )

        human_score = record[
            "human_score"
        ]

        signed_error = (
            predicted_score
            - human_score
        )

        results.append(
            {
                "student": record["student"],
                "question": record["question"],
                "human_score_0_10": human_score,
                "semantic_similarity": similarity,
                "predicted_score_0_10": predicted_score,
                "absolute_error": abs(
                    signed_error
                ),
                "signed_error": signed_error,
            }
        )

        y_true.append(
            human_score
        )

        y_pred.append(
            predicted_score
        )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    metrics = calculate_metrics(
        y_true,
        y_pred
    )

    # --------------------------------------------------------
    # Student summaries
    # --------------------------------------------------------

    grouped = {}

    for result in results:

        student = result[
            "student"
        ]

        if student not in grouped:

            grouped[student] = {
                "human": [],
                "predicted": [],
            }

        grouped[student][
            "human"
        ].append(
            result[
                "human_score_0_10"
            ]
        )

        grouped[student][
            "predicted"
        ].append(
            result[
                "predicted_score_0_10"
            ]
        )

    student_summary = {}

    for student, values in grouped.items():

        human_mean = float(
            np.mean(
                values["human"]
            )
        )

        predicted_mean = float(
            np.mean(
                values["predicted"]
            )
        )

        student_summary[student] = {
            "human_mean_score_0_10": human_mean,
            "semantic_mean_score_0_10": predicted_mean,
            "absolute_error": abs(
                predicted_mean
                - human_mean
            ),
        }

    # --------------------------------------------------------
    # JSON
    # --------------------------------------------------------

    output = {
        "experiment": (
            "phase2_stage1_semantic_baseline"
        ),
        "method": {
            "name": (
                "Sentence Transformers "
                "semantic similarity"
            ),
            "model": MODEL_NAME,
            "score_mapping": (
                "cosine_similarity * 10"
            ),
            "normalized_embeddings": True,
        },
        "dataset": {
            "students": len(students),
            "records_evaluated": len(records),
        },
        "metrics": metrics,
        "student_summary": student_summary,
        "question_results": results,
    }

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False
        )

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    fieldnames = [
        "student",
        "question",
        "human_score_0_10",
        "semantic_similarity",
        "predicted_score_0_10",
        "absolute_error",
        "signed_error",
    ]

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            results
        )

    # --------------------------------------------------------
    # Summary JSON
    # --------------------------------------------------------

    with open(
        OUTPUT_SUMMARY,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            {
                "experiment": (
                    "phase2_stage1_semantic_baseline"
                ),
                "metrics": metrics,
                "student_summary": student_summary,
            },
            f,
            indent=2
        )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("SEMANTIC BASELINE RESULTS")
    print("-" * 70)

    print(
        f"Records       : {metrics['n']}"
    )

    print(
        f"MAE           : "
        f"{metrics['mae']:.4f}"
    )

    print(
        f"RMSE          : "
        f"{metrics['rmse']:.4f}"
    )

    if metrics["pearson_r"] is not None:

        print(
            f"Pearson r     : "
            f"{metrics['pearson_r']:.4f}"
        )

    if metrics["spearman_rho"] is not None:

        print(
            f"Spearman rho  : "
            f"{metrics['spearman_rho']:.4f}"
        )

    print()
    print("Student-level results:")
    print()

    for student in sorted(
        student_summary
    ):

        s = student_summary[
            student
        ]

        print(
            f"{student:10s} "
            f"human="
            f"{s['human_mean_score_0_10']:.2f} "
            f"semantic="
            f"{s['semantic_mean_score_0_10']:.2f} "
            f"error="
            f"{s['absolute_error']:.2f}"
        )

    print()
    print("Files written:")
    print(OUTPUT_JSON)
    print(OUTPUT_CSV)
    print(OUTPUT_SUMMARY)
    print()
    print(
        "Semantic baseline complete."
    )


if __name__ == "__main__":
    main()
