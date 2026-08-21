import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity


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
    / "tfidf"
)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


MODEL_ANSWERS_FILE = DATA_DIR / "dsa_model_answers.json"
BENCHMARK_FILE = DATA_DIR / "provisional_human_benchmark.csv"


OUTPUT_JSON = RESULTS_DIR / "tfidf_results.json"
OUTPUT_CSV = RESULTS_DIR / "tfidf_question_scores.csv"
OUTPUT_SUMMARY = RESULTS_DIR / "tfidf_summary.json"


# ============================================================
# LOAD JSON
# ============================================================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# LOAD STUDENTS
# ============================================================

def load_student_files():

    students = []

    for path in sorted(DATA_DIR.glob("student_*.json")):

        data = load_json(path)

        if "student" not in data or "answers" not in data:
            raise ValueError(
                f"Invalid student file format: {path}"
            )

        students.append(data)

    if not students:
        raise RuntimeError(
            f"No student_*.json files found in {DATA_DIR}"
        )

    return students


# ============================================================
# LOAD BENCHMARK
# ============================================================

def load_benchmark(path):

    benchmark = {}

    with open(path, "r", encoding="utf-8") as f:

        reader = csv.DictReader(f)

        required_columns = {
            "student",
            "question",
            "score_0_10",
        }

        if not required_columns.issubset(reader.fieldnames):
            raise ValueError(
                "Benchmark CSV must contain: "
                "student, question, score_0_10"
            )

        for row in reader:

            key = (
                row["student"],
                row["question"],
            )

            benchmark[key] = float(
                row["score_0_10"]
            )

    return benchmark


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y_true, y_pred):

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

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
    print("TF-IDF + Cosine Similarity Baseline")
    print("=" * 70)

    print()
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Data         : {DATA_DIR}")
    print(f"Results      : {RESULTS_DIR}")
    print()

    # --------------------------------------------------------
    # Load model answers
    # --------------------------------------------------------

    model_data = load_json(
        MODEL_ANSWERS_FILE
    )

    # --------------------------------------------------------
    # Load students
    # --------------------------------------------------------

    students = load_student_files()

    print(
        f"Students loaded: {len(students)}"
    )

    # --------------------------------------------------------
    # Load benchmark
    # --------------------------------------------------------

    benchmark = load_benchmark(
        BENCHMARK_FILE
    )

    print(
        f"Benchmark rows: {len(benchmark)}"
    )

    # --------------------------------------------------------
    # Construct evaluation records
    # --------------------------------------------------------

    records = []

    for student_data in students:

        student_id = student_data["student"]

        for question_id, student_answer in (
            student_data["answers"].items()
        ):

            if question_id not in model_data:

                print(
                    f"WARNING: no model answer "
                    f"for {question_id}"
                )

                continue

            benchmark_key = (
                student_id,
                question_id,
            )

            if benchmark_key not in benchmark:

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
                        benchmark_key
                    ],
                }
            )

    print(
        f"Evaluation records: {len(records)}"
    )

    # --------------------------------------------------------
    # TF-IDF
    # --------------------------------------------------------

    corpus = []

    for record in records:

        corpus.append(
            record["model_answer"]
        )

        corpus.append(
            record["student_answer"]
        )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        sublinear_tf=True,
    )

    tfidf_matrix = vectorizer.fit_transform(
        corpus
    )

    print(
        f"TF-IDF vocabulary size: "
        f"{len(vectorizer.vocabulary_)}"
    )

    # --------------------------------------------------------
    # Calculate similarity
    # --------------------------------------------------------

    y_true = []
    y_pred = []

    results = []

    for index, record in enumerate(records):

        model_vector = tfidf_matrix[
            index * 2
        ]

        student_vector = tfidf_matrix[
            index * 2 + 1
        ]

        similarity = float(
            cosine_similarity(
                model_vector,
                student_vector
            )[0][0]
        )

        # Baseline mapping:
        #
        # similarity 0.0 -> score 0
        # similarity 1.0 -> score 10
        #
        predicted_score = (
            similarity * 10.0
        )

        predicted_score = max(
            0.0,
            min(10.0, predicted_score)
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
                "tfidf_cosine_similarity": similarity,
                "predicted_score_0_10": predicted_score,
                "absolute_error": abs(
                    signed_error
                ),
                "signed_error": signed_error,
            }
        )

        y_true.append(human_score)
        y_pred.append(predicted_score)

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    metrics = calculate_metrics(
        y_true,
        y_pred
    )

    # --------------------------------------------------------
    # Per-student summary
    # --------------------------------------------------------

    student_summary = {}

    for result in results:

        student = result["student"]

        if student not in student_summary:

            student_summary[student] = {
                "human_scores": [],
                "predicted_scores": [],
            }

        student_summary[student][
            "human_scores"
        ].append(
            result["human_score_0_10"]
        )

        student_summary[student][
            "predicted_scores"
        ].append(
            result["predicted_score_0_10"]
        )

    for student in student_summary:

        human_scores = student_summary[
            student
        ]["human_scores"]

        predicted_scores = student_summary[
            student
        ]["predicted_scores"]

        human_mean = float(
            np.mean(human_scores)
        )

        predicted_mean = float(
            np.mean(predicted_scores)
        )

        student_summary[student] = {
            "human_mean_score_0_10": human_mean,
            "tfidf_mean_score_0_10": predicted_mean,
            "absolute_error": abs(
                predicted_mean
                - human_mean
            ),
        }

    # --------------------------------------------------------
    # JSON output
    # --------------------------------------------------------

    output = {
        "experiment": (
            "phase2_stage1_tfidf_baseline"
        ),
        "method": {
            "name": "TF-IDF + cosine similarity",
            "ngram_range": [1, 2],
            "sublinear_tf": True,
            "score_mapping": (
                "cosine_similarity * 10"
            ),
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
            ensure_ascii=False,
        )

    # --------------------------------------------------------
    # CSV output
    # --------------------------------------------------------

    fieldnames = [
        "student",
        "question",
        "human_score_0_10",
        "tfidf_cosine_similarity",
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
        writer.writerows(results)

    # --------------------------------------------------------
    # Summary output
    # --------------------------------------------------------

    with open(
        OUTPUT_SUMMARY,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            {
                "experiment": (
                    "phase2_stage1_tfidf_baseline"
                ),
                "metrics": metrics,
                "student_summary": student_summary,
            },
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Terminal output
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("TF-IDF RESULTS")
    print("-" * 70)

    print(
        f"Records       : {metrics['n']}"
    )

    print(
        f"MAE           : {metrics['mae']:.4f}"
    )

    print(
        f"RMSE          : {metrics['rmse']:.4f}"
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

        s = student_summary[student]

        print(
            f"{student:10s} "
            f"human={s['human_mean_score_0_10']:.2f} "
            f"tfidf={s['tfidf_mean_score_0_10']:.2f} "
            f"error={s['absolute_error']:.2f}"
        )

    print()
    print("Files written:")
    print(OUTPUT_JSON)
    print(OUTPUT_CSV)
    print(OUTPUT_SUMMARY)
    print()
    print("TF-IDF baseline complete.")


if __name__ == "__main__":
    main()
