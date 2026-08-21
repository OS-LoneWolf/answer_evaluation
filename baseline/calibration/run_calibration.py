import csv
import json
import math
from pathlib import Path

import joblib
import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, spearmanr


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path("/home/robot/Projects/intership")

INPUT_FILE = (
    PROJECT_ROOT
    / "student_evaluation"
    / "results"
    / "phase2_stage2"
    / "llm"
    / "llm_results.json"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "student_evaluation"
    / "results"
    / "phase2_stage3"
    / "calibration"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OOF_JSON = RESULTS_DIR / "calibration_oof_results.json"
OOF_CSV = RESULTS_DIR / "calibration_oof_scores.csv"
SUMMARY_JSON = RESULTS_DIR / "calibration_summary.json"
MODEL_FILE = RESULTS_DIR / "calibration_nn.joblib"


# ============================================================
# CONFIGURATION
# ============================================================

CRITERIA = [
    "correctness",
    "completeness",
    "relevance",
    "clarity",
]

SCORES = [str(i) for i in range(1, 11)]

RANDOM_STATE = 42

# Small network intentionally chosen because the dataset
# currently contains only 28 records.
HIDDEN_LAYERS = (16, 8)

# L2 regularization.
ALPHA = 0.01

MAX_ITER = 5000

# MLPRegressor with LBFGS is generally better suited to
# very small datasets than a large neural network trained
# with mini-batches.
SOLVER = "lbfgs"


# ============================================================
# HELPERS
# ============================================================

def load_results():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"\nStage 2 result file was not found:\n"
            f"{INPUT_FILE}\n\n"
            f"Run Stage 2 first, or copy the Stage 2 output "
            f"from the machine where Qwen evaluation was performed."
        )

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def validate_distribution(
    distribution,
    criterion,
    record_description,
):
    if not isinstance(distribution, dict):
        raise ValueError(
            f"{record_description}: "
            f"{criterion} distribution is not an object."
        )

    values = []

    for score in SCORES:

        if score not in distribution:
            raise ValueError(
                f"{record_description}: "
                f"{criterion} missing score {score}."
            )

        value = float(distribution[score])

        if not math.isfinite(value):
            raise ValueError(
                f"{record_description}: "
                f"{criterion} score {score} is not finite."
            )

        if value < 0:
            raise ValueError(
                f"{record_description}: "
                f"{criterion} score {score} is negative."
            )

        values.append(value)

    total = sum(values)

    if total <= 0:
        raise ValueError(
            f"{record_description}: "
            f"{criterion} distribution sums to zero."
        )

    # Normalize because Stage 2 may have small floating-point
    # deviations from exactly 1.0.
    values = [
        value / total
        for value in values
    ]

    return values


def distribution_features(result):
    """
    Convert the four 1..10 distributions into 40 features.

    Feature ordering:

    correctness_1 ... correctness_10
    completeness_1 ... completeness_10
    relevance_1 ... relevance_10
    clarity_1 ... clarity_10
    """

    distributions = result.get(
        "dimension_probability_distributions"
    )

    if distributions is None:
        raise ValueError(
            "Missing dimension_probability_distributions"
        )

    features = []

    description = (
        f"{result.get('student', '?')}/"
        f"{result.get('question', '?')}"
    )

    for criterion in CRITERIA:

        if criterion not in distributions:
            raise ValueError(
                f"{description}: missing {criterion}"
            )

        values = validate_distribution(
            distributions[criterion],
            criterion,
            description,
        )

        features.extend(values)

    return features


def expected_score(distribution):
    return sum(
        (index + 1) * probability
        for index, probability in enumerate(distribution)
    )


def raw_llm_score(result):
    """
    Recover the Stage 2 weighted score directly from the
    probability distributions.

    This is deliberately calculated independently so that
    Stage 3 has a clean baseline against which to compare
    calibration.
    """

    distributions = result[
        "dimension_probability_distributions"
    ]

    weights = {
        "correctness": 0.40,
        "completeness": 0.30,
        "relevance": 0.15,
        "clarity": 0.15,
    }

    score = 0.0

    for criterion in CRITERIA:

        values = validate_distribution(
            distributions[criterion],
            criterion,
            (
                f"{result.get('student', '?')}/"
                f"{result.get('question', '?')}"
            ),
        )

        score += (
            weights[criterion]
            * expected_score(values)
        )

    return score


def prepare_dataset(data):
    X = []
    y = []
    students = []
    records = []

    question_results = data.get(
        "question_results",
        []
    )

    if not question_results:
        raise ValueError(
            "No question_results found in Stage 2 output."
        )

    for result in question_results:

        if result.get("evaluation_status") != "success":
            continue

        student = result.get("student")
        question = result.get("question")

        if not student or not question:
            continue

        human_score = result.get(
            "human_benchmark_score_0_10"
        )

        if human_score is None:
            raise ValueError(
                f"Missing human score for "
                f"{student}/{question}"
            )

        features = distribution_features(result)

        X.append(features)
        y.append(float(human_score))
        students.append(student)

        records.append(
            {
                "student": student,
                "question": question,
                "human_score": float(human_score),
                "raw_llm_score":
                    raw_llm_score(result),
            }
        )

    if not X:
        raise ValueError(
            "No successful Stage 2 evaluations found."
        )

    return (
        np.asarray(X, dtype=np.float64),
        np.asarray(y, dtype=np.float64),
        students,
        records,
    )


# ============================================================
# MODEL
# ============================================================

def build_model():
    """
    Calibration network.

    Input:
        40 probability features

    Network:
        40 -> 16 -> 8 -> 1

    Target is normalized to 0..1 before training.
    """

    return Pipeline(
        [
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "calibration_nn",
                MLPRegressor(
                    hidden_layer_sizes=HIDDEN_LAYERS,
                    activation="relu",
                    solver=SOLVER,
                    alpha=ALPHA,
                    max_iter=MAX_ITER,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y_true, y_pred):

    y_true = np.asarray(
        y_true,
        dtype=float,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float,
    )

    metrics = {
        "n": int(len(y_true)),
        "mae": float(
            mean_absolute_error(
                y_true,
                y_pred,
            )
        ),
        "rmse": float(
            math.sqrt(
                mean_squared_error(
                    y_true,
                    y_pred,
                )
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
        metrics["pearson_r"] = float(
            pearsonr(
                y_true,
                y_pred,
            ).statistic
        )

    if (
        len(y_true) >= 2
        and np.std(y_true) > 0
    ):
        metrics["spearman_rho"] = float(
            spearmanr(
                y_true,
                y_pred,
            ).statistic
        )

    return metrics


# ============================================================
# LEAVE-ONE-STUDENT-OUT
# ============================================================

def leave_one_student_out(
    X,
    y,
    students,
    records,
):
    """
    Every student's four answers are held out together.

    This prevents answers from the same student appearing in
    both training and test sets in the same fold.
    """

    unique_students = sorted(
        set(students)
    )

    oof_predictions = np.full(
        len(y),
        np.nan,
        dtype=float,
    )

    fold_results = []

    print()
    print("=" * 70)
    print("LEAVE-ONE-STUDENT-OUT CALIBRATION")
    print("=" * 70)

    print(
        f"Students : {len(unique_students)}"
    )

    print(
        f"Records  : {len(y)}"
    )

    print()

    for fold_index, held_out_student in enumerate(
        unique_students,
        start=1,
    ):

        test_mask = np.asarray(
            [
                student == held_out_student
                for student in students
            ]
        )

        train_mask = ~test_mask

        X_train = X[train_mask]
        y_train = y[train_mask]

        X_test = X[test_mask]
        y_test = y[test_mask]

        print(
            f"[Fold {fold_index}/"
            f"{len(unique_students)}] "
            f"held out: {held_out_student}"
        )

        print(
            f"  Train records: {len(X_train)}"
        )

        print(
            f"  Test records : {len(X_test)}"
        )

        # Normalize target to 0..1.
        y_train_normalized = (
            y_train / 10.0
        )

        model = build_model()

        model.fit(
            X_train,
            y_train_normalized,
        )

        predictions = model.predict(
            X_test
        )

        predictions = (
            predictions * 10.0
        )

        # Scores must stay inside the grading range.
        predictions = np.clip(
            predictions,
            0.0,
            10.0,
        )

        test_indices = np.where(
            test_mask
        )[0]

        for local_index, global_index in enumerate(
            test_indices
        ):
            oof_predictions[
                global_index
            ] = predictions[local_index]

        fold_metrics = calculate_metrics(
            y_test,
            predictions,
        )

        fold_result = {
            "fold": fold_index,
            "held_out_student":
                held_out_student,
            "train_records":
                int(len(X_train)),
            "test_records":
                int(len(X_test)),
            "metrics":
                fold_metrics,
        }

        fold_results.append(
            fold_result
        )

        print(
            f"  MAE  : "
            f"{fold_metrics['mae']:.4f}"
        )

        print(
            f"  RMSE : "
            f"{fold_metrics['rmse']:.4f}"
        )

        print()

    if np.isnan(oof_predictions).any():
        raise RuntimeError(
            "Some records did not receive "
            "an out-of-fold prediction."
        )

    return (
        oof_predictions,
        fold_results,
    )


# ============================================================
# SAVE OOF RESULTS
# ============================================================

def save_oof_results(
    records,
    oof_predictions,
):
    output = []

    for index, record in enumerate(records):

        calibrated = float(
            oof_predictions[index]
        )

        raw = float(
            record["raw_llm_score"]
        )

        human = float(
            record["human_score"]
        )

        output.append(
            {
                "student":
                    record["student"],

                "question":
                    record["question"],

                "human_score_0_10":
                    human,

                "raw_llm_score_0_10":
                    raw,

                "calibrated_score_0_10":
                    calibrated,

                "raw_absolute_error":
                    abs(raw - human),

                "calibrated_absolute_error":
                    abs(calibrated - human),

                "calibrated_signed_error":
                    calibrated - human,
            }
        )

    with open(
        OOF_JSON,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
        )

    with open(
        OOF_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        fieldnames = [
            "student",
            "question",
            "human_score_0_10",
            "raw_llm_score_0_10",
            "calibrated_score_0_10",
            "raw_absolute_error",
            "calibrated_absolute_error",
            "calibrated_signed_error",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(output)

    return output


# ============================================================
# STUDENT-LEVEL SUMMARY
# ============================================================

def student_summary(oof_results):

    grouped = {}

    for row in oof_results:

        student = row["student"]

        grouped.setdefault(
            student,
            {
                "human": [],
                "raw": [],
                "calibrated": [],
            },
        )

        grouped[student]["human"].append(
            row["human_score_0_10"]
        )

        grouped[student]["raw"].append(
            row["raw_llm_score_0_10"]
        )

        grouped[student]["calibrated"].append(
            row["calibrated_score_0_10"]
        )

    output = {}

    for student, values in grouped.items():

        human = float(
            np.mean(values["human"])
        )

        raw = float(
            np.mean(values["raw"])
        )

        calibrated = float(
            np.mean(values["calibrated"])
        )

        output[student] = {
            "human_mean":
                human,

            "raw_llm_mean":
                raw,

            "calibrated_mean":
                calibrated,

            "raw_absolute_error":
                abs(raw - human),

            "calibrated_absolute_error":
                abs(calibrated - human),
        }

    return output


# ============================================================
# FINAL MODEL
# ============================================================

def train_final_model(X, y):

    print("=" * 70)
    print("TRAINING FINAL CALIBRATION MODEL")
    print("=" * 70)

    print(
        f"Input features : {X.shape[1]}"
    )

    print(
        f"Training rows  : {X.shape[0]}"
    )

    print(
        f"Architecture   : "
        f"{X.shape[1]} -> "
        f"{HIDDEN_LAYERS[0]} -> "
        f"{HIDDEN_LAYERS[1]} -> 1"
    )

    model = build_model()

    model.fit(
        X,
        y / 10.0,
    )

    joblib.dump(
        model,
        MODEL_FILE,
    )

    print()
    print(
        f"Saved model: {MODEL_FILE}"
    )

    return model


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("PHASE 2 - STAGE 3")
    print("LLM-Rubric-Style Calibration Neural Network")
    print("=" * 70)

    print()
    print(
        f"Input : {INPUT_FILE}"
    )

    print(
        f"Output: {RESULTS_DIR}"
    )

    print()

    # --------------------------------------------------------
    # Load Stage 2
    # --------------------------------------------------------

    data = load_results()

    X, y, students, records = (
        prepare_dataset(data)
    )

    print(
        f"Successful Stage 2 records : {len(y)}"
    )

    print(
        f"Input feature dimensions   : {X.shape[1]}"
    )

    print(
        f"Unique students             : "
        f"{len(set(students))}"
    )

    print()

    # --------------------------------------------------------
    # Stage 2 raw baseline
    # --------------------------------------------------------

    raw_scores = np.asarray(
        [
            record["raw_llm_score"]
            for record in records
        ],
        dtype=float,
    )

    raw_metrics = calculate_metrics(
        y,
        raw_scores,
    )

    print("=" * 70)
    print("STAGE 2 RAW LLM BASELINE")
    print("=" * 70)

    print(
        f"MAE          : "
        f"{raw_metrics['mae']:.4f}"
    )

    print(
        f"RMSE         : "
        f"{raw_metrics['rmse']:.4f}"
    )

    if raw_metrics["pearson_r"] is not None:
        print(
            f"Pearson r    : "
            f"{raw_metrics['pearson_r']:.4f}"
        )

    if raw_metrics["spearman_rho"] is not None:
        print(
            f"Spearman rho : "
            f"{raw_metrics['spearman_rho']:.4f}"
        )

    print()

    # --------------------------------------------------------
    # LOSO calibration
    # --------------------------------------------------------

    (
        oof_predictions,
        fold_results,
    ) = leave_one_student_out(
        X,
        y,
        students,
        records,
    )

    calibrated_metrics = calculate_metrics(
        y,
        oof_predictions,
    )

    print("=" * 70)
    print("CALIBRATED OOF RESULTS")
    print("=" * 70)

    print(
        f"MAE          : "
        f"{calibrated_metrics['mae']:.4f}"
    )

    print(
        f"RMSE         : "
        f"{calibrated_metrics['rmse']:.4f}"
    )

    if calibrated_metrics["pearson_r"] is not None:
        print(
            f"Pearson r    : "
            f"{calibrated_metrics['pearson_r']:.4f}"
        )

    if calibrated_metrics["spearman_rho"] is not None:
        print(
            f"Spearman rho : "
            f"{calibrated_metrics['spearman_rho']:.4f}"
        )

    print()

    # --------------------------------------------------------
    # Save OOF predictions
    # --------------------------------------------------------

    oof_results = save_oof_results(
        records,
        oof_predictions,
    )

    # --------------------------------------------------------
    # Student summary
    # --------------------------------------------------------

    students_result = student_summary(
        oof_results
    )

    print("Student-level OOF results:")
    print()

    for student in sorted(
        students_result
    ):

        result = students_result[student]

        print(
            f"{student:10s} "
            f"human="
            f"{result['human_mean']:.2f} "
            f"raw="
            f"{result['raw_llm_mean']:.2f} "
            f"calibrated="
            f"{result['calibrated_mean']:.2f}"
        )

    print()

    # --------------------------------------------------------
    # Final model trained on all records
    # --------------------------------------------------------

    final_model = train_final_model(
        X,
        y,
    )

    # Avoid unused variable warnings.
    _ = final_model

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    improvement_mae = (
        raw_metrics["mae"]
        - calibrated_metrics["mae"]
    )

    improvement_rmse = (
        raw_metrics["rmse"]
        - calibrated_metrics["rmse"]
    )

    summary = {
        "experiment":
            "phase2_stage3_llm_rubric_style_calibration",

        "input_file":
            str(INPUT_FILE),

        "dataset": {
            "records":
                int(len(y)),

            "students":
                int(len(set(students))),

            "features":
                int(X.shape[1]),
        },

        "feature_definition": {
            "correctness":
                "10 probabilities for scores 1..10",

            "completeness":
                "10 probabilities for scores 1..10",

            "relevance":
                "10 probabilities for scores 1..10",

            "clarity":
                "10 probabilities for scores 1..10",

            "total_features":
                40,
        },

        "calibration_model": {
            "type":
                "MLPRegressor",

            "architecture":
                [
                    40,
                    HIDDEN_LAYERS[0],
                    HIDDEN_LAYERS[1],
                    1,
                ],

            "activation":
                "relu",

            "solver":
                SOLVER,

            "alpha":
                ALPHA,

            "target_scaling":
                "human_score / 10",
        },

        "evaluation_protocol": {
            "method":
                "leave-one-student-out",

            "reason":
                "Prevents answers from the same student "
                "appearing in both training and test folds.",
        },

        "raw_llm_metrics":
            raw_metrics,

        "calibrated_oof_metrics":
            calibrated_metrics,

        "improvement": {
            "mae_reduction":
                improvement_mae,

            "rmse_reduction":
                improvement_rmse,

            "calibration_improved_mae":
                calibrated_metrics["mae"]
                < raw_metrics["mae"],

            "calibration_improved_rmse":
                calibrated_metrics["rmse"]
                < raw_metrics["rmse"],
        },

        "folds":
            fold_results,

        "student_summary":
            students_result,

        "artifacts": {
            "oof_json":
                str(OOF_JSON),

            "oof_csv":
                str(OOF_CSV),

            "summary":
                str(SUMMARY_JSON),

            "model":
                str(MODEL_FILE),
        },
    }

    with open(
        SUMMARY_JSON,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    print()
    print("=" * 70)
    print("STAGE 3 COMPLETE")
    print("=" * 70)

    print()
    print("Artifacts:")
    print(OOF_JSON)
    print(OOF_CSV)
    print(SUMMARY_JSON)
    print(MODEL_FILE)


if __name__ == "__main__":
    main()
