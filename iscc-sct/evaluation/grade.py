import json
from pathlib import Path

import iscc_sct as sct


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "tests"

STUDENT_FILE = DATA_DIR / "student1.json"
MODEL_FILE = DATA_DIR / "modelanswer.json"
RUBRIC_FILE = DATA_DIR / "rubric.json"

BITS = 64

# Initial experimental threshold.
# This must be calibrated later using human-graded data.
SIMILARITY_THRESHOLD = 0.50


# ---------------------------------------------------------
# Load JSON
# ---------------------------------------------------------

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


student_data = load_json(STUDENT_FILE)
model_data = load_json(MODEL_FILE)
rubric_data = load_json(RUBRIC_FILE)


# ---------------------------------------------------------
# ISCC semantic similarity
# ---------------------------------------------------------

def semantic_similarity(text1, text2):

    code1 = sct.create(
        text1,
        bits=BITS
    ).iscc

    code2 = sct.create(
        text2,
        bits=BITS
    ).iscc

    distance = sct.iscc_distance(
        code1,
        code2
    )

    similarity = 1.0 - (
        distance / BITS
    )

    return {
        "code1": code1,
        "code2": code2,
        "distance": distance,
        "similarity": similarity,
    }


# ---------------------------------------------------------
# Semantic criterion
# ---------------------------------------------------------

def score_semantic_criterion(student_answer, criterion):

    result = semantic_similarity(
        student_answer,
        criterion["description"]
    )

    similarity = result["similarity"]

    max_marks = criterion["marks"]

    if similarity < SIMILARITY_THRESHOLD:
        awarded = 0.0
    else:
        awarded = min(
            similarity * max_marks,
            max_marks
        )

    result["awarded_marks"] = awarded
    result["max_marks"] = max_marks

    return result


# ---------------------------------------------------------
# Minimum count criterion
# ---------------------------------------------------------

def score_minimum_count(student_answer, criterion):

    required_count = criterion["required_count"]

    options = criterion["options"]

    max_marks = criterion["marks"]

    matched_options = []

    for option in options:

        result = semantic_similarity(
            option,
            student_answer
        )

        if result["similarity"] >= SIMILARITY_THRESHOLD:

            matched_options.append({
                "option": option,
                "similarity": result["similarity"],
                "distance": result["distance"],
            })

    matched_count = len(matched_options)

    effective_count = min(
        matched_count,
        required_count
    )

    awarded = (
        effective_count /
        required_count
    ) * max_marks

    return {
        "matched_count": matched_count,
        "required_count": required_count,
        "matched_options": matched_options,
        "awarded_marks": awarded,
        "max_marks": max_marks,
    }


# ---------------------------------------------------------
# Grade question
# ---------------------------------------------------------

def grade_question(question_id):

    student_answer = student_data["answers"][question_id]

    model_answer = model_data[question_id]["model_answer"]

    rubric = rubric_data[question_id]

    print("\n" + "=" * 70)
    print(f"QUESTION: {question_id}")
    print("=" * 70)

    print("\nStudent Answer:")
    print(student_answer)

    print("\nModel Answer:")
    print(model_answer)

    total_awarded = 0.0

    criterion_results = []

    for criterion in rubric["criteria"]:

        criterion_id = criterion["id"]

        criterion_type = criterion["type"]

        print("\n" + "-" * 70)
        print(criterion_id)

        if criterion_type == "semantic":

            result = score_semantic_criterion(
                student_answer,
                criterion
            )

            print(
                f"ISCC Hamming distance: "
                f"{result['distance']}"
            )

            print(
                f"Semantic similarity: "
                f"{result['similarity']:.4f}"
            )

            print(
                f"Marks: "
                f"{result['awarded_marks']:.2f}/"
                f"{result['max_marks']}"
            )

        elif criterion_type == "minimum_count":

            result = score_minimum_count(
                student_answer,
                criterion
            )

            print(
                f"Matched: "
                f"{result['matched_count']}/"
                f"{result['required_count']}"
            )

            print(
                f"Marks: "
                f"{result['awarded_marks']:.2f}/"
                f"{result['max_marks']}"
            )

            for option in result["matched_options"]:

                print(
                    f"  {option['option']}: "
                    f"similarity="
                    f"{option['similarity']:.4f}, "
                    f"distance="
                    f"{option['distance']}"
                )

        else:

            raise ValueError(
                f"Unknown criterion type: {criterion_type}"
            )

        total_awarded += result["awarded_marks"]

        criterion_results.append({
            "criterion_id": criterion_id,
            "type": criterion_type,
            **result,
        })

    max_marks = rubric["max_marks"]

    print("\n" + "-" * 70)

    print(
        f"{question_id} SCORE: "
        f"{total_awarded:.2f}/{max_marks}"
    )

    return {
        "question_id": question_id,
        "max_marks": max_marks,
        "awarded_marks": total_awarded,
        "criteria": criterion_results,
    }


# ---------------------------------------------------------
# Grade complete student
# ---------------------------------------------------------

def main():

    results = []

    total_marks = 0.0

    total_max_marks = 0.0

    for question_id in student_data["answers"]:

        if question_id not in model_data:
            continue

        if question_id not in rubric_data:
            continue

        result = grade_question(
            question_id
        )

        results.append(result)

        total_marks += result[
            "awarded_marks"
        ]

        total_max_marks += result[
            "max_marks"
        ]

    percentage = (
        total_marks /
        total_max_marks *
        100
    )

    print("\n")
    print("=" * 70)
    print("FINAL RESULT")
    print("=" * 70)

    print(
        f"Student: "
        f"{student_data['student']}"
    )

    print(
        f"Marks: "
        f"{total_marks:.2f}/"
        f"{total_max_marks:.2f}"
    )

    print(
        f"Percentage: "
        f"{percentage:.2f}%"
    )

    output = {
        "student": student_data["student"],
        "total_marks": total_marks,
        "max_marks": total_max_marks,
        "percentage": percentage,
        "questions": results,
    }

    RESULTS_DIR = BASE_DIR / "results"
    RESULTS_DIR.mkdir(exist_ok=True)

    output_file = RESULTS_DIR / "evaluation_results.json"

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(
        f"\nDetailed results saved to:"
        f" {output_file}"
    )


if __name__ == "__main__":
    main()
