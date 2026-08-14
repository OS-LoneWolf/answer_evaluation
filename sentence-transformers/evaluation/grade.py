import json
from pathlib import Path

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "tests"

STUDENT_FILE = DATA_DIR / "student1.json"
MODEL_FILE = DATA_DIR / "modelanswer.json"
RUBRIC_FILE = DATA_DIR / "rubric.json"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Initial baseline threshold.
# This is NOT scientifically calibrated yet.
THRESHOLD = 0.50


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
# Load model
# ---------------------------------------------------------

print(f"Loading model: {MODEL_NAME}")

model = SentenceTransformer(MODEL_NAME)


# ---------------------------------------------------------
# Similarity
# ---------------------------------------------------------

def semantic_similarity(text1, text2):
    embeddings = model.encode(
        [text1, text2],
        convert_to_tensor=True,
        normalize_embeddings=True,
    )

    similarity = cos_sim(
        embeddings[0],
        embeddings[1]
    ).item()

    return float(similarity)


# ---------------------------------------------------------
# Score semantic criterion
# ---------------------------------------------------------

def score_semantic_criterion(student_answer, criterion):
    similarity = semantic_similarity(
        student_answer,
        criterion["description"]
    )

    marks = criterion["marks"]

    if similarity < THRESHOLD:
        awarded = 0.0
    else:
        # Initial baseline:
        # proportional scoring after threshold.
        awarded = marks * similarity

        # Never exceed maximum marks.
        awarded = min(awarded, marks)

    return similarity, awarded


# ---------------------------------------------------------
# Score minimum_count criterion
# ---------------------------------------------------------

def score_minimum_count(student_answer, criterion):

    required_count = criterion["required_count"]
    options = criterion["options"]
    max_marks = criterion["marks"]

    matched_options = []

    for option in options:

        similarity = semantic_similarity(
            option,
            student_answer
        )

        if similarity >= THRESHOLD:
            matched_options.append({
                "option": option,
                "similarity": similarity
            })

    matched_count = len(matched_options)

    effective_count = min(
        matched_count,
        required_count
    )

    awarded = (
        effective_count / required_count
    ) * max_marks

    return {
        "matched_count": matched_count,
        "required_count": required_count,
        "matched_options": matched_options,
        "awarded": awarded,
    }


# ---------------------------------------------------------
# Grade one question
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

    print("\nCriteria:")

    total_awarded = 0.0

    criterion_results = []

    for criterion in rubric["criteria"]:

        criterion_id = criterion["id"]
        criterion_type = criterion["type"]
        max_marks = criterion["marks"]

        if criterion_type == "semantic":

            similarity, awarded = score_semantic_criterion(
                student_answer,
                criterion
            )

            result = {
                "criterion_id": criterion_id,
                "type": criterion_type,
                "similarity": similarity,
                "max_marks": max_marks,
                "awarded_marks": awarded,
            }

            print(
                f"\n{criterion_id}"
                f"\nSimilarity: {similarity:.4f}"
                f"\nMarks: {awarded:.2f}/{max_marks}"
            )

        elif criterion_type == "minimum_count":

            result_data = score_minimum_count(
                student_answer,
                criterion
            )

            awarded = result_data["awarded"]

            result = {
                "criterion_id": criterion_id,
                "type": criterion_type,
                "max_marks": max_marks,
                "awarded_marks": awarded,
                **result_data,
            }

            print(
                f"\n{criterion_id}"
                f"\nMatched: "
                f"{result_data['matched_count']}/"
                f"{result_data['required_count']}"
                f"\nMarks: {awarded:.2f}/{max_marks}"
            )

            if result_data["matched_options"]:
                print("Matched concepts:")

                for item in result_data["matched_options"]:
                    print(
                        f"  - {item['option']} "
                        f"({item['similarity']:.4f})"
                    )

        else:

            raise ValueError(
                f"Unknown criterion type: {criterion_type}"
            )

        total_awarded += awarded
        criterion_results.append(result)

    max_marks = rubric["max_marks"]

    print("\n" + "-" * 70)
    print(
        f"Q{question_id[1:]} SCORE: "
        f"{total_awarded:.2f}/{max_marks}"
    )
    print("-" * 70)

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
            print(
                f"WARNING: {question_id} missing from model answers"
            )
            continue

        if question_id not in rubric_data:
            print(
                f"WARNING: {question_id} missing from rubric"
            )
            continue

        result = grade_question(question_id)

        results.append(result)

        total_marks += result["awarded_marks"]
        total_max_marks += result["max_marks"]

    percentage = (
        total_marks / total_max_marks * 100
        if total_max_marks
        else 0
    )

    print("\n")
    print("=" * 70)
    print("FINAL RESULT")
    print("=" * 70)

    print(f"Student: {student_data['student']}")
    print(f"Marks: {total_marks:.2f}/{total_max_marks:.2f}")
    print(f"Percentage: {percentage:.2f}%")

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

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(f"\nDetailed results saved to: {output_file}")


if __name__ == "__main__":
    main()
