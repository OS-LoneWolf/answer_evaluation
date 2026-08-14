import json


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


st = load(
    "sentence-transformers/results/evaluation_results.json"
)

iscc = load(
    "iscc-sct/results/evaluation_results.json"
)


print("=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(
    f"Student: {st['student']}"
)

print()

print(
    f"Sentence Transformers: "
    f"{st['total_marks']:.2f}/"
    f"{st['max_marks']:.2f} "
    f"({st['percentage']:.2f}%)"
)

print(
    f"ISCC-SCT: "
    f"{iscc['total_marks']:.2f}/"
    f"{iscc['max_marks']:.2f} "
    f"({iscc['percentage']:.2f}%)"
)

print()

print(
    "Difference: "
    f"{abs(st['total_marks'] - iscc['total_marks']):.2f} marks"
)
