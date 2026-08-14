# Automated Student Answer Evaluation using ISCC-SCT

## Objective

This module evaluates student answers using ISCC Semantic
Text-Codes.

## Method

1. Load student answer.
2. Load model answer and rubric.
3. Generate ISCC Semantic Text-Codes.
4. Calculate Hamming distance.
5. Normalize distance into an experimental similarity score.
6. Map similarity to rubric marks.
7. Produce the final score.

## Similarity

For a 64-bit code:

similarity = 1 - (Hamming distance / 64)

## Important

The score mapping is an experimental heuristic.

ISCC-SCT is a semantic similarity/fingerprinting method,
not a dedicated educational grading model.

## Limitations

Semantic similarity does not directly measure factual
correctness or rubric satisfaction.

## Future Work

- Calibrate similarity thresholds.
- Evaluate sentence-level comparison.
- Compare against human grading.
- Investigate criterion-level scoring.
- Compare against Sentence Transformers and NLI.
