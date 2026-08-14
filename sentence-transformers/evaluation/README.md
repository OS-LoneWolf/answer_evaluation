# Automated Student Answer Evaluation

## Objective

This module evaluates student answers against model answers
using Sentence Transformer semantic embeddings.

## Input

- student1.json
- modelanswer.json
- rubric.json

## Method

1. Load student answer.
2. Load reference answer.
3. Load rubric.
4. Generate Sentence Transformer embeddings.
5. Calculate cosine similarity.
6. Compare similarity against the rubric criterion.
7. Convert similarity into criterion marks.
8. Sum criterion marks.
9. Produce total score.

## Model

sentence-transformers/all-MiniLM-L6-v2

## Scoring

The current implementation uses a heuristic similarity
threshold and proportional scoring.

This is an experimental baseline and has not yet been
calibrated against human grades.

## Limitations

Semantic similarity does not guarantee factual correctness.
The system may assign high scores to answers containing
conceptual errors if the overall semantic content is similar
to the reference.

## Future Work

- Calibrate threshold using human-graded data.
- Perform sentence/claim-level comparison.
- Evaluate CrossEncoder models.
- Compare predictions against human grades.
- Add NLI-based verification.
- Compare with LLM-Rubric.
