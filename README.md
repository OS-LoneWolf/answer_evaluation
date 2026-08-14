# Automated Student Answer Evaluation

## 1. Project Overview

The objective of this project is to develop an automated system for evaluating
student-written answers against reference/model answers and assigning marks
based on the content of the student's response.

The long-term goal is:

    Student Answer
          |
          v
    Automated Evaluation
          |
          +----------------------+
          |                      |
          v                      v
    Criterion-level          Explanation
       marks                    |
          |                      |
          +----------+-----------+
                     |
                     v
              Final Marks
              / Percentage
              / Feedback

The system is being developed incrementally by evaluating different approaches
and comparing their results against human/expert grading.

The approaches currently being investigated are:

1. Sentence Transformers
2. ISCC-SCT
3. LLM-Rubric (planned next)

The first two approaches are being used as semantic-similarity baselines.
LLM-Rubric will be investigated as a more advanced rubric-based evaluation
approach.

---

# 2. Repositories Used

The following open-source repositories were cloned and evaluated.

## 2.1 Sentence Transformers

Official repository:

https://github.com/huggingface/sentence-transformers

Sentence Transformers provides pretrained transformer models for generating
dense sentence/text embeddings and performing semantic textual similarity,
semantic search, retrieval, reranking, and related tasks.

For this project, Sentence Transformers is used to convert:

    Student Answer
           |
           v
       Embedding
           |
           v
    Reference/Rubric
       Embedding
           |
           v
    Cosine Similarity

The framework supports both bi-encoder embedding models and Cross-Encoder
models. The current experiment uses a bi-encoder embedding model.

Reference:

https://github.com/huggingface/sentence-transformers

---

## 2.2 ISCC-SCT

Official repository:

https://github.com/iscc/iscc-sct

ISCC-SCT provides Semantic Text-Code functionality for generating compact
semantic representations of text and comparing them using semantic-code
distance.

The experiment uses:

    Student Answer
           |
           v
    ISCC Semantic Text-Code
           |
           v
    Hamming Distance
           |
           v
    Normalized Similarity
           |
           v
        Marks

ISCC-SCT is being investigated as a lightweight semantic similarity/fingerprinting
baseline.

Reference:

https://github.com/iscc/iscc-sct

---

## 2.3 LLM-Rubric

Official repository:

https://github.com/microsoft/LLM-Rubric

LLM-Rubric is a research framework developed by Microsoft for automated
evaluation of natural language texts.

Unlike a simple embedding similarity system, LLM-Rubric uses an LLM to
evaluate text according to multiple rubric questions and then uses a
calibration model to predict human judgments.

The original work was published at ACL 2024:

"LLM-Rubric: A Multidimensional, Calibrated Approach to Automated Evaluation
of Natural Language Texts"

Reference:

https://github.com/microsoft/LLM-Rubric

Paper:

https://aclanthology.org/2024.acl-long.745/

LLM-Rubric is planned as the next major evaluation approach in this project.

---

# 3. Project Directory Structure

The current project is organized approximately as follows:

    Projects/intership/
    |
    +-- README.md
    |
    +-- compare_results.py
    |
    +-- sentence-transformers/
    |   |
    |   +-- sentence_transformers/
    |   |       Original Sentence Transformers source code
    |   |
    |   +-- evaluation/
    |   |   +-- __init__.py
    |   |   +-- grade.py
    |   |   +-- README.md
    |   |
    |   +-- tests/
    |   |   +-- student1.json
    |   |   +-- modelanswer.json
    |   |   +-- rubric.json
    |   |
    |   +-- results/
    |       +-- evaluation_results.json
    |
    +-- iscc-sct/
        |
        +-- iscc_sct/
        |       Original ISCC-SCT source code
        |
        +-- evaluation/
        |   +-- __init__.py
        |   +-- grade.py
        |   +-- README.md
        |
        +-- tests/
        |   +-- student1.json
        |   +-- modelanswer.json
        |   +-- rubric.json
        |
        +-- results/
            +-- evaluation_results.json

The original source code of the two libraries is kept separate from the
experimental grading code.

This allows the evaluation pipeline to be developed without unnecessarily
modifying the upstream implementations.

---

# 4. Input Data

The current experiment uses three JSON files.

## 4.1 student1.json

Contains the student's answers.

Example:

    {
        "student": "student1",
        "answers": {
            "Q1": "...",
            "Q2": "...",
            "Q3": "...",
            "Q4": "..."
        }
    }

The student answer is the text that the evaluation system needs to grade.

---

## 4.2 modelanswer.json

Contains the reference/model answer for each question.

Example:

    {
        "Q1": {
            "question": "Explain inheritance and its types.",
            "model_answer": "Inheritance is the OOP mechanism..."
        }
    }

The model answer provides the expected conceptual content.

---

## 4.3 rubric.json

The rubric defines what should receive marks.

For example:

    {
        "Q1": {
            "max_marks": 5,
            "criteria": [
                {
                    "id": "q1_definition",
                    "description": "...",
                    "marks": 2,
                    "type": "semantic"
                }
            ]
        }
    }

The rubric is important because comparing the complete student answer against
the complete model answer is not sufficient for educational grading.

Different parts of an answer may deserve different marks.

---

# 5. Why a Rubric is Required

A simple similarity score answers:

    "How similar are these two pieces of text?"

However, educational grading requires answering:

    "Did the student satisfy each required learning criterion?"

For example, Q1 has:

    max_marks = 5

with:

    Definition = 2 marks
    Types = 3 marks

Therefore, a student could write an answer that is semantically similar to
the model answer but still miss an important required concept.

A rubric allows the grading system to evaluate individual criteria.

---

# 6. Sentence Transformers Baseline

## 6.1 Model Used

The current implementation uses:

    sentence-transformers/all-MiniLM-L6-v2

The model produces 384-dimensional embeddings.

The official Sentence Transformers documentation demonstrates the use of
`all-MiniLM-L6-v2` for generating embeddings and calculating semantic
similarity.

Reference:

https://github.com/huggingface/sentence-transformers

---

# 7. Sentence Transformer Evaluation Pipeline

The current implementation follows this process:

    1. Load student JSON
    2. Load model answer JSON
    3. Load rubric JSON
    4. Load Sentence Transformer model
    5. Convert text into embeddings
    6. Calculate cosine similarity
    7. Compare similarity with a threshold
    8. Convert similarity to marks
    9. Sum criterion marks
    10. Produce final marks
    11. Save detailed JSON results

Conceptually:

    Student Text
          |
          v
    Sentence Transformer
          |
          v
    Student Embedding
          |
          |
          +----------------------+
                                 |
                                 v
                         Cosine Similarity
                                 ^
                                 |
          +----------------------+
          |
          v
    Criterion Text
          |
          v
    Criterion Embedding

---

# 8. Sentence Transformer Numerical Parameters

Current model:

    sentence-transformers/all-MiniLM-L6-v2

Embedding dimension:

    384

Similarity metric:

    Cosine similarity

Current experimental threshold:

    0.50

The threshold is currently a heuristic value.

It has NOT been calibrated against a human-graded dataset.

Therefore, 0.50 should not be interpreted as a scientifically validated
grading threshold.

---

# 9. Sentence Transformer Scoring Formula

For a semantic criterion:

    similarity = cosine(student_embedding, criterion_embedding)

Current experimental scoring:

    if similarity < threshold:

        marks = 0

    otherwise:

        marks = maximum_marks * similarity

with:

    threshold = 0.50

and:

    marks <= maximum_marks

Therefore, for a criterion worth 2 marks:

    similarity = 0.80

would currently produce approximately:

    2 * 0.80 = 1.60 marks

This is an experimental scoring rule and will be replaced or calibrated
after human evaluation data is available.

---

# 10. Important Sentence Transformer Limitation

Semantic similarity does not necessarily mean correctness.

For example:

    Student:
    "Inheritance allows a class to receive properties from another class."

This may receive high semantic similarity to:

    "Inheritance allows a derived class to acquire properties and
     behaviours from a base class."

However, a student may also write something that is semantically related
but technically incorrect.

Therefore:

    High similarity != guaranteed factual correctness

This is one of the main reasons why a pure embedding-based grading system
may not be sufficient for the final system.

---

# 11. Important Q1 Rubric Limitation

Q1 currently contains a criterion similar to:

    {
        "type": "minimum_count",
        "required_count": 3
    }

The intended meaning is:

    The student must correctly name and briefly explain
    at least three types of inheritance.

The initial baseline implementation has an important limitation.

It checks whether inheritance-type concepts are semantically related to the
student answer, but this does not fully verify that the student:

    1. Named the type
    2. Correctly explained the type
    3. Explained it accurately
    4. Did not contain a contradictory statement

For example:

    Student:
    "Multiple inheritance means one parent class has many child classes."

The answer contains the phrase "multiple inheritance", but the explanation
is actually describing hierarchical inheritance.

A simple semantic similarity system may incorrectly give credit.

This is a significant limitation of the current baseline.

Future versions should perform criterion-level or claim-level evaluation
instead of comparing only against the entire answer.

---

# 12. ISCC-SCT Baseline

ISCC-SCT is being used as a second semantic similarity approach.

Repository:

https://github.com/iscc/iscc-sct

The pipeline is:

    Student Answer
          |
          v
    Semantic Text-Code
          |
          v
    Hamming Distance
          |
          v
    Normalized Similarity
          |
          v
        Marks

The advantage of this approach is that text can be represented using a
compact semantic code rather than storing a large embedding vector.

---

# 13. ISCC-SCT Numerical Parameters

Current implementation uses:

    Semantic code size = 64 bits

The comparison produces:

    Hamming distance

The current implementation converts Hamming distance into an experimental
similarity score:

    similarity = 1 - (distance / 64)

Examples:

    distance = 0
    similarity = 1.00

    distance = 16
    similarity = 0.75

    distance = 32
    similarity = 0.50

    distance = 64
    similarity = 0.00

Current experimental threshold:

    0.50

Again, this threshold has not yet been calibrated against human grading.

---

# 14. ISCC-SCT Scoring

For a semantic criterion:

    if similarity < 0.50:

        marks = 0

    otherwise:

        marks = maximum_marks * similarity

The score is then capped at the criterion's maximum marks.

For example, if a criterion is worth 2 marks:

    similarity = 0.80

would produce:

    2 * 0.80 = 1.60 marks

This is an experimental mapping created for this project.

It is NOT a grading formula provided by ISCC-SCT.

ISCC-SCT is being used as a semantic similarity mechanism, not as an
educational grading model.

---

# 15. ISCC-SCT Limitation

The main limitation is similar to the Sentence Transformer approach.

Semantic similarity does not directly measure:

    - factual correctness
    - completeness
    - required concepts
    - logical consistency
    - quality of explanation
    - misconceptions

For example, two answers can be semantically close while one contains an
important technical error.

Therefore ISCC-SCT should be considered a baseline rather than a final
automated grading solution.

---

# 16. Comparison of the Two Approaches

The two systems solve approximately the same initial problem using different
representations.

## Sentence Transformers

    Text
     |
     v
    Dense vector
     |
     v
    Cosine similarity

Advantages:

- Strong general-purpose semantic representation
- 384-dimensional embeddings with the selected MiniLM model
- Easy to compare many text pairs
- Large ecosystem of pretrained models
- Supports Cross-Encoder/reranker models
- Can later be fine-tuned for a specific grading task

Disadvantages:

- Similarity is not the same as correctness
- Requires threshold calibration
- Can miss factual contradictions
- Long answers may contain several concepts that are mixed together
- Current scoring formula is heuristic

---

## ISCC-SCT

    Text
     |
     v
    Semantic Text-Code
     |
     v
    Hamming distance

Advantages:

- Compact semantic representation
- Fast comparison
- Simple distance-based comparison
- Useful as a semantic/fingerprinting baseline

Disadvantages:

- Not specifically designed for educational grading
- Hamming distance alone does not represent rubric satisfaction
- Does not directly verify facts
- Does not directly understand whether an explanation is correct
- Current distance-to-marks conversion is heuristic
- Threshold needs calibration

---

# 17. Current Experimental Conclusion

The current experiments demonstrate that both approaches can detect semantic
relatedness between a student answer and a reference/criterion.

The Sentence Transformer experiment previously produced an example cosine
similarity of approximately:

    0.8423

for the complete student/model answer comparison.

This indicates strong semantic similarity between the two answers.

However, this number should NOT be interpreted as:

    "Student scored 84.23%."

It only indicates semantic similarity under that embedding model.

Similarly, ISCC-SCT produced semantic codes and a Hamming distance between
the compared texts.

The Hamming distance can be transformed into an experimental similarity
score, but this is not equivalent to educational grading accuracy.

Therefore, the current conclusion is:

    Both approaches are useful as semantic baselines,
    but neither should yet be treated as a reliable final
    automatic grading system.

---

# 18. Why the Two Models Are Still Useful

The purpose of these experiments is not simply to obtain the highest
similarity score.

The purpose is to establish baseline systems.

We need to answer:

    Can a simple semantic model automatically approximate
    human grading?

If yes:

    How accurate is it?

If no:

    What information is missing?

The two baselines allow us to establish this experimentally before moving
to a more sophisticated LLM-based approach.

---

# 19. Current Evaluation Metrics

At the current stage, we record:

    - cosine similarity
    - ISCC Hamming distance
    - normalized similarity
    - criterion marks
    - question marks
    - total marks
    - percentage

However, these are NOT yet sufficient to claim grading accuracy.

To calculate actual grading performance, we need human/expert scores.

For example:

    Human grade:             4.0 / 5
    Sentence Transformer:   3.8 / 5
    ISCC-SCT:                3.4 / 5

Once enough examples are available, we can calculate:

    - Mean Absolute Error (MAE)
    - Root Mean Squared Error (RMSE)
    - Pearson correlation
    - Spearman correlation
    - Kendall correlation
    - exact agreement
    - agreement within ±0.5 marks
    - agreement within ±1 mark

These metrics will allow a meaningful comparison between models.

---

# 20. Why Human-Graded Data is Necessary

The final objective is not:

    "Does the model answer look similar?"

The objective is:

    "Does the automatic grade agree with an expert grader?"

Therefore we need a dataset containing:

    Question
    Student Answer
    Model Answer
    Rubric
    Human/Expert Marks

For example:

    Q1
    Student answer
    Model answer
    Rubric
    Human score = 4/5

This becomes the ground truth for evaluating automatic grading.

---

# 21. Planned Improvement to Sentence Transformers

The current Sentence Transformer implementation is a baseline.

The next improvement will be criterion-level evaluation.

Instead of:

    entire student answer
              |
              v
    entire criterion

we should identify relevant student sentences/claims.

For example:

    Student Answer
          |
          v
    Sentence segmentation
          |
          +---- Sentence 1
          +---- Sentence 2
          +---- Sentence 3
          +---- Sentence 4
          |
          v
    Match relevant sentence to criterion
          |
          v
    Evaluate semantic similarity

This should reduce false positives caused by comparing an entire answer
against a small criterion.

---

# 22. Planned Cross-Encoder Experiment

Sentence Transformers also provides Cross-Encoder models.

A bi-encoder independently embeds two texts and compares their embeddings.

A Cross-Encoder processes both texts together.

Conceptually:

Bi-Encoder:

    Student Answer -> embedding
    Criterion     -> embedding

    embedding <-> embedding
          |
          v
      similarity

Cross-Encoder:

    Student Answer + Criterion
              |
              v
          Transformer
              |
              v
          score

Cross-Encoders can be more accurate for predefined text pairs but are
generally more computationally expensive than embedding-based comparison.

Therefore a Cross-Encoder is a useful next baseline before moving to a
full LLM-based grader.

Reference:

https://github.com/huggingface/sentence-transformers

---

# 23. Planned LLM-Rubric Experiment

The next major stage is LLM-Rubric.

Repository:

https://github.com/microsoft/LLM-Rubric

Paper:

https://aclanthology.org/2024.acl-long.745/

LLM-Rubric is different from the current approaches.

Instead of only calculating semantic similarity, the LLM is prompted with
individual evaluation questions/rubric dimensions.

Conceptually:

    Student Answer
          +
       Rubric
          |
          v
         LLM
          |
          v
    Criterion-level predictions
          |
          v
    Calibration model
          |
          v
    Predicted human grade

The original LLM-Rubric approach combines multiple LLM-derived rubric
predictions using a small feed-forward calibration network with
judge-specific and judge-independent parameters.

This makes it particularly relevant to our problem because educational
grading is naturally rubric-based.

---

# 24. LLM-Rubric Hyperparameters

The official repository reports the following hyperparameters for its
real-data evaluation experiment:

    input_size:          36
    output_size:         9
    num_judges:          13
    all_data_size:       223
    finetune_output:     -1
    num_answers:         4
    batch_size:           64
    learning_rate:       0.001
    layer1_size:          25
    layer2_size:          25
    pretraining_epochs:  20
    finetuning_epochs:   30
    random_seed:          43

These are the repository's reported experiment settings.

They should NOT automatically be copied to our student-answer dataset.

Our task has a different structure and scale.

We will determine appropriate hyperparameters for our dataset.

The official repository provides scripts for finding hyperparameters on
new data.

Reference:

https://github.com/microsoft/LLM-Rubric

---

# 25. Important Difference Between Our Task and Original LLM-Rubric

The original LLM-Rubric research evaluates natural-language responses using
multiple rubric questions and models human judgments.

Our problem is educational assessment.

Our rubric dimensions may be:

    - Definition correctness
    - Required concepts
    - Explanation correctness
    - Examples
    - Technical accuracy
    - Completeness
    - Misconceptions

Therefore we will adapt the LLM-Rubric methodology rather than assuming
the original task configuration is directly applicable.

---

# 26. Planned LLM-Rubric Pipeline

The proposed pipeline is:

    Student Answer
          |
          v
       Rubric
          |
          v
    LLM evaluation questions
          |
          +----------------------+
          |          |           |
          v          v           v
       Criterion  Criterion   Criterion
          1          2           3
          |          |           |
          +----------+-----------+
                     |
                     v
             LLM predictions
                     |
                     v
             Calibration model
                     |
                     v
               Final grade

The calibration model will be trained using human-graded examples.

---

# 27. LLM-Rubric Testing Strategy

LLM-Rubric will be evaluated using the same student answers and rubrics
where possible.

The comparison will eventually look like:

    Human Grade
          |
          +-----------------------------+
          |             |               |
          v             v               v
    Sentence       ISCC-SCT       LLM-Rubric
    Transformer

For every question, we will compare predicted marks with human marks.

Example:

    Question | Human | ST | ISCC | LLM-Rubric
    ---------|-------|----|------|-----------
    Q1       | 4.0   |3.8 | 3.5  | 4.1
    Q2       | 5.0   |4.7 | 4.2  | 4.8
    Q3       | 4.0   |3.6 | 3.7  | 4.0
    Q4       | 3.0   |3.5 | 3.1  | 3.0

This will allow us to determine which approach is closest to human grading.

---

# 28. What "Accuracy" Means in This Project

We should avoid using only semantic similarity percentage as accuracy.

For educational grading, accuracy should be defined using agreement with
human/expert grades.

Possible measurements:

## Mean Absolute Error

    MAE = average(|predicted_mark - human_mark|)

Lower is better.

---

## Root Mean Squared Error

    RMSE = sqrt(mean((predicted_mark - human_mark)^2))

Lower is better.

RMSE penalizes larger errors more strongly.

---

## Correlation

Pearson correlation can measure linear agreement.

Spearman correlation can measure rank-order agreement.

These metrics are particularly useful for comparing automatic graders
against human annotations.

---

# 29. Current Limitations

The current prototype has several limitations.

## 29.1 Small Dataset

Currently we only have one student's answer set with four questions.

This is far too small to train or properly validate a grading model.

---

## 29.2 No Human Ground Truth

We currently do not have independently assigned expert marks for a
large number of student answers.

Without this, we cannot calculate reliable grading accuracy.

---

## 29.3 Heuristic Threshold

The current threshold:

    0.50

was selected as an initial experimental value.

It has not been statistically calibrated.

---

## 29.4 Heuristic Score Mapping

The current formula:

    marks = maximum_marks * similarity

is a simple baseline.

Educational marks do not necessarily scale linearly with semantic
similarity.

---

## 29.5 Semantic Similarity != Factual Correctness

An answer can be semantically similar but technically wrong.

---

## 29.6 Whole-Answer Comparison

Comparing the complete student answer with a criterion can hide:

    - missing concepts
    - incorrect explanations
    - contradictions
    - misconceptions

---

## 29.7 Minimum Count Criteria

The current implementation of `minimum_count` is not sufficient for
checking both:

    "Did the student mention the concept?"

and:

    "Did the student correctly explain the concept?"

This must be improved.

---

## 29.8 No Feedback Generation

The current system primarily produces scores.

It does not yet reliably generate useful teacher-style feedback.

---

## 29.9 No Calibration

Neither Sentence Transformers nor ISCC-SCT has yet been calibrated against
human educational grading.

---

# 30. Security and Privacy Considerations

If real student examination data is used, personally identifiable information
should not be stored in a public repository.

Before publishing data:

    - anonymize student identities
    - remove student IDs
    - remove personal information
    - remove institutional information
    - obtain required permissions
    - avoid committing private examination material

The current repository should preferably contain synthetic or anonymized
examples.

---

# 31. Current Status

Current status:

    [x] Clone Sentence Transformers
    [x] Create isolated Python environment
    [x] Install Sentence Transformers
    [x] Test semantic similarity
    [x] Create student/model/rubric JSON files
    [x] Implement initial Sentence Transformer grader

    [x] Clone ISCC-SCT
    [x] Create isolated Python environment
    [x] Install ISCC-SCT
    [x] Test Semantic Text-Code generation
    [x] Test Hamming-distance comparison
    [x] Implement initial ISCC-SCT grader

    [x] Create result files
    [x] Create comparison script

    [ ] Improve criterion-level grading
    [ ] Add sentence/claim-level matching
    [ ] Evaluate Cross-Encoder
    [ ] Collect human-graded dataset
    [ ] Calibrate thresholds
    [ ] Calculate MAE/RMSE/correlation
    [ ] Clone/install LLM-Rubric
    [ ] Adapt rubric for educational grading
    [ ] Train/calibrate LLM-Rubric
    [ ] Compare LLM-Rubric against baselines
    [ ] Select final approach

---

# 32. Planned Experimental Roadmap

## Phase 1 - Semantic Baselines

Completed/ongoing:

    Sentence Transformers
             +
          ISCC-SCT

Goal:

    Establish simple semantic-similarity baselines.

---

## Phase 2 - Improve Sentence-Level Evaluation

Planned:

    Sentence Transformer
          +
    sentence/claim segmentation
          +
    criterion matching

Goal:

    Improve handling of multi-concept answers.

---

## Phase 3 - Cross-Encoder

Evaluate a Cross-Encoder model using:

    Student Answer + Rubric Criterion

Goal:

    Determine whether joint pair scoring improves over independent
    embeddings and cosine similarity.

---

## Phase 4 - Human-Graded Dataset

Create a dataset containing:

    Question
    Student Answer
    Model Answer
    Rubric
    Human Marks

Preferably multiple human/expert graders should evaluate a subset so that
inter-rater agreement can also be studied.

---

## Phase 5 - Calibration

Use human marks to optimize:

    similarity thresholds
    scoring functions
    criterion weights

Evaluate:

    MAE
    RMSE
    Pearson
    Spearman
    Kendall
    agreement within tolerance

---

## Phase 6 - LLM-Rubric

Install and reproduce the Microsoft LLM-Rubric implementation.

Repository:

https://github.com/microsoft/LLM-Rubric

Adapt the rubric questions to educational assessment.

Use the student answer and rubric as input.

Train/calibrate using human-graded data.

---

## Phase 7 - Final Comparison

Compare:

    Human
       |
       +---- Sentence Transformer
       |
       +---- ISCC-SCT
       |
       +---- Cross-Encoder
       |
       +---- LLM-Rubric

Evaluate both:

    Accuracy/agreement
    Computational cost
    Inference time
    Memory requirements
    Ease of deployment
    Explainability
    Robustness
    Scalability

---

# 33. Expected Final System

The desired final architecture is:

    Student Answer
          |
          v
    Answer Preprocessing
          |
          v
    Question + Rubric
          |
          v
    Criterion Evaluation
          |
          +-----------------------+
          |                       |
          v                       v
    Semantic Model             LLM Model
          |                       |
          +-----------+-----------+
                      |
                      v
               Score Calibration
                      |
                      v
                 Final Marks
                      |
          +-----------+-----------+
          |                       |
          v                       v
       Score                  Feedback
          |                       |
          +-----------+-----------+
                      |
                      v
              Teacher/Student Report

---

# 34. Final Objective

The final objective is not simply to find a model that produces a high
similarity score.

The objective is to develop an automated assessment system that:

    1. Understands the expected answer
    2. Understands the grading rubric
    3. Identifies concepts present in the student answer
    4. Identifies missing concepts
    5. Detects incorrect explanations
    6. Assigns marks consistently
    7. Produces useful feedback
    8. Agrees closely with expert human graders
    9. Can process many answer sheets efficiently

The two current semantic approaches provide the baseline required to
measure whether more sophisticated approaches provide a meaningful
improvement.

---

# 35. Overall Conclusion

Sentence Transformers and ISCC-SCT successfully demonstrate that semantic
similarity can be used as an initial mechanism for comparing student answers
with reference material.

However, the experiments also show an important conceptual limitation:

    Semantic similarity is not equivalent to educational correctness.

A grading system must evaluate individual rubric criteria and detect both
presence and correctness of concepts.

Therefore, the current systems should be considered baseline experiments,
not production-ready automatic graders.

The next major step is to obtain a sufficiently large human-graded dataset,
improve criterion-level evaluation, evaluate stronger pairwise models such
as Cross-Encoders, and then investigate LLM-Rubric as a calibrated
rubric-based evaluation system.

The final model will be selected based on empirical agreement with human
grading rather than raw semantic similarity.

---

# 36. References

## Sentence Transformers

https://github.com/huggingface/sentence-transformers

## ISCC-SCT

https://github.com/iscc/iscc-sct

## LLM-Rubric

https://github.com/microsoft/LLM-Rubric

## LLM-Rubric Paper

https://aclanthology.org/2024.acl-long.745/

## Sentence Transformers Documentation

https://www.sbert.net/
