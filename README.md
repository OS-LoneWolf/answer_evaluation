# Technical Report — Student Answer Evaluation Pipeline

## 1. Project objective

The project is building an **automated student-answer evaluation system** that can estimate how well a student's answer matches the expected knowledge in a reference/model answer.

The important point is that we are **not simply trying to measure text similarity**.

The final system is intended to answer something closer to:

> *“Given this question, the expected answer, and the student's answer, how correct, complete, relevant, and clear is the student's response, and what score would a human evaluator likely assign?”*

The development is divided into four stages so that each increasingly sophisticated approach can be evaluated against the previous one.

The overall pipeline is:

```text
                    Student Answer
                         │
          ┌──────────────┼──────────────┐
          │              │              │
          ▼              ▼              ▼
       TF-IDF      Sentence Transformer  LLM
          │              │              │
          ▼              ▼              ▼
       Baseline       Baseline       Probabilistic
                                      Evaluation
                                          │
                                          ▼
                                   Calibration Model
                                          │
                                          ▼
                                    Final Score
```

The stages are deliberately separated because we want to establish **empirical evidence that each additional level of sophistication actually improves evaluation quality**.

---

# 2. Dataset and ground truth

Our evaluation dataset lives under:

```text
/home/robot/Projects/intership/student_evaluation/data/
```

The important inputs include:

```text
dsa_model_answers.json
provisional_human_benchmark.json
provisional_human_benchmark.csv

student_1.json
student_2.json
student_3.json
student_4.json
student_5.json
student_6.json
student_7.json

student_evaluation_rubric.csv
```

The dataset currently contains:

* **7 students**
* **28 evaluation records**
* effectively **4 questions per student**

The reference answers come from:

```text
dsa_model_answers.json
```

Student responses come from the corresponding:

```text
student_*.json
```

The provisional human benchmark is our closest thing to **ground truth**:

```text
provisional_human_benchmark.csv
```

This is important because every automated evaluator needs something to be compared against.

---

# 3. Stage 1 — Classical baselines

Stage 1 answers a very important research question:

> **How well can relatively simple text-comparison techniques evaluate student answers without an LLM?**

We intentionally use two different approaches.

```text
                 Student Answer
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
       TF-IDF              Sentence Transformer
          │                         │
          ▼                         ▼
    Cosine Similarity         Embedding Similarity
          │                         │
          └────────────┬────────────┘
                       ▼
                Predicted Score
```

These are **baselines**, not the final system.

---

# 4. Stage 1A — TF-IDF baseline

Implementation:

```text
/home/robot/Projects/intership/baseline/
```

with the TF-IDF baseline script.

Conceptually, TF-IDF converts text into a numerical vector.

For a term (t) in document (d):

[
TFIDF(t,d)=TF(t,d)\times IDF(t)
]

where:

[
IDF(t)=\log\left(\frac{N}{df(t)}\right)
]

The resulting student-answer vector is compared with the model-answer vector using cosine similarity:

[
\cos(\theta)=
\frac{A\cdot B}
{|A||B|}
]

A high cosine similarity means the student answer contains similar textual vocabulary and distribution to the reference answer.

We then map that similarity into an estimated score.

### Why this is useful

TF-IDF gives us a deliberately simple reference point.

For example:

```text
Model answer:
A queue is a linear data structure...

Student:
A queue stores elements in FIFO order...
```

The two answers might express the same knowledge using different wording.

TF-IDF may fail because lexical overlap is relatively low.

Conversely:

```text
Student:
A queue is a linear data structure...
```

could achieve high similarity even if the answer contains technically incorrect information later.

Therefore TF-IDF measures:

> **lexical similarity**

rather than actual conceptual correctness.

---

# 5. Stage 1B — Sentence Transformer semantic baseline

The second baseline uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

This converts sentences/text into dense semantic embeddings.

Conceptually:

[
f(text)\rightarrow \mathbf{e}\in\mathbb{R}^{384}
]

Two texts can therefore be close in embedding space even if they use different vocabulary.

For example:

```text
"The queue follows FIFO."

"A queue removes the earliest inserted element first."
```

may have relatively high semantic similarity despite limited word overlap.

The system calculates similarity between:

```text
student answer embedding
            │
            │ cosine similarity
            ▼
model answer embedding
```

and maps that similarity into an estimated score.

---

# 6. Actual Stage 1 results

We have already executed Stage 1.

### Sentence Transformer

```text
Records       : 28
MAE           : 1.1059
RMSE          : 1.4428
Pearson r     : 0.9452
Spearman rho  : 0.2377
```

### TF-IDF

```text
Records       : 28
MAE           : 4.4961
RMSE          : 4.9010
Pearson r     : 0.7875
Spearman rho  : 0.4198
```

The outputs are stored under:

```text
/home/robot/Projects/intership/student_evaluation/results/phase2_stage1/
```

with:

```text
semantic/
    semantic_results.json
    semantic_question_scores.csv
    semantic_summary.json

tfidf/
    tfidf_results.json
    tfidf_question_scores.csv
    tfidf_summary.json
```

---

# 7. How to interpret those metrics

There are four important metrics.

## MAE

Mean Absolute Error:

[
MAE=
\frac{1}{N}
\sum_{i=1}^{N}
|y_i-\hat y_i|
]

If:

```text
human score = 8
predicted = 7
```

the error is:

```text
1
```

Lower is better.

---

## RMSE

Root Mean Squared Error:

[
RMSE=
\sqrt{
\frac{1}{N}
\sum_{i=1}^{N}
(y_i-\hat y_i)^2
}
]

RMSE penalizes large mistakes more heavily than MAE.

This matters because an evaluator that usually predicts correctly but occasionally gives a completely wrong score should be penalized.

Lower is better.

---

# 8. Pearson correlation

Pearson correlation measures the strength of a **linear relationship** between human and predicted scores.

[
r =
\frac{cov(X,Y)}
{\sigma_X\sigma_Y}
]

Our semantic baseline obtained:

```text
Pearson r = 0.9452
```

which indicates a strong linear relationship in this dataset.

However, correlation alone does **not** mean the predictions are numerically accurate.

For example:

```text
Human:      5   6   7   8
Prediction: 7   8   9  10
```

could still have very high correlation despite systematically overestimating.

That's why MAE/RMSE are also necessary.

---

# 9. Spearman correlation

Spearman correlation measures whether the evaluator preserves **rank ordering**.

For example, suppose humans rank answers:

```text
A > B > C
```

and the model produces:

```text
A > B > C
```

then ranking performance is good even if absolute scores differ.

This is useful because automated grading has two related but different objectives:

1. Predict the actual score.
2. Correctly distinguish stronger and weaker answers.

---

# 10. What Stage 1 tells us

Our Stage 1 experiment already gives us an interesting baseline.

The Sentence Transformer substantially outperforms TF-IDF in MAE/RMSE:

```text
                    MAE       RMSE
TF-IDF              4.4961    4.9010
Semantic            1.1059    1.4428
```

That provides evidence that **semantic representation is much more useful than purely lexical overlap for this dataset**.

But neither baseline understands the actual grading criteria.

They don't explicitly understand:

```text
Correctness
Completeness
Relevance
Clarity
```

That motivates Stage 2.

---

# 11. Stage 2 — LLM evaluator

Stage 2 introduces a local LLM:

```text
Qwen3:4B
```

running through:

```text
Ollama
```

The important architectural decision is that this is intended to operate **offline**.

There is no requirement for an external OpenAI/API call during evaluation.

The pipeline becomes:

```text
Question
    +
Model Answer
    +
Student Answer
    +
Rubric
    │
    ▼
 Qwen3:4B
    │
    ▼
Probability distributions
```

---

# 12. Why we don't simply ask the LLM for an 8/10

A naive implementation would ask:

```text
Score this answer from 1 to 10.
```

and receive:

```text
8
```

That throws away a lot of information.

Instead, our approach asks the model to produce a **probability distribution over scores 1–10**.

For example:

```text
Correctness

1  → 0.01
2  → 0.02
3  → 0.03
4  → 0.05
5  → 0.08
6  → 0.12
7  → 0.20
8  → 0.30
9  → 0.15
10 → 0.04
```

The probabilities should satisfy:

[
\sum_{k=1}^{10}P(k)=1
]

This is much richer than:

```text
correctness = 8
```

---

# 13. Four LLM evaluation dimensions

The LLM evaluates four dimensions:

### Correctness

Is the answer factually and technically correct?

### Completeness

Does it cover the important concepts expected by the question/reference answer?

### Relevance

Does it actually answer the question without unnecessary material?

### Clarity

Is the response understandable, organized and sufficiently clear?

The rubric assigns:

```text
Correctness   = 0.40
Completeness  = 0.30
Relevance     = 0.15
Clarity       = 0.15
```

Therefore correctness has the greatest influence on the final evaluation.

---

# 14. Converting distributions into expected scores

Suppose the LLM produces:

```text
P(7) = 0.2
P(8) = 0.5
P(9) = 0.3
```

The expected score is:

[
E[S]
====

7(0.2)+8(0.5)+9(0.3)
]

[
E[S]=8.1
]

So rather than throwing away the distribution, we can retain:

```text
distribution
+
expected score
+
uncertainty
```

This is particularly important for Stage 3.

---

# 15. Why this is LLM-Rubric-inspired

We are **not simply running the original LLM-Rubric repository as a black-box component**.

Instead, our architecture adopts the relevant methodological idea:

> **Use an LLM to produce richer probabilistic evaluation signals, then learn/calibrate those signals against human evaluation.**

This is much more appropriate for our project because we have:

```text
Student answers
Model answers
Human benchmark
LLM evaluations
```

and therefore can train a downstream calibration model.

The distinction is important when explaining the project:

> **Our system is LLM-Rubric-inspired rather than a direct reproduction of the original implementation.**

---

# 16. Stage 2 architecture

The Stage 2 input is effectively:

```text
┌───────────────────────────┐
│ Question                  │
├───────────────────────────┤
│ Reference answer          │
├───────────────────────────┤
│ Student answer            │
├───────────────────────────┤
│ Evaluation rubric         │
└──────────────┬────────────┘
               │
               ▼
          Qwen3:4B
               │
               ▼
┌──────────────────────────────┐
│ Correctness distribution     │
│ Completeness distribution    │
│ Relevance distribution       │
│ Clarity distribution         │
└──────────────────────────────┘
```

The important point is that **Stage 2 does not yet have to be perfectly calibrated**.

It provides features/signals for Stage 3.

---

# 17. Stage 2 output

Our Stage 2 implementation is designed to store its results under:

```text
/home/robot/Projects/intership/student_evaluation/results/phase2_stage2/llm/
```

The exact Stage 2 output will become the input to the calibration stage.

This is why we designed Stage 3 and Stage 4 to be independent of actually running Qwen locally.

Your friend can run Stage 2 on a faster machine, generate the results, and those results can then be brought back into the project.

---

# 18. The important hardware consideration

Our local Qwen3:4B test demonstrated that the model is usable but CPU inference is slow.

The issue isn't that the methodology is wrong.

The issue is:

```text
Qwen3:4B
     │
     ▼
100% CPU inference
     │
     ▼
slow token generation
```

The system therefore shouldn't be architecturally tied to one particular machine.

This is why the project is being designed so that:

```text
Stage 1
   ↓
Stage 2 results
   ↓
Stage 3
   ↓
Stage 4
```

can be executed separately.

The Stage 2 results can be generated on a stronger machine and transferred back.

---

# 19. Stage 3 — Calibration

Stage 3 is where the project becomes significantly more interesting from a research perspective.

We now have:

```text
LLM evaluation
       │
       ▼
probability distributions
       │
       ▼
calibration model
       │
       ▼
human-like score
```

The fundamental problem is:

> **An LLM's evaluation is not necessarily calibrated to our human benchmark.**

Suppose the LLM consistently evaluates strong answers too generously.

For example:

```text
Human:
8.0

LLM:
9.1
```

or:

```text
Human:
7.2

LLM:
8.5
```

A calibration model can learn that systematic behavior.

---

# 20. Calibration input features

The calibration model can use information such as:

```text
Correctness distribution
Completeness distribution
Relevance distribution
Clarity distribution
```

From those distributions we can derive features such as:

```text
Expected correctness
Expected completeness
Expected relevance
Expected clarity
```

plus uncertainty-related information.

For example:

[
E[C]=\sum_{k=1}^{10}kP(C=k)
]

and similarly:

[
E[Co],E[R],E[Cl]
]

We can also derive:

```text
variance
entropy
maximum probability
most likely score
```

from each distribution.

This means the calibration model isn't limited to:

```text
LLM score = 8.7
```

It can see **how confident the LLM was in that score**.

---

# 21. Example of the richer representation

Suppose:

```text
Correctness:
8 → 0.70
9 → 0.25
10 → 0.05
```

versus:

```text
Correctness:
6 → 0.20
7 → 0.20
8 → 0.20
9 → 0.20
10 → 0.20
```

Both could have similar expected values.

But they mean completely different things.

The first says:

> "I am fairly confident this answer is around 8."

The second says:

> "I have very little confidence about the appropriate score."

A calibration model can potentially exploit that distinction.

---

# 22. Human benchmark becomes the training target

This is the key transition.

Stage 2:

```text
Student answer
      ↓
LLM
      ↓
LLM evaluation
```

Stage 3:

```text
Student answer
      ↓
LLM
      ↓
LLM evaluation
      ↓
Calibration NN
      ↓
Human benchmark score
```

The human benchmark becomes the target variable:

[
X = \text{LLM-derived features}
]

[
y = \text{human score}
]

The calibration model learns:

[
f(X)\approx y
]

---

# 23. Why a neural network?

The calibration layer can be a small neural network because the number of inputs is relatively small and the objective is nonlinear calibration.

For example:

```text
LLM features
      │
      ▼
Dense layer
      │
      ▼
Activation
      │
      ▼
Dense layer
      │
      ▼
Predicted human score
```

It doesn't need to be a huge model.

The LLM already provides the linguistic intelligence.

The calibration model's job is different:

> **learn the relationship between LLM judgments and our human benchmark.**

This is an important conceptual separation.

---

# 24. Stage 3 is not another LLM

This is worth emphasizing.

We are not doing:

```text
LLM → another LLM → score
```

Instead:

```text
LLM
 │
 │ probabilistic evaluation
 ▼
small supervised calibration model
 │
 │ trained against human benchmark
 ▼
human-aligned score
```

That makes the architecture computationally much cheaper after Stage 2.

---

# 25. Stage 4 — Final evaluator

Stage 4 combines everything into a usable evaluation system.

The intended architecture is:

```text
                    QUESTION
                       │
          ┌────────────┴────────────┐
          │                         │
          ▼                         ▼
    MODEL ANSWER              STUDENT ANSWER
          │                         │
          └────────────┬────────────┘
                       │
                       ▼
                Evaluation LLM
                       │
         ┌─────────────┼─────────────┐
         │             │             │
         ▼             ▼             ▼
    Correctness   Completeness   Relevance
         │             │             │
         └─────────────┼─────────────┘
                       │
                    Clarity
                       │
                       ▼
              LLM probability data
                       │
                       ▼
              Calibration model
                       │
                       ▼
                 Final Score
```

---

# 26. Final score

The final score should represent the calibrated human-like evaluation.

For example:

```text
Final score: 8.3 / 10
```

Alongside it we retain the dimension scores:

```text
Correctness:   8.8
Completeness:  7.9
Relevance:     9.2
Clarity:       8.1
```

The exact final aggregation should remain consistent with the trained calibration architecture rather than arbitrarily averaging numbers after training.

---

# 27. Feedback generation

The final system can also produce interpretable feedback.

For example:

```text
Score: 8.3 / 10

Correctness:    8.8
Completeness:   7.9
Relevance:      9.2
Clarity:        8.1

Missing concepts:
- Concept A
- Concept B

Suggestions:
- Explain Concept A more explicitly.
- Include an example of Concept B.
```

This is where the system becomes useful educationally rather than being merely a numerical regression model.

---

# 28. The crucial distinction between scoring and feedback

There are actually two different tasks:

### Task A — scoring

```text
answer → score
```

This is what we are primarily evaluating scientifically.

### Task B — feedback

```text
answer → explanation / missing concepts / suggestions
```

Feedback can be generated from the LLM's evaluation.

We should **not allow feedback generation to contaminate the benchmark score**.

The numerical score should come from the controlled evaluation/calibration pipeline.

---

# 29. Full four-stage progression

The entire research progression can therefore be summarized as:

### Stage 1

```text
Text
 ↓
TF-IDF / Sentence Transformer
 ↓
Similarity
 ↓
Baseline score
```

Question being answered:

> How far can conventional similarity methods get us?

---

### Stage 2

```text
Question
+
Reference
+
Student answer
+
Rubric
 ↓
Qwen3:4B
 ↓
4 probabilistic evaluations
```

Question:

> Can a local LLM reason about multiple grading dimensions better than similarity?

---

### Stage 3

```text
LLM distributions
       +
Human benchmark
       ↓
Calibration NN
       ↓
Calibrated score
```

Question:

> Can we systematically correct the LLM's grading bias using human data?

---

### Stage 4

```text
Question
Reference
Student
   │
   ▼
LLM
   │
   ▼
Probabilistic rubric evaluation
   │
   ▼
Calibration model
   │
   ▼
Final score
   │
   ├── Dimension scores
   ├── Missing concepts
   └── Feedback
```

Question:

> Can we turn the research pipeline into a complete student-answer evaluation system?

---

# 30. Why the comparison is scientifically useful

At the end, we can compare:

```text
                         MAE     RMSE     Pearson     Spearman
──────────────────────────────────────────────────────────────
TF-IDF                   ...      ...       ...          ...
Sentence Transformer     ...      ...       ...          ...
LLM                      ...      ...       ...          ...
LLM + Calibration        ...      ...       ...          ...
```

This gives us a clear experimental story.

We aren't claiming:

> "LLMs are better because they're more advanced."

Instead, we can experimentally test:

[
Performance_{TFIDF}
]

versus

[
Performance_{Semantic}
]

versus

[
Performance_{LLM}
]

versus

[
Performance_{LLM+Calibration}
]

That is a much stronger research argument.

---

# 31. Current project architecture

The repository is organized roughly as:

```text
/home/robot/Projects/intership/

├── .git/
│
├── student_evaluation/
│   ├── data/
│   │
│   └── results/
│       ├── phase2_stage1/
│       │   ├── semantic/
│       │   └── tfidf/
│       │
│       ├── phase2_stage2/
│       │   └── llm/
│       │
│       ├── phase2_stage3/
│       │
│       └── phase2_stage4/
│
├── baseline/
│   ├── ...
│   └── llm/
│       └── run_llm_baseline.py
│
└── LLM-Rubric/
```

This separation is intentional.

`student_evaluation/` is primarily the **data/results area**.

The actual experimental code lives under the project-level directories such as:

```text
baseline/
```

and the LLM-Rubric repository is kept separately because our implementation is **inspired by its methodology rather than simply replacing our project with that repository**.

---

# 32. Environment separation

Another important engineering decision is that the project does not force every component into one Python environment.

For example:

```text
Sentence Transformer environment
        │
        └── Stage 1 semantic baseline

TF-IDF environment
        │
        └── Stage 1 TF-IDF baseline

LLM-Rubric / Python 3.10 environment
        │
        └── Stage 2 / calibration-related components
```

Ollama itself is outside the Python virtual environment:

```text
Python program
      │
      ▼
Ollama HTTP/API
      │
      ▼
Qwen3:4B
```

This makes the system easier to reproduce on another machine.

---

# 33. Offline design

The eventual target is:

```text
Internet
   X
```

during actual evaluation.

Instead:

```text
Local Python
     │
     ▼
Local Ollama
     │
     ▼
Local Qwen3:4B
```

The model is already downloaded locally.

The Sentence Transformer model can similarly be cached locally.

Therefore, once the required models/packages are installed, the actual evaluation pipeline can operate without external model APIs.

This is particularly useful for:

* privacy
* reproducibility
* avoiding API costs
* controlled experiments
* offline academic environments

---

# 34. One important limitation

Our current dataset is relatively small:

```text
28 evaluation records
```

That matters enormously for Stage 3.

A neural calibration model trained on only 28 examples can easily overfit.

Therefore, when we eventually evaluate Stage 3/4, we need to be careful about claiming generalization.

A strong experimental approach would use something like:

```text
training set
validation set
test set
```

or cross-validation where appropriate.

For a tiny dataset, cross-validation may be more statistically meaningful than simply splitting 28 examples once.

This should be treated as an experimental-design issue, not hidden.

---

# 35. What the project is ultimately demonstrating

The central research hypothesis can be expressed as:

> **Semantic similarity provides a stronger baseline than lexical similarity, an LLM can provide richer multidimensional probabilistic evaluation, and supervised calibration of those LLM judgments can potentially produce scores that better align with human evaluation.**

The experimental chain is therefore:

[
\text{Lexical Similarity}
\rightarrow
\text{Semantic Similarity}
\rightarrow
\text{LLM Evaluation}
\rightarrow
\text{Calibrated LLM Evaluation}
]

Each step adds a different capability.

---

## 36. The simplest explanation to give another researcher

If you need to explain the entire project in two minutes, I'd phrase it like this:

> **We are building a student-answer evaluation system in four stages. First, we establish conventional baselines using TF-IDF and Sentence Transformer cosine similarity. This tells us how well simple lexical and semantic similarity can approximate human scores. Second, we replace similarity-based evaluation with a local Qwen3:4B evaluator that receives the question, reference answer, student answer and rubric, and outputs probability distributions over 1–10 for correctness, completeness, relevance and clarity. Third, inspired by LLM-Rubric, we treat those probabilistic LLM evaluations as features and train a small calibration model against our human benchmark. This allows the system to learn systematic biases in the LLM's grading. Finally, Stage 4 packages the calibrated evaluator into an end-to-end system producing a final score, dimension-level scores and qualitative feedback. The key experimental comparison is TF-IDF → semantic similarity → raw LLM evaluation → calibrated LLM evaluation, measured against the human benchmark using MAE, RMSE, Pearson and Spearman correlations.**

That is the technical story of the project.
