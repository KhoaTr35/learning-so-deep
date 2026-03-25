# Assignment 2: Robustness and Generalization Study

## Description

Study how model performance changes under realistic distribution shifts:
- noisy inputs
- class imbalance
- limited labeled data
- domain transfer

Goal: quantify robustness and improve generalization under non-ideal conditions.

---

## Scope

| Area | Baseline | Advanced |
|---|---|---|
| Shift Testing | IID validation | OOD robustness suite |
| Data Strategy | Standard split | Stratified + stress split |
| Model Strategy | Single model | Ensembling / consistency regularization |

---

## Experiment Template (Advanced)

### 1. Problem Statement

- Robustness question:
- Generalization target:
- Primary robustness metric:
- Deployment assumptions:

### 2. Dataset Shift Profile

| Shift Type | Construction Method | Severity Levels | Purpose |
|---|---|---|---|
| Noise shift | | | |
| Style/domain shift | | | |
| Class prior shift | | | |
| Label sparsity shift | | | |

### 3. Hypotheses

1. `H1`:  
2. `H2`:  
3. `H3`:  

### 4. Experiment Matrix

| Exp ID | Scenario | Model | Defense/Method | Metric | Notes |
|---|---|---|---|---|---|
| A2-E01 | | | | | |
| A2-E02 | | | | | |
| A2-E03 | | | | | |

### 5. Results Dashboard

| Exp ID | IID Score | OOD Score | Robustness Gap | Calibration Error | Notes |
|---|---:|---:|---:|---:|---|
| A2-E01 | | | | | |
| A2-E02 | | | | | |
| A2-E03 | | | | | |

### 6. Stability and Sensitivity

| Factor | Tested Values | Best Value | Sensitivity Observation |
|---|---|---|---|
| Learning rate | | | |
| Weight decay | | | |
| Augmentation strength | | | |

### 7. Error Analysis

- Most fragile scenarios:
- Spurious correlations detected:
- Confidence overestimation cases:
- Failure examples and root causes:

### 8. Reproducibility Checklist

- Shift generation scripts versioned
- Seed list and run count recorded
- Evaluation protocol frozen before final run
- Confidence intervals reported

### 9. Conclusions

- Most robust configuration:
- Practical tradeoffs (accuracy vs robustness):
- Next iteration plan:

---

[Back to Home](index.md)
