# Assignment 1: Deep Learning Models Comparison

## Description

Compare model families across three tasks:
- image classification
- text classification
- multimodal classification

Goal: build strong baselines, evaluate rigorously, and document tradeoffs.

---

## Scope

| Track | Baseline | Advanced |
|---|---|---|
| Image | CNN | Vision Transformer |
| Text | LSTM | Transformer Encoder |
| Multimodal | Late Fusion | Zero-shot / Few-shot |

---

## Experiment Template (Advanced)

### 1. Problem Statement

- Task objective:
- Primary metric:
- Success threshold:
- Constraints (compute/time/data):

### 2. Dataset Card

| Field | Details |
|---|---|
| Dataset name | |
| Source | |
| Train/Val/Test split | |
| Number of classes | |
| Known biases | |
| License | |

### 3. Hypotheses

1. `H1`:  
2. `H2`:  
3. `H3`:  

### 4. Experiment Matrix

| Exp ID | Model | Key Change | LR | Batch Size | Epochs | Notes |
|---|---|---|---:|---:|---:|---|
| A1-E01 | | | | | | |
| A1-E02 | | | | | | |
| A1-E03 | | | | | | |

### 5. Results Dashboard

| Exp ID | Accuracy | Precision | Recall | F1 | Inference Time (ms) | Params (M) |
|---|---:|---:|---:|---:|---:|---:|
| A1-E01 | | | | | | |
| A1-E02 | | | | | | |
| A1-E03 | | | | | | |

### 6. Error Analysis

- Most frequent failure class:
- Confusion pairs:
- Data quality issues found:
- Labeling ambiguities:

### 7. Ablation Study

| Variant | Removed/Changed Component | Metric Delta | Interpretation |
|---|---|---:|---|
| Abl-01 | | | |
| Abl-02 | | | |

### 8. Reproducibility Checklist

- Random seed recorded
- Environment and package versions pinned
- Data preprocessing steps fixed
- Training command documented
- Model checkpoints saved

### 9. Conclusions

- Best model and why:
- What failed and why:
- Next iteration plan:

---

[Back to Home](index.md)
