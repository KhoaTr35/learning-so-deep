# Assignment 3: Deployment and Model Optimization

## Description

Convert research models into deployable systems with constraints:
- low latency
- low memory footprint
- stable throughput
- maintainable inference pipeline

Goal: keep quality high while meeting production-style performance targets.

---

## Scope

| Area | Baseline | Advanced |
|---|---|---|
| Model export | Native checkpoint | ONNX / TorchScript |
| Optimization | FP32 | Quantization + pruning |
| Serving | Single request | Batched and profiled inference |

---

## Experiment Template (Advanced)

### 1. Problem Statement

- Deployment target platform:
- Service-level objective (SLO):
- Quality floor metric:
- Hard constraints (RAM/CPU/GPU):

### 2. System Profile

| Component | Version/Type | Notes |
|---|---|---|
| Runtime | | |
| Hardware | | |
| Serving framework | | |
| Batch policy | | |

### 3. Hypotheses

1. `H1`:  
2. `H2`:  
3. `H3`:  

### 4. Optimization Matrix

| Exp ID | Technique | Target Model | Expected Gain | Risk |
|---|---|---|---|---|
| A3-E01 | | | | |
| A3-E02 | | | | |
| A3-E03 | | | | |

### 5. Benchmark Dashboard

| Exp ID | Accuracy/F1 | P50 Latency (ms) | P95 Latency (ms) | Throughput (req/s) | Model Size (MB) |
|---|---:|---:|---:|---:|---:|
| A3-E01 | | | | | |
| A3-E02 | | | | | |
| A3-E03 | | | | | |

### 6. Cost and Efficiency

| Exp ID | CPU Utilization | GPU Utilization | Memory Peak | Estimated Cost / 1M req |
|---|---:|---:|---:|---:|
| A3-E01 | | | | |
| A3-E02 | | | | |
| A3-E03 | | | | |

### 7. Risk Review

- Numerical stability risks:
- Post-optimization regressions:
- Edge case inputs:
- Fallback strategy:

### 8. Reproducibility Checklist

- Benchmark script committed
- Warm-up policy fixed
- Load generation parameters documented
- Versioned exported artifacts archived

### 9. Conclusions

- Best deployment candidate:
- Tradeoffs accepted:
- Final recommendation:

---

[Back to Home](index.md)
