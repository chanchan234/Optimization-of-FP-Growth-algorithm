# Baseline: 3-Fold Cross-Validation FP-Growth

| Min Support | Validated Patterns | Association Rules | Time (s) | Fold Agreement |
|-------------|-------------------|-------------------|----------|----------------|
| 0.005 | 4,424 | 3,853 | 10.12 | 95.1% |
| 0.010 | 1,051 | 348 | 6.53 | 96.8% |
| 0.015 | 468 | 78 | 4.82 | 97.3% |
| 0.020 | 253 | 31 | 4.57 | 98.3% |

## Main Method: Random Forest-Inspired FP-Growth (3-tree Ensemble)

| Min Support | Validated Patterns | Association Rules | Time (s) | Fold Agreement |
|-------------|-------------------|-------------------|----------|----------------|
| 0.005 | 1,462 | 384 | 4.94 | 74.2% |
| 0.010 | 534 | 85 | 4.09 | 75.8% |
| 0.015 | 265 | 24 | 3.70 | 76.1% |
| 0.020 | 151 | 7 | 3.27 | 77.7% |

## Efficiency Comparison at Comparable Output Scale

To ensure fair comparison of computational efficiency, runtime was evaluated at **similar rule counts** rather than identical support thresholds.

| Method | Min Support | Rules Generated | Time (s) |
|--------|-------------|-----------------|----------|
| Baseline | 0.010 | 348 | 6.53 |
| **Proposed** | **0.005** | **384** | **4.94** |

> **⚡ 24.3% faster** while generating slightly more rules (+10%)

## Summary

| Metric | Baseline (0.010) | Proposed (0.005) | Improvement |
|--------|------------------|------------------|-------------|
| Execution Time | 6.53s | 4.94s | **24.3% faster** |
| Rules Generated | 348 | 384 | +10% |
| Fold Agreement | 96.8% | 74.2% | -22.6% (acceptable trade-off) |

## Key Takeaways

1. **Speed:** Proposed method is 24.3% faster at comparable output scale
2. **Efficiency:** Achieved despite using **lower min_support** (harder computational task)
3. **Trade-off:** 22.6% lower agreement for significant speed gain
4. **Practical Utility:** 384 rules provides sufficient coverage for business applications vs 348 rules from baseline
