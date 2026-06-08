# Optimization-of-FP-Growth-algorithm

## Business Problem

An e-commerce company uses FP-Growth to find product associations for recommendations. But "frequent" doesn't mean "reliable" – patterns that look good in historical data often fail on new customers, causing irrelevant recommendations and wasted marketing time.

## Key Result

**Random Forest FP-Growth achieves 91.23% prediction accuracy** — a **+15.5% improvement** over standard FP-Growth (75.74%), while producing **95% fewer rules** (187 vs 3,801).

| Method | Accuracy | Rules | Time |
|--------|----------|-------|------|
| Standard FP-Growth | 75.74% | 3,801 | 6.63s |
| **RF-FP-Growth (Adapted)** | **91.23%** | **187** | **9.26s** |

> **Bottom Line:** More accurate recommendations, cleaner results for business teams to review, 40% slower – worth the trade-off for batch analytics.

## The Gap: Frequency ≠ Reliability

FP-Growth efficiently identifies frequent patterns. However, frequency alone does not guarantee reliability. A pattern that appears often in one dataset may:
- Occur by chance in the sample
- Not generalize to new customer data
- Lead to bad recommendations if trusted blindly

**This project solves that gap.**

---

  
## Optimization Strategy: Parallel Processing

Use parallel processing to minimize the speed trade-off of ensemble methods.


## Optimization Attempt 1: Multi-Fold Cross-Validation 

**Idea:** Split data into folds → run FP-Growth on each fold → keep patterns that appear consistently across folds.



## Optimization Attempt 2: Random Forest-Inspired Ensemble (Proposed)

**Idea:** 
- 3 bootstrap samples (different transactions + features per tree)
- Parallel FP-Growth on each tree
- Ensemble voting: keep patterns with ≥2 votes

---
---
## Accuracy Validation Method
To evaluate rule quality, we use hold-out accuracy - a standard evaluation protocol in association rule mining 

**Protocol:**
1. Split Data: 80% training / 20% test (unseen data)

2. Discover Rules: Run FP-Growth on training set only

3. Test Predictions: Apply rules to test transactions

4. Calculate Accuracy: Correct Predictions / Total Predictions
5. 
## 📊 Results (Tested at min_support = 0.005, 0.010, 0.015, 0.020)

### Best Performance: min_support = 0.005

| Method | Accuracy | Patterns | Rules | Time (s) |
|--------|----------|----------|-------|----------|
| Standard FP-Growth | 75.74% | 4,400 | 3,801 | 6.63 |
| CV-FP-Growth | 78.02% | 4,424 | 3,853 | 12.42 |
| **RF-FP-Growth (Adapted)** | **91.23%** | **1,302** | **187** | **9.26** |

> **Key Finding:** RF FP-Growth achieves **91.23% accuracy** at `min_support=0.005` - a **+15.5% improvement** over standard FP-Growth. 





###  Key Limitation: Support Sensitivity

| min_support | RF-FP-Growth Accuracy | vs Standard | Verdict |
|-------------|----------------------|-------------|---------|
| **0.005** | **91.23%** | +15.5% | ✅ **RF wins** |
| 0.010 | 72.01% | -3% |  Standard wins |
| 0.015 | 56.33% | -14% |  Standard wins |
| 0.020 | 60.39% | -5% |  Standard wins |

> **Key Insight:** RF-FP-Growth's advantage is **support-specific**. It only achieves superior accuracy at `min_support=0.005`. At higher supports, standard FP-Growth performs better.
### Additional Limitations

| Limitation | Why It Matters |
|:---|:---|
| **Single dataset only** | Results on UCI Online Retail may not generalize to other domains (e.g., sparse clickstreams, dense supermarket baskets) |
| **CV-FP-Growth tested only at supports ≥0.005** | Cross-validation may perform better at lower supports (<0.005) – not tested |
| **No Apriori baseline** | Standard FP-Growth comparison only; Apriori may perform differently |
| **No lift/conviction metrics** | Accuracy alone doesn't capture rule interestingness |

## Conclusion

Within the scope of this study (UCI Online Retail, `min_support ≥ 0.005`):
- RF-FP-Growth wins at `min_support=0.005` (+15.5% accuracy, 95% fewer rules)
- Standard FP-Growth wins at higher supports (0.010–0.020)
- CV-FP-Growth showed no meaningful improvement

**Generalization beyond this dataset requires future validation.**
**📁 More detailed results  available in `result.md`**

## Code Running Instructions


### Environment Requirements
- **Operating System**: Linux (Ubuntu/Debian) or WSL on Windows
- **Python**: Version 3.8 or higher
- **CPU**: 3 cores recommended (configurable: 1-8+ cores supported)
- **RAM**: 8GB recommended

### Required Files
Before running, ensure these files are in your working directory:
- `fp_rf_parallel.py` - Main ensemble method
- `fp_stardard_cv.py` - Comparsion of the cross validated and standard method 
-  `requirements.txt` - Python dependencies
- `online_retail_cleaned.csv` - Preprocessed dataset (included in the master branch)



### Step-by-Step Execution
```bash
git clone https://github.com/yourname/ensemble-fp-growth.git
cd ensemble-fp-growth
python3 -m venv fp_env && source fp_env/bin/activate
pip install -r requirements.txt
```

```Run file with bash 
python fp_standard_cv.py --data_path "/mnt/your_path/online_retail_cleaned.csv"
python fp_rf_parallel.py --data_path "/mnt/your_path/online_retail_cleaned.csv"
```
### For custom parameters for the standard and the cross validated method
```
python fp_stardard_cv.py \
    --data_path "/mnt/c/Users/85295/online_retail_cleaned.csv" \
    --min_item_support 0.002 \
    --support_thresholds 0.005 0.01 0.015 0.02 \
```
### For custom parameters for the RF-FP growth  
```
python fp_rf_parallel.py \
    --data_path "/mnt/c/Users/85295/online_retail_cleaned.csv" \
    --min_item_support 0.002 \
    --fp_min_support_values 0.005 0.01 0.015 0.02 \
    --n_trees 3 \
    --sample_ratio 0.7 \
    --feature_ratio 0.6 \
    --min_tree_votes 2
```
# Technical details for parallel processing 
RF-FP-Growth and cross vliadated method  uses `multiprocessing.Pool` to build all 3 FP-Growth trees simultaneously across CPU cores. However, standard fp growth tree can only support sequential execution.
## References

Liu, G., Zhang, H., & Wong, L. (2011). Controlling false positives in association rule mining. *Proceedings of the VLDB Endowment*, *5*(2), 145–156. https://arxiv.org/abs/1110.6652

Scheffer, T. (2001). Finding association rules that trade support optimally against confidence. In *Proceedings of the 5th European Conference on Principles of Data Mining and Knowledge Discovery (PKDD)* (pp. 424–435). Springer. https://doi.org/10.1007/3-540-44794-6_34

