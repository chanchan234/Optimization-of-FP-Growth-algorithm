# Optimization-of-FP-Growth-algorithm

##  Key Result

**Random Forest FP-Growth achieves 91.23% prediction accuracy** — a **+15.5% improvement** over standard FP-Growth (75.74%), while producing **95% fewer rules** (187 vs 3,801).

| Method | Accuracy | Rules | Time |
|--------|----------|-------|------|
| Standard FP-Growth | 75.74% | 3,801 | 6.63s |
| **RF-FP-Growth (Adapted)** | **91.23%** | **187** | **9.26s** |

> **Bottom Line:** More accurate, cleaner results, slightly slower (40%). Worth the trade-off.


 ## The Gap: Frequency ≠ Reliability

FP-Growth efficiently identifies frequent patterns. However, frequency alone does not guarantee stability. A pattern that appears often in one dataset may:
- Occur by chance in the sample
- Not generalize to new data
- Lead to incorrect business decisions if trusted blindly



---
## Accuracy Validation Method
To evaluate rule quality, we use hold-out accuracy - a standard evaluation protocol in association rule mining 

**Protocol:**
1. Split Data: 80% training / 20% test (unseen data)

2. Discover Rules: Run FP-Growth on training set only

3. Test Predictions: Apply rules to test transactions

4. Calculate Accuracy: Correct Predictions / Total Predictions
   
## Optimization Strategy: Parallel Processing

Use parallel processing to minimize the speed trade-off of ensemble methods.


## Optimization Attempt 1: Multi-Fold Cross-Validation (Baseline)

**Idea:** Split data into folds → run FP-Growth on each fold → keep patterns that appear consistently across folds.



## Optimization Attempt 2: Random Forest-Inspired Ensemble (Proposed)

**Idea:** 
- 3 bootstrap samples (different transactions + features per tree)
- Parallel FP-Growth on each tree
- Ensemble voting: keep patterns with ≥2 votes

---

## 📊 Results (Tested at min_support = 0.005, 0.010, 0.015, 0.020)

### Best Performance: min_support = 0.005

| Method | Accuracy | Patterns | Rules | Time (s) |
|--------|----------|----------|-------|----------|
| Standard FP-Growth | 75.74% | 4,400 | 3,801 | 6.63 |
| CV-FP-Growth | 78.02% | 4,424 | 3,853 | 12.42 |
| **RF-FP-Growth (Adapted)** | **91.23%** | **1,302** | **187** | **9.26** |

> **Key Finding:** RF FP-Growth achieves **91.23% accuracy** at `min_support=0.005` - a **+15.5% improvement** over standard FP-Growth. 





### Limitation: Support Sensitivity

| min_support | RF-FP-Growth Accuracy | vs Standard | Verdict |
|-------------|----------------------|-------------|---------|
| **0.005** | **91.23%** | +15.5% | ✅ **RF wins** |
| 0.010 | 72.01% | -3% |  Standard wins |
| 0.015 | 56.33% | -14% |  Standard wins |
| 0.020 | 60.39% | -5% |  Standard wins |

> **Key Insight:** RF-FP-Growth's advantage is **support-specific**. It only achieves superior accuracy at `min_support=0.005`. At higher supports, standard FP-Growth performs better.
 ##  Conclusion
RF-FP-Growth is proven effective after optimization while cross validated method is not worth it.

**📁 More detailed results  available in `result.md`**

## Code Running Instructions


### Environment Requirements
- **Operating System**: Linux (Ubuntu/Debian) or WSL on Windows
- **Python**: Version 3.8 or higher
- **CPU**: 3 cores recommended (configurable: 1-8+ cores supported)
- **RAM**: 8GB recommended

### Required Files
Before running, ensure these files are in your working directory:
- `fp_parallel_main.py` - Main ensemble method
- `fp_growth_baseline.py` - Baseline comparison method
- `fp_growth_normal.py` - Baseline comparison method
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
python fp_efficiency.py --data_path "/mnt/your_path/online_retail_cleaned.csv"
python fp_parallel.py --data_path "/mnt/your_path/online_retail_cleaned.csv"
```
### For custom parameters for the standard and the cross validated method
```
python fp_efficiency.py \
    --data_path "/mnt/c/Users/85295/online_retail_cleaned.csv" \
    --min_item_support 0.002 \
    --support_thresholds 0.005 0.01 0.015 0.02 \
```
### for custom parameters for the RF-FP growth  
```
python fp_parallel.py \
    --data_path "/mnt/c/Users/85295/online_retail_cleaned.csv" \
    --min_item_support 0.002 \
    --fp_min_support_values 0.005 0.01 0.015 0.02 \
    --n_trees 3 \
    --sample_ratio 0.7 \
    --feature_ratio 0.6 \
    --min_tree_votes 2
```
# Technical details for parallel processing 
The chunking of data using  Python for parallel processing will not increase the speed due to  heavy lifting property of Python . Therefore, parallel execution is  not used for standard FP growth but can support the baseline and adapted method . 

