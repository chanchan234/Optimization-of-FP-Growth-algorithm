# Optimization-of-FP-Growth-algorithm
 **The Gap** : Frequency ≠ Reliability


 
FP-Growth efficiently identifies frequent patterns. However, frequency alone does not guarantee stability. A pattern that appears often in one dataset may: 



- Occur by chance in the sample
- Not generalize to new data
- Lead to incorrect business decisions if trusted blindly

   
## The Optimising strategy : Use parallel processing to minimise the trade off of lower speed

## Optimization Attempt 1: Multi-Fold Cross-Validation (Baseline)

**Idea:** Split data into folds → run FP-Growth on each fold → keep patterns that appear consistently across folds.

**Note:** Cross-validation is standard in ML, but its application to FP-Growth pattern validation has not been previously formalized. 

## Optimization Attempt 2: Random Forest-Inspired Ensemble (Adapted Method)

**Insight:** Data sampling increases processing speed.

**Idea:** Bootstrap samples (3 trees) → parallel FP-Growth → ensemble voting (keep patterns with cross-tree agreement)
## Key Results (at min_support=0.010)

| Method | Rules | Time | Validation |
|--------|-------|------|------------|
| Standard | 338 | 2.46s |  0% |
| Baseline | 348 | 6.53s | 96.8% |
| **Adapted ** | **85** | **4.09s** |  75.8% |

**What this proves:**
- Adapted method filters out 75% of rules (338 → 85) that standard keeps
- Adapted method is 40% faster than baseline with meaningful validation
- Adapted are slower than standard - that's the **cost of validation**


**Bottom line:** Adapted method gives you validated rules at practical speed

-


**The full result will be on result.md**

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
- `online_retail_cleaned.csv` - Preprocessed dataset (included)
- `requirements.txt` - Python dependencies


### Step-by-Step Execution
```bash
git clone https://github.com/yourname/ensemble-fp-growth.git
cd ensemble-fp-growth
python3 -m venv fp_env && source fp_env/bin/activate
pip install -r requirements.txt
```

###**Update data path** in both `.py` files:
python
DATA_PATH = "/path/to/online_retail_cleaned.csv"


**Run:**
```bash
python fp_parallel_main.py      # Main ensemble method
python baseline_fp_growth.py   # Baseline comparison
python normal_fp_growth.py    # standard fp growth
```

## Configuration

**Ensemble params** (`fp_parallel_main.py`):
```python
RF_CONFIG = {
    'n_trees': 3,
    'sample_ratio': 0.7,
    'feature_ratio': 0.6,
    'random_seed': 12345   
}
```
# Technical details for parallel processing 
The chunking of data using  Python for parallel processing will not increase the speed due to  heavy lifting property of Python . Therefore, parallel execution is  not used for standard FP growth but can support the baseline and adapted method . 

