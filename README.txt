
Optimising the FP growth algorithm for market basket analysis 
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](LICENSE)

## The Problem: Standard FP-Growth Has No Quality Control
Standard FP-Growth mines all frequent patterns from data - but accepts every pattern regardless of whether it would reappear on new data. This  creates a fundamental problem for business decision-making:

 1.High false positive rate - Patterns that occur by chance look identical to genuine patterns
 2.No validation mechanism - No way to know which rules are stable vs. statistical noise

## The Optimising strategy : Use parallel processing to minimise the trade off of lower speed

Optimization Attempt 1: Multi-Fold Cross-Validation (Baseline)
The idea: Split data into multiple folds, run FP-Growth on each fold, keep only patterns that appear consistently across folds.

**Note:** While cross-validation is standard in machine learning, its application to FP-Growth pattern validation has not been previously formalized. I implement it here as our baseline validation method.

Optimization Attempt 2: Random Forest-Inspired Ensemble (Adapted Method)
Insights: the sampling of data is expected to increase the speed of the process 
The idea: Create multiple bootstrap samples of the data (3 trees) , run  FP- Growth on the sample in parallel . Finally, apply ensemble voting(keep only patterns that appear consistently across trees)


## Key Results (at min_support=0.010)

| Method | Rules | Time | Validation |
|--------|-------|------|------------|
| Standard | 338 | 2.46s |  0% |
| Baseline | 348 | 6.53s | 96.8% |
| **Ours** | **85** | **4.09s** |  75.8% |

**What this proves:**
- Our method filters out 75% of rules (338 → 85) that standard keeps
- Our method is 40% faster than baseline with meaningful validation
- Yes, we are slower than standard - that's the **cost of validation**


**Bottom line:** Standard = fast but no trust. Baseline = trust but slow. **Ours = trustworthy + fast enough for daily business.**


**The full result will be on full_result.txt**

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

**Update data path** in both `.py` files:
python
DATA_PATH = "/path/to/online_retail_cleaned.csv"


**Run:**
```bash
python fp_parallel_main.py      # Main ensemble method
python baseline_fp_growth.py    # Baseline comparison
```

## Configuration

**CPU cores** (adjustable in both scripts):
```python
N_JOBS = 3   (optimised performance )

```
Remarks:  the chunking of data using  python for parallel processing  will not increase the speed as Python's heavy lifting property. Therefore, parallel execution s also not used for standard FP growth. 

**Ensemble params** (`fp_parallel_main.py`):
```python
RF_CONFIG = {
    'n_trees': 3,
    'sample_ratio': 0.7,
    'feature_ratio': 0.6,
    'random_seed': 12345   
}
```







