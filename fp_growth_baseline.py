

import time
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import fpgrowth, association_rules
from sklearn.model_selection import KFold
from multiprocessing import Pool
from collections import defaultdict
import warnings
import gc

warnings.filterwarnings('ignore')

RANDOM_SEED= 12345 

def load_and_prepare_data(file_path, min_item_support=0.002):
    """Load and prepare data"""
    print(f"Loading data from: {file_path}")
    df_raw = pd.read_csv(file_path)
    print(f"Raw data shape: {len(df_raw)} rows x {df_raw.shape[1]} columns")
    
    # Filter and clean
    df_raw = df_raw[~df_raw['InvoiceNo'].astype(str).str.startswith('C')]
    df_raw = df_raw.dropna(subset=['StockCode'])
    df_raw = df_raw[df_raw['Quantity'] > 0]
    print(f"After cleaning: {len(df_raw)} rows")
    
    # Create binary matrix
    print("Creating binary matrix...")
    basket = pd.crosstab(df_raw['InvoiceNo'], df_raw['StockCode'])
    basket = basket.astype(bool).astype(int)
    
    # Remove rare items
    item_frequency = basket.sum(axis=0) / len(basket)
    items_to_keep = item_frequency[item_frequency >= min_item_support].index
    removed_items = basket.shape[1] - len(items_to_keep)
    basket = basket[items_to_keep]
    basket = basket[basket.sum(axis=1) > 0]
    
    print(f"Final: {len(basket)} transactions x {basket.shape[1]} items")
    print(f"Removed {removed_items} rare items (<{min_item_support:.1%} frequency)")
    
    # Statistics
    sparsity = (basket == 0).sum().sum() / (basket.shape[0] * basket.shape[1])
    avg_items = basket.sum(axis=1).mean()
    print(f"Data Sparsity: {sparsity:.2%}")
    print(f"Average items per transaction: {avg_items:.2f}")
    print(f"Memory: {basket.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n")
    
    return basket

def run_fold_worker(args):
    """
    Worker function: Run FP-Growth on a single fold
    Args: (fold_idx, train_indices, df_full, min_support)
    """
    fold_idx, train_indices, df_full, min_support = args
    
    try:
        # train_indices are integer positions, use iloc
        train_df = df_full.iloc[train_indices]
        
        # Run FP-Growth
        freq = fpgrowth(train_df, min_support=min_support, use_colnames=True, max_len=4)
        
        if freq.empty:
            return fold_idx, {}, 0
        
        patterns_dict = {tuple(sorted(row['itemsets'])): row['support'] for _, row in freq.iterrows()}
        return fold_idx, patterns_dict, len(freq)
    
    except Exception as e:
        print(f"Fold {fold_idx} error: {e}")
        return fold_idx, {}, 0

def parallel_cross_validation(df, min_support=0.01, n_folds=3, min_fold_agreement=3, n_workers=3):
    """
    Parallel Cross-Validation FP-Growth
    """
    
    print(f"\n{'='*60}")
    print(f"PARALLEL {n_folds}-FOLD CROSS-VALIDATION FP-GROWTH")
    print(f"{'='*60}")
    print(f"Min Support: {min_support}")
    print(f"Folds: {n_folds}")
    print(f"Min Fold Agreement: {min_fold_agreement}/{n_folds}")
    print(f"Parallel Workers: {n_workers}")
    print(f"{'='*60}")
    
    start = time.time()
    
    # ========== FIX: Reset index to get integer positions ==========
    df_reset = df.reset_index(drop=True)
    n_rows = len(df_reset)
    
    # Setup K-Fold with integer positions
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    
    # Prepare arguments for parallel processing
    fold_args = []
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(range(n_rows)), 1):
        # train_idx is already integer positions
        fold_args.append((fold_idx, train_idx, df_reset, min_support))
    
    print(f"\nRunning {n_folds} folds in parallel on {min(n_folds, n_workers)} cores...")
    
    # Run all folds in parallel
    with Pool(processes=4 ) as pool:
        results = pool.map(run_fold_worker, fold_args)
    
    # Collect results from all folds
    fold_patterns = {}
    fold_supports = {}
    pattern_counts = {}
    
    for fold_idx, patterns_dict, n_patterns in results:
        fold_patterns[fold_idx] = set(patterns_dict.keys())
        fold_supports[fold_idx] = patterns_dict
        pattern_counts[fold_idx] = n_patterns
        print(f"  Fold {fold_idx}: {n_patterns} patterns found")
    
    # Aggregate patterns across folds
    pattern_fold_count = defaultdict(int)
    pattern_supports = defaultdict(list)
    
    for fold_idx, patterns in fold_patterns.items():
        for pattern in patterns:
            pattern_fold_count[pattern] += 1
            pattern_supports[pattern].append(fold_supports[fold_idx][pattern])
    
    # Calculate statistics
    total_unique_patterns = len(pattern_fold_count)
    
    # Check if any patterns found
    if total_unique_patterns == 0:
        print("\n" + "="*50)
        print("⚠️ NO PATTERNS FOUND")
        print("="*50)
        print("\nPossible solutions:")
        print("  1. Lower min_support (try 0.005, 0.01, or 0.015)")
        print("  2. Reduce n_folds to 3 (each fold gets more data)")
        print("  3. Lower min_item_support when loading data")
        print(f"\nCurrent min_support: {min_support}")
        
        elapsed = time.time() - start
        
        return {
            'patterns': pd.DataFrame(),
            'rules': pd.DataFrame(),
            'time': elapsed,
            'total_patterns_across_folds': 0,
            'patterns_in_all_folds': 0,
            'n_validated': 0,
            'n_rules': 0,
            'min_support': min_support,
            'n_folds': n_folds,
            'success': False
        }
    
    patterns_in_all_folds = sum(1 for count in pattern_fold_count.values() if count == n_folds)
    
    print(f"\n{'='*50}")
    print(f"VALIDATION RESULTS")
    print(f"{'='*50}")
    print(f"Total unique patterns across all folds: {total_unique_patterns}")
    print(f"Patterns in all {n_folds} folds: {patterns_in_all_folds} ({patterns_in_all_folds/total_unique_patterns*100:.1f}%)")
    
    # Filter patterns by fold agreement
    validated_patterns = []
    for pattern, fold_count in pattern_fold_count.items():
        if fold_count >= min_fold_agreement:
            avg_support = np.mean(pattern_supports[pattern])
            support_std = np.std(pattern_supports[pattern])
            support_cv = support_std / (avg_support + 1e-8)
            
            validated_patterns.append({
                'itemsets': list(pattern),
                'itemset_size': len(pattern),
                'support': avg_support,
                'support_std': support_std,
                'support_cv': support_cv,
                'fold_count': fold_count,
                'fold_agreement': fold_count / n_folds,
                'stability': 1 - min(1, support_cv)
            })
    
    validated_df = pd.DataFrame(validated_patterns).sort_values(['fold_agreement', 'support'], ascending=[False, False])
    
    print(f"\nPatterns in ≥{min_fold_agreement} folds: {len(validated_df)} ({len(validated_df)/total_unique_patterns*100:.1f}%)")
    
    if len(validated_df) > 0:
        print(f"Avg stability score: {validated_df['stability'].mean():.2%}")
        print(f"Avg fold agreement: {validated_df['fold_agreement'].mean():.1%}")
    
    # Generate rules from validated patterns
    rules = pd.DataFrame()
    if len(validated_df) >= 2:
        print("\nGenerating association rules from validated patterns...")
        try:
            rules_df = validated_df[['itemsets', 'support']].copy()
            rules_df['itemsets'] = rules_df['itemsets'].apply(frozenset)
            rules = association_rules(rules_df, metric="lift", min_threshold=1.5)
            
            if not rules.empty:
                rules = rules[rules['confidence'] >= 0.5]
                rules = rules.sort_values(['confidence','lift'], ascending=[False, False])
                print(f"  Rules generated: {len(rules)}")
        except Exception as e:
            print(f"  Rule generation error: {e}")
    
    elapsed = time.time() - start
    
    print(f"\n{'='*50}")
    print(f"FINAL RESULTS")
    print(f"{'='*50}")
    print(f"Validated patterns: {len(validated_df)}")
    print(f"Association rules: {len(rules)}")
    print(f"Total time: {elapsed:.2f}s")
    
    # Show top patterns if any
    if not validated_df.empty:
        print(f"\n{'='*50}")
        print(f"TOP 10 VALIDATED PATTERNS (by fold agreement & support)")
        print(f"{'='*50}")
        for i, row in validated_df.head(10).iterrows():
            items = ', '.join(row['itemsets'][:5])
            if len(row['itemsets']) > 5:
                items += f" +{len(row['itemsets'])-5}"
            print(f"\n{i+1}. {items}")
            print(f"   Support: {row['support']:.4f} | Stability: {row['stability']:.2%}")
            print(f"   Fold Agreement: {row['fold_agreement']:.0%} ({row['fold_count']}/{n_folds} folds)")
            print(f"   Support Std: {row['support_std']:.4f}")
    
    # Show top rules
    if not rules.empty:
        print(f"\n{'='*50}")
        print(f"TOP 10 VALIDATED RULES (by Lift)")
        print(f"{'='*50}")
        for i, (_, r) in enumerate(rules.head(10).iterrows(), 1):
            ant = ', '.join(list(r['antecedents'])[:3])
            con = ', '.join(list(r['consequents'])[:2])
            if len(r['antecedents']) > 3:
                ant += f" +{len(r['antecedents'])-3}"
            print(f"\n{i}. {ant} -> {con}")
            print(f"   Lift: {r['lift']:.2f} | Confidence: {r['confidence']:.2%} | Support: {r['support']:.4f}")
    
    return {
        'patterns': validated_df,
        'rules': rules,
        'time': elapsed,
        'total_patterns_across_folds': total_unique_patterns,
        'patterns_in_all_folds': patterns_in_all_folds,
        'n_validated': len(validated_df),
        'n_rules': len(rules),
        'min_support': min_support,
        'n_folds': n_folds,
        'fold_pattern_counts': pattern_counts,
        'success': True
    }

if __name__ == "__main__":
    print("="*60)
    print("PARALLEL 3-FOLD CROSS-VALIDATION FP-GROWTH")
    print("OPTIMIZED FOR 4-CORE CPU")
    print("="*60)
    
    print("="*60)
    
    # Configuration
    DATA_PATH = "/mnt/c/Users/85295/online_retail_cleaned.csv"
    
    # Load data with lower min_item_support
    df = load_and_prepare_data(DATA_PATH, min_item_support=0.002)
    
    # Test different support thresholds
    SUPPORT_THRESHOLDS = [0.005,0.01,0.015,0.02]
    
    all_results = []
    
    for min_sup in SUPPORT_THRESHOLDS:
        print("\n" + "🔍"*30)
        print(f"TESTING MIN SUPPORT = {min_sup}")
        print("🔍"*30)
        
        result = parallel_cross_validation(
            df=df,
            min_support=min_sup,
            n_folds=3,
            min_fold_agreement=2,  # 60% agreement (similar to Random Forest)
            n_workers=4  # Use all 4 cores
        )
        
        if result['success']:
            all_results.append(result)
            print(f"\n✅ Success! Found {result['n_validated']} validated patterns")
        else:
            print(f"\n❌ No patterns found at min_support={min_sup}")
        
        gc.collect()
    
    # Summary
    if all_results:
        print("\n" + "="*60)
        print("SUMMARY OF SUCCESSFUL RUNS")
        print("="*60)
        print(f"\n{'Min Support':<12} {'Validated Patterns':<20} {'Rules':<10} {'Time (s)':<10}")
        print("-"*55)
        for r in all_results:
            print(f"{r['min_support']:<12.3f} {r['n_validated']:<20} {r['n_rules']:<10} {r['time']:<10.2f}")
    
    print("\n" + "="*60)
    print("✅ COMPLETE")
    print("="*60)
    
    print("\n" + "="*60)
    print("PERFORMANCE NOTE")
    print("="*60)
    