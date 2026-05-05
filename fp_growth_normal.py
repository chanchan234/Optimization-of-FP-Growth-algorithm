import time
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import fpgrowth, association_rules
import warnings
import gc

warnings.filterwarnings('ignore')

RANDOM_SEED = 12345

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

def standard_fpgrowth(df, min_support=0.01, max_len=4):
    """
    STANDARD FP-GROWTH (No Cross-Validation, No Parallel Processing)
    Single run on full dataset
    """
    
    print(f"\n{'='*60}")
    print(f"STANDARD FP-GROWTH (No CV, No Parallel)")
    print(f"{'='*60}")
    print(f"Min Support: {min_support}")
    print(f"Max Pattern Length: {max_len}")
    print(f"{'='*60}")
    
    start = time.time()
    
    # Run FP-Growth once on full dataset
    freq = fpgrowth(df, min_support=min_support, use_colnames=True, max_len=max_len)
    
    if freq.empty:
        print(f"  No patterns found at min_support={min_support}")
        elapsed = time.time() - start
        return {
            'patterns': pd.DataFrame(),
            'rules': pd.DataFrame(),
            'time': elapsed,
            'n_patterns': 0,
            'n_rules': 0,
            'min_support': min_support,
            'success': False
        }
    
    print(f"  Patterns found: {len(freq)}")
    
    # Convert patterns to readable format
    patterns_list = []
    for _, row in freq.iterrows():
        patterns_list.append({
            'itemsets': list(row['itemsets']),
            'itemset_size': len(row['itemsets']),
            'support': row['support']
        })
    
    patterns_df = pd.DataFrame(patterns_list).sort_values('support', ascending=False)
    
    # Generate association rules
    rules = pd.DataFrame()
    if len(freq) >= 2:
        print("\nGenerating association rules...")
        try:
            rules = association_rules(freq, metric="lift", min_threshold=1.5)
            if not rules.empty:
                rules = rules[rules['confidence'] >= 0.5]
                rules = rules.sort_values(['confidence', 'lift'], ascending=[False, False])
                print(f"  Rules generated: {len(rules)}")
        except Exception as e:
            print(f"  Rule generation error: {e}")
    
    elapsed = time.time() - start
    
    print(f"\n{'='*50}")
    print(f"RESULTS")
    print(f"{'='*50}")
    print(f"Total patterns: {len(patterns_df)}")
    print(f"Association rules: {len(rules)}")
    print(f"Total time: {elapsed:.2f}s")
    
    # Show top patterns
    if not patterns_df.empty:
        print(f"\n{'='*50}")
        print(f"TOP 10 PATTERNS (by support)")
        print(f"{'='*50}")
        for i, row in patterns_df.head(10).iterrows():
            items = ', '.join(row['itemsets'][:5])
            if len(row['itemsets']) > 5:
                items += f" +{len(row['itemsets'])-5}"
            print(f"\n{i+1}. {items}")
            print(f"   Support: {row['support']:.4f} | Size: {row['itemset_size']}")
    
    # Show top rules
    if not rules.empty:
        print(f"\n{'='*50}")
        print(f"TOP 10 RULES (by Lift)")
        print(f"{'='*50}")
        for i, (_, r) in enumerate(rules.head(10).iterrows(), 1):
            ant = ', '.join(list(r['antecedents'])[:3])
            con = ', '.join(list(r['consequents'])[:2])
            if len(r['antecedents']) > 3:
                ant += f" +{len(r['antecedents'])-3}"
            print(f"\n{i}. {ant} -> {con}")
            print(f"   Lift: {r['lift']:.2f} | Confidence: {r['confidence']:.2%} | Support: {r['support']:.4f}")
    
    return {
        'patterns': patterns_df,
        'rules': rules,
        'time': elapsed,
        'n_patterns': len(patterns_df),
        'n_rules': len(rules),
        'min_support': min_support,
        'success': True
    }

if __name__ == "__main__":
    print("="*60)
    print("STANDARD FP-GROWTH")
    print("No Cross-Validation | No Parallel Processing | Single Core")
    print("="*60)
    
    # Configuration - SAME as baseline
    DATA_PATH = "/mnt/c/Users/85295/online_retail_cleaned.csv"
    
    # Load data - SAME as baseline
    df = load_and_prepare_data(DATA_PATH, min_item_support=0.002)
    
    # Test different support thresholds - SAME 4 values as baseline
    SUPPORT_THRESHOLDS = [0.005, 0.010, 0.015, 0.020]
    
    all_results = []
    
    for min_sup in SUPPORT_THRESHOLDS:
        print("\n" + "🔍"*30)
        print(f"TESTING MIN SUPPORT = {min_sup}")
        print("🔍"*30)
        
        result = standard_fpgrowth(
            df=df,
            min_support=min_sup,
            max_len=4
        )
        
        if result['success']:
            all_results.append(result)
            print(f"\n✅ Success! Found {result['n_patterns']} patterns in {result['time']:.2f}s")
        else:
            print(f"\n❌ No patterns found at min_support={min_sup}")
        
        gc.collect()
    
    # Summary - SAME format as baseline
    if all_results:
        print("\n" + "="*60)
        print("SUMMARY OF STANDARD FP-GROWTH (No CV, No Parallel)")
        print("="*60)
        print(f"\n{'Min Support':<12} {'Patterns':<15} {'Rules':<10} {'Time (s)':<10}")
        print("-"*50)
        for r in all_results:
            print(f"{r['min_support']:<12.3f} {r['n_patterns']:<15} {r['n_rules']:<10} {r['time']:<10.2f}")
    
    print("\n" + "="*60)
    print("✅ COMPLETE")
    print("="*60)