import time
import random
import pandas as pd
import numpy as np
from collections import defaultdict
from mlxtend.frequent_patterns import fpgrowth, association_rules
from multiprocessing import Pool
from tqdm import tqdm
import warnings
import gc

warnings.filterwarnings('ignore')

MASTER_SEED = 12345
np.random.seed(MASTER_SEED)
random.seed(MASTER_SEED)

RANDOM_SEEDS = {
    'data_loading': MASTER_SEED,
    'bootstrap': MASTER_SEED + 1,
    'feature_sampling': MASTER_SEED + 2,
    'fpgrowth': MASTER_SEED + 3,
    'voting': MASTER_SEED + 4,
}

def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)

def get_worker_seed(base_seed, tree_idx):
    return base_seed + tree_idx * 100 + RANDOM_SEEDS['bootstrap']

def load_and_prepare_data(file_path, min_item_support=0.005, random_seed=None):
    if random_seed:
        set_all_seeds(random_seed)
    
    print(f"Loading data from: {file_path}")
    df_raw = pd.read_csv(file_path)
    print(f"Raw data shape: {len(df_raw)} rows x {df_raw.shape[1]} columns")
    print(f"Columns: {list(df_raw.columns)}")
    print("\nFirst 3 rows:")
    print(df_raw.head(3))
    
    original_rows = len(df_raw)
    df_raw = df_raw[~df_raw['InvoiceNo'].astype(str).str.startswith('C')]
    print(f"\nRemoved {original_rows - len(df_raw)} cancelled transactions")
    
    df_raw = df_raw.dropna(subset=['StockCode'])
    
    df_raw = df_raw[df_raw['Quantity'] > 0]
    print(f"After removing returns: {len(df_raw)} rows")
    
    print("\nCreating binary matrix (transactions x items)...")
    
    basket = pd.crosstab(df_raw['InvoiceNo'], df_raw['StockCode'])
    
    basket = basket.astype(bool).astype(int)
    
    print(f"Binary matrix created: {len(basket)} transactions x {basket.shape[1]} items")
    
    item_frequency = basket.sum(axis=0) / len(basket)
    items_to_keep = item_frequency[item_frequency >= min_item_support].index
    removed_items = basket.shape[1] - len(items_to_keep)
    basket = basket[items_to_keep]
    print(f"Removed {removed_items} rare items (<{min_item_support:.1%} frequency)")
    print(f"Items remaining: {basket.shape[1]}")
    
    basket = basket[basket.sum(axis=1) > 0]
    print(f"Transactions remaining: {len(basket)}")
    
    sparsity = (basket == 0).sum().sum() / (basket.shape[0] * basket.shape[1])
    avg_items_per_trans = basket.sum(axis=1).mean()
    print(f"\nData Statistics:")
    print(f"  Data Sparsity: {sparsity:.2%}")
    print(f"  Average items per transaction: {avg_items_per_trans:.2f}")
    print(f"  Memory usage: {basket.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    return basket

def worker(args):
    idx, df_shared, min_sup, sample_ratio, feature_ratio, n_rows, n_cols, base_seed = args
    
    worker_seed = get_worker_seed(base_seed, idx)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    
    try:
        row_idx = random.choices(range(n_rows), k=int(n_rows * sample_ratio))
        
        n_feat = max(1, int(n_cols * feature_ratio))
        feat_idx = random.sample(range(n_cols), n_feat)
        
        df_sample = df_shared.iloc[row_idx, feat_idx]
        df_sample = df_sample.loc[:, df_sample.any(axis=0)]
        
        if df_sample.shape[1] == 0:
            return idx, pd.DataFrame(), {'tree_id': idx, 'n_items': 0, 'n_trans': len(row_idx)}
        
        max_len = min(4, df_sample.shape[1])
        freq = fpgrowth(df_sample, min_support=min_sup, use_colnames=True, max_len=max_len)
        
        tree_info = {
            'tree_id': idx,
            'n_items': df_sample.shape[1],
            'n_trans': len(row_idx),
            'n_patterns': len(freq)
        }
        
        return idx, freq, tree_info
        
    except Exception as e:
        print(f"Worker {idx} error: {e}")
        return idx, pd.DataFrame(), {'tree_id': idx, 'error': str(e)}

def run_random_forest_fpgrowth(df, min_support, n_trees=3, sample_ratio=0.7, 
                               feature_ratio=0.7, min_tree_votes=2, 
                               random_seed=MASTER_SEED):
    if random_seed:
        set_all_seeds(random_seed)
    
    print(f"\n{'='*60}")
    print(f"RANDOM FOREST FP-GROWTH")
    print(f"{'='*60}")
    print(f"Configuration:")
    print(f"  Min Support: {min_support}")
    print(f"  Trees: {n_trees}")
    print(f"  Row Sample: {sample_ratio:.0%}")
    print(f"  Feature Sample: {feature_ratio:.0%}")
    print(f"  Min Tree Votes: {min_tree_votes}")
    print(f"{'='*60}")
    
    start = time.time()
    
    n_rows, n_cols = len(df), df.shape[1]
    
    base_seed = random_seed if random_seed else MASTER_SEED
    args = [(i, df, min_support, sample_ratio, feature_ratio, n_rows, n_cols, base_seed) 
            for i in range(n_trees)]
    
    with Pool(processes=3 ) as pool:
        results = list(tqdm(pool.imap(worker, args), total=n_trees, desc="Growing Trees"))
    
    tree_infos = []
    tree_patterns = []
    
    for idx, freq, info in results:
        tree_infos.append(info)
        if not freq.empty:
            tree_patterns.append(freq)
    
    print(f"\nForest Statistics:")
    print(f"  Trees with patterns: {len(tree_patterns)}/{n_trees}")
    for info in tree_infos:
        if 'error' not in info:
            print(f"    Tree {info['tree_id']}: {info['n_trans']} transactions, "
                  f"{info['n_items']} items, {info['n_patterns']} patterns")
    
    patterns = defaultdict(list)
    for freq in tree_patterns:
        if not freq.empty:
            for _, row in freq.iterrows():
                patterns[frozenset(row['itemsets'])].append(row['support'])
    
    if not patterns:
        print("\nNo patterns found in any tree")
        return None
    
    merged = []
    for p, supps in patterns.items():
        if len(supps) >= min_tree_votes:
            merged.append({
                'itemsets': list(p), 
                'support': np.mean(supps),
                'support_std': np.std(supps) if len(supps) > 1 else 0,
                'trees': len(supps),
                'agreement': len(supps) / n_trees,
                'vote_ratio': len(supps) / n_trees
            })
    
    if not merged:
        print("\nNo patterns survived voting threshold")
        return None
    
    merged_df = pd.DataFrame(merged).sort_values(['agreement', 'support'], ascending=False)
    
    print(f"\nEnsemble Pattern Statistics:")
    print(f"  Total unique patterns across all trees: {len(patterns)}")
    print(f"  Patterns with {min_tree_votes}+ votes: {len(merged_df)}")
    if len(merged_df) > 0:
        print(f"  Average items per pattern: {merged_df['itemsets'].str.len().mean():.2f}")
        print(f"  Average support: {merged_df['support'].mean():.4f}")
        print(f"  Average agreement: {merged_df['agreement'].mean():.1%}")
        
        print(f"\n  Top 5 Most Consistent Patterns:")
        for i, row in merged_df.head(5).iterrows():
            items = ', '.join(row['itemsets'][:5])
            if len(row['itemsets']) > 5:
                items += f" +{len(row['itemsets'])-5} more"
            print(f"    {items}")
            print(f"      Support: {row['support']:.4f}, Agreement: {row['agreement']:.1%} "
                  f"(in {row['trees']}/{n_trees} trees)")
    
    rules = pd.DataFrame()
    if len(merged_df) >= 2:
        rules_df = merged_df[['itemsets', 'support']].copy()
        rules_df['itemsets'] = rules_df['itemsets'].apply(frozenset)
        try:
            rules = association_rules(rules_df, metric="confidence", min_threshold=0.5)
            if not rules.empty:
                
                rules = rules.sort_values(['confidence', 'lift'], ascending=[False, False])
        except Exception as e:
            print(f"Rule generation error: {e}")
    
    elapsed = time.time() - start
    
    print(f"\nFinal Results:")
    print(f"  Reliable Itemsets: {len(merged_df)}")
    print(f"  Association Rules: {len(rules)}")
    print(f"  Total Time: {elapsed:.2f}s")
    
    if not rules.empty:
        print(f"\nTop 10 Rules (by Lift):")
        for i, (_, r) in enumerate(rules.head(10).iterrows(), 1):
            ant = ', '.join(list(r['antecedents'])[:3])
            con = ', '.join(list(r['consequents'])[:2])
            if len(r['antecedents']) > 3:
                ant += f" +{len(r['antecedents'])-3}"
            print(f"\n  {i}. {ant} -> {con}")
            print(f"     Lift: {r['lift']:.2f} | Confidence: {r['confidence']:.2%} | "
                  f"Support: {r['support']:.4f}")
    
    return {
        'min_sup': min_support,
        'n_trees': n_trees,
        'itemsets': len(merged_df),
        'rules': len(rules),
        'time': elapsed,
        'unique_patterns': len(patterns),
        'trees_with_patterns': len(tree_patterns),
        'avg_agreement': merged_df['agreement'].mean() if len(merged_df) > 0 else 0
    }

if __name__ == "__main__":
    print("="*60)
    print("RANDOM FOREST FP-GROWTH FOR RETAIL DATA")
    print("="*60)
    print(f"Master Random Seed: {MASTER_SEED}")
    print("="*60)
    
    DATA_PATH = "/mnt/c/Users/85295/online_retail_cleaned.csv"
    
    RF_CONFIG = {
        'n_trees': 3,
        'sample_ratio': 0.7,
        'feature_ratio': 0.6,
        'min_tree_votes': 2,
        'random_seed': MASTER_SEED
    }
    
    SUPPORT_THRESHOLDS = [0.005, 0.01, 0.015,0.02]
    
    df = load_and_prepare_data(
        DATA_PATH, 
        min_item_support=0.005,
        random_seed=RANDOM_SEEDS['data_loading']
    )
    
    results = []
    for min_sup in SUPPORT_THRESHOLDS:
        result = run_random_forest_fpgrowth(
            df=df,
            min_support=min_sup,
            **RF_CONFIG
        )
        if result:
            results.append(result)
        
        gc.collect()
    
    if results:
        print("\n" + "="*60)
        print("FINAL SUMMARY")
        print("="*60)
        print(f"{'Support':<10} {'Itemsets':<12} {'Rules':<8} {'Trees w/Pat':<12} {'Agreement':<10} {'Time(s)':<8}")
        print("-"*60)
        for r in results:
            print(f"{r['min_sup']:<10.3f} {r['itemsets']:<12} {r['rules']:<8} "
                  f"{r['trees_with_patterns']}/{RF_CONFIG['n_trees']:<9} "
                  f"{r['avg_agreement']:.1%}      {r['time']:<8.2f}")
    
    print("\nClearing final cache...")
    del df
    gc.collect()
    
    print("\nComplete!")