import time
import random
import pandas as pd
import numpy as np
from collections import defaultdict
from mlxtend.frequent_patterns import fpgrowth, association_rules
from sklearn.model_selection import train_test_split
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

def load_and_prepare_data(file_path, min_item_support=0.002, random_seed=None):  # CHANGED: default to 0.002
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
    
    with Pool(processes=3) as pool:
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
        'avg_agreement': merged_df['agreement'].mean() if len(merged_df) > 0 else 0,
        'rules_df': rules,
        'merged_df': merged_df
    }


def evaluate_predictive_power(rf_result, df_full, test_size=0.2, random_seed=MASTER_SEED):
    """Evaluate predictive power of Random Forest FP-Growth on hold-out test set"""
    
    print("\n" + "="*60)
    print("PREDICTIVE POWER EVALUATION")
    print("="*60)
    
    train_df, test_df = train_test_split(
        df_full, 
        test_size=test_size, 
        random_state=random_seed,
        shuffle=True
    )
    
    print(f"Training set: {len(train_df)} transactions")
    print(f"Test set: {len(test_df)} transactions")
    
    print("\nRe-training Random Forest FP-Growth on training set...")
    rf_train_result = run_random_forest_fpgrowth(
        train_df, 
        min_support=rf_result['min_sup'],
        n_trees=rf_result['n_trees'],
        sample_ratio=0.7,
        feature_ratio=0.7,
        min_tree_votes=2,
        random_seed=random_seed
    )
    
    if rf_train_result is None or rf_train_result['rules_df'].empty:
        print("❌ No rules generated from training data")
        return None
    
    def calculate_prediction_accuracy(rules_df, test_data):
        """Calculate how accurately rules predict on test data"""
        if rules_df.empty:
            return 0.0, 0, 0, []
        
        hits = 0
        total_predictions = 0
        rule_stats = []
        
        test_sample = test_data
        if len(test_data) > 1000:
            test_sample = test_data.sample(1000, random_state=random_seed)
        
        for _, rule in rules_df.head(100).iterrows():
            antecedents = set(rule['antecedents'])
            consequents = set(rule['consequents'])
            
            rule_hits = 0
            rule_total = 0
            
            for idx in test_sample.index:
                transaction = test_sample.loc[idx]
                trans_items = set(transaction[transaction == 1].index)
                
                if antecedents.issubset(trans_items):
                    rule_total += 1
                    total_predictions += 1
                    if consequents.issubset(trans_items):
                        rule_hits += 1
                        hits += 1
            
            if rule_total > 0:
                rule_stats.append({
                    'rule': str(list(antecedents)[:3]) + '→' + str(list(consequents)[:2]),
                    'accuracy': rule_hits / rule_total,
                    'support': rule['support'],
                    'confidence': rule['confidence'],
                    'lift': rule['lift']
                })
        
        overall_accuracy = hits / total_predictions if total_predictions > 0 else 0.0
        return overall_accuracy, hits, total_predictions, rule_stats
    
    accuracy, hits, total, rule_stats = calculate_prediction_accuracy(
        rf_train_result['rules_df'], test_df
    )
    
    print("\n" + "="*60)
    print("PREDICTIVE POWER RESULTS")
    print("="*60)
    
    print(f"\n📊 Random Forest FP-Growth:")
    print(f"   Prediction Accuracy: {accuracy:.2%} ({hits}/{total} correct predictions)")
    print(f"   Rules evaluated: {len(rule_stats)}")
    
    if len(rule_stats) > 0:
        rule_stats_df = pd.DataFrame(rule_stats).sort_values('accuracy', ascending=False)
        print(f"\n🏆 Top 5 Most Accurate Rules:")
        for i, row in rule_stats_df.head(5).iterrows():
            print(f"   {i+1}. {row['rule']}")
            print(f"      Accuracy: {row['accuracy']:.2%} | Lift: {row['lift']:.2f} | Conf: {row['confidence']:.2%}")
    
    return {
        'accuracy': accuracy,
        'hits': hits,
        'total': total,
        'rule_stats': pd.DataFrame(rule_stats) if rule_stats else pd.DataFrame(),
        'train_patterns': rf_train_result['itemsets'],
        'train_rules': rf_train_result['rules']
    }


if __name__ == "__main__":
    import argparse
    
    # Parse command-line arguments (with sensible defaults)
    parser = argparse.ArgumentParser(description='RANDOM FOREST FP-GROWTH with Predictive Power Evaluation')
    parser.add_argument('--data_path', type=str, required=True,
                       
                        help='Path to the input CSV file')
    parser.add_argument('--min_item_support', type=float, default=0.002,
                        help='Minimum item support for data filtering (default: 0.002)')
    parser.add_argument('--fp_min_support_values', type=float, nargs='+', 
                        default=[0.005, 0.01, 0.015, 0.02],
                        help='FP-Growth min_support values to test')
    parser.add_argument('--n_trees', type=int, default=3,
                        help='Number of trees in random forest (default: 3)')
    parser.add_argument('--sample_ratio', type=float, default=0.7,
                        help='Row sampling ratio (default: 0.7)')
    parser.add_argument('--feature_ratio', type=float, default=0.6,
                        help='Feature sampling ratio (default: 0.6)')
    parser.add_argument('--min_tree_votes', type=int, default=2,
                        help='Minimum votes to keep pattern (default: 2)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("RANDOM FOREST FP-GROWTH FOR RETAIL DATA")
    print("WITH PREDICTIVE POWER EVALUATION")
    print("="*60)
    print(f"Master Random Seed: {MASTER_SEED}")
    print("="*60)
    
    DATA_PATH = args.data_path
    RF_CONFIG = {
        'n_trees': args.n_trees,
        'sample_ratio': args.sample_ratio,
        'feature_ratio': args.feature_ratio,
        'min_tree_votes': args.min_tree_votes,
        'random_seed': MASTER_SEED
    }
    
    FIXED_MIN_ITEM_SUPPORT = args.min_item_support
    FP_MIN_SUPPORT_VALUES = args.fp_min_support_values
    
    all_results = []
    all_predictive_results = []
    
    # Load data
    print("\n" + "█"*80)
    print(f"LOADING DATA WITH min_item_support = {FIXED_MIN_ITEM_SUPPORT}")
    print("█"*80)
    
    df = load_and_prepare_data(
        DATA_PATH, 
        min_item_support=FIXED_MIN_ITEM_SUPPORT,
        random_seed=RANDOM_SEEDS['data_loading']
    )
    
    # Test different FP-Growth min_support values
    for fp_sup in FP_MIN_SUPPORT_VALUES:
        print("\n" + "🔍"*30)
        print(f"TESTING FP-GROWTH min_support = {fp_sup}")
        print(f"(Data pre-filtered with min_item_support={FIXED_MIN_ITEM_SUPPORT})")
        print("🔍"*30)
        
        result = run_random_forest_fpgrowth(
            df=df,
            min_support=fp_sup,
            **RF_CONFIG
        )
        
        if result:
            result['min_item_support'] = FIXED_MIN_ITEM_SUPPORT
            result['fp_min_support'] = fp_sup
            all_results.append(result)
            
            print("\n" + "📊"*30)
            print(f"EVALUATING PREDICTIVE POWER")
            print("📊"*30)
            
            predictive = evaluate_predictive_power(result, df, test_size=0.2)
            
            if predictive:
                all_predictive_results.append({
                    'min_item_support': FIXED_MIN_ITEM_SUPPORT,
                    'fp_min_support': fp_sup,
                    'prediction_accuracy': predictive['accuracy'],
                    'hits': predictive['hits'],
                    'total': predictive['total'],
                    'train_patterns': predictive['train_patterns'],
                    'train_rules': predictive['train_rules']
                })
        
        gc.collect()
    
    # Print final summary
    if all_results:
        print("\n" + "="*80)
        print("FINAL SUMMARY - PATTERN DISCOVERY")
        print(f"(min_item_support = {FIXED_MIN_ITEM_SUPPORT})")
        print("="*80)
        print(f"{'FP Min Sup':<12} {'Itemsets':<12} {'Rules':<10} {'Agreement':<12} {'Time(s)':<10}")
        print("-"*80)
        for r in sorted(all_results, key=lambda x: x['fp_min_support']):
            print(f"{r['fp_min_support']:<12.3f} {r['itemsets']:<12} {r['rules']:<10} "
                  f"{r['avg_agreement']:<11.1%} {r['time']:<10.2f}")
    
    if all_predictive_results:
        print("\n" + "="*80)
        print("FINAL SUMMARY - PREDICTIVE POWER")
        print(f"(min_item_support = {FIXED_MIN_ITEM_SUPPORT})")
        print("="*80)
        print(f"{'FP Min Sup':<12} {'Prediction Acc':<18} {'Correct/Total':<18} {'Train Patterns':<15}")
        print("-"*80)
        
        for p in sorted(all_predictive_results, key=lambda x: x['fp_min_support']):
            print(f"{p['fp_min_support']:<12.3f} "
                  f"{p['prediction_accuracy']:<17.2%} "
                  f"{p['hits']}/{p['total']:<15} "
                  f"{p['train_patterns']:<15}")
        
        best = max(all_predictive_results, key=lambda x: x['prediction_accuracy'])
        print(f"\n🏆 BEST PREDICTIVE PERFORMANCE:")
        print(f"   FP-Growth min_support: {best['fp_min_support']}")
        print(f"   Prediction Accuracy: {best['prediction_accuracy']:.2%}")
    
    print("\n" + "="*60)
    print("✅ COMPLETE")
    print("="*60)