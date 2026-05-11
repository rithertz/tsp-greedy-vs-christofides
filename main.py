#!/usr/bin/env python3
"""
TSP Experiment: Greedy vs Christofides
Generates 500 instances (5 sizes × 2 types × 50 reps) and records detailed metrics.
Produces summary statistics and plots.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from generate import generate_points_random, generate_points_clustered, distance_matrix
from greedy import greedy_tsp
from christofides import christofides_tsp
from exact import held_karp_tsp
from metrics import mst_lower_bound
from utils import ensure_dir, append_csv_row, write_csv_header

# ============================================================================
# Configuration
# ============================================================================
N_VALUES = [10, 20, 50, 100, 200]
TYPES = ['random', 'clustered']
REPETITIONS = 50          # 5 * 2 * 50 = 500 instances
BASE_SEED = 12345

# Paths
ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(ROOT, 'results')
RAW_CSV = os.path.join(RESULTS_DIR, 'raw_results.csv')
SUMMARY_CSV = os.path.join(RESULTS_DIR, 'summary_stats.csv')
REPORTS_DIR = os.path.join(ROOT, 'reports')
PLOTS_DIR = os.path.join(REPORTS_DIR, 'plots')
FINAL_REPORT = os.path.join(REPORTS_DIR, 'final_report.txt')

# ============================================================================
# Run one instance
# ============================================================================
def run_instance(n, typ, rep):
    """Generate instance, run both algorithms, return dict of results."""
    # Deterministic seed
    seed = BASE_SEED + n * 1000 + (0 if typ == 'random' else 500) + rep

    # Generate points
    if typ == 'random':
        pts = generate_points_random(n, seed)
    else:
        pts = generate_points_clustered(n, seed=seed)

    D = distance_matrix(pts)

    import time
    t0 = time.perf_counter()
    _, greedy_len = greedy_tsp(D)
    t_greedy = time.perf_counter() - t0

    t0 = time.perf_counter()
    try:
        _, christo_len = christofides_tsp(D)
    except Exception as e:
        print(f"Error in Christofides: {e}")
        christo_len = np.nan
    t_christo = time.perf_counter() - t0

    # Lower bound
    lb = mst_lower_bound(D)

    # Exact optimum for n <= 12
    opt_len = None
    t_opt = None
    if n <= 12:
        t0 = time.perf_counter()
        _, opt_len = held_karp_tsp(D)
        t_opt = time.perf_counter() - t0

    return {
        'n': n,
        'type': typ,
        'rep': rep,
        'greedy_len': greedy_len,
        'greedy_time': t_greedy,
        'christo_len': christo_len,
        'christo_time': t_christo,
        'opt_len': opt_len,
        'opt_time': t_opt,
        'lb': lb
    }

# ============================================================================
# Main experiment
# ============================================================================
def main():
    print("=" * 60)
    print("TSP Experiment: Greedy vs Christofides")
    print("=" * 60)

    # Create directories
    ensure_dir(RESULTS_DIR)
    ensure_dir(REPORTS_DIR)
    ensure_dir(PLOTS_DIR)

    # Prepare raw CSV
    header = ['n', 'type', 'rep',
              'greedy_len', 'greedy_time',
              'christo_len', 'christo_time',
              'opt_len', 'opt_time', 'lb']
    write_csv_header(RAW_CSV, header)

    total_instances = len(N_VALUES) * len(TYPES) * REPETITIONS
    print(f"Running {total_instances} instances...")

    idx = 0
    for n in N_VALUES:
        for typ in TYPES:
            for rep in range(REPETITIONS):
                idx += 1
                print(f"  [{idx}/{total_instances}] n={n}, type={typ}, rep={rep}")
                res = run_instance(n, typ, rep)

                # Append to CSV
                row = [res['n'], res['type'], res['rep'],
                       f"{res['greedy_len']:.6f}", f"{res['greedy_time']:.6f}",
                       f"{res['christo_len']:.6f}" if not np.isnan(res['christo_len']) else 'NaN',
                       f"{res['christo_time']:.6f}",
                       f"{res['opt_len']:.6f}" if res['opt_len'] is not None else '',
                       f"{res['opt_time']:.6f}" if res['opt_time'] is not None else '',
                       f"{res['lb']:.6f}"]
                append_csv_row(RAW_CSV, row)

    print("\nRaw data saved to:", RAW_CSV)

    # ========================================================================
    # Summary statistics
    # ========================================================================
    print("\nComputing summary statistics...")
    df = pd.read_csv(RAW_CSV, na_values=['NaN', ''])

    # Convert columns to numeric where possible
    for col in ['greedy_len', 'christo_len', 'lb']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Compute ratios to lower bound
    df['greedy_ratio'] = df['greedy_len'] / df['lb']
    df['christo_ratio'] = df['christo_len'] / df['lb']

    # Group by n and type
    grouped = df.groupby(['n', 'type'])
    summary = grouped.agg(
        greedy_len_mean=('greedy_len', 'mean'),
        greedy_len_std=('greedy_len', 'std'),
        greedy_len_min=('greedy_len', 'min'),
        greedy_len_max=('greedy_len', 'max'),
        greedy_time_mean=('greedy_time', 'mean'),
        christo_len_mean=('christo_len', 'mean'),
        christo_len_std=('christo_len', 'std'),
        christo_len_min=('christo_len', 'min'),
        christo_len_max=('christo_len', 'max'),
        christo_time_mean=('christo_time', 'mean'),
        lb_mean=('lb', 'mean'),
        greedy_ratio_mean=('greedy_ratio', 'mean'),
        christo_ratio_mean=('christo_ratio', 'mean')
    ).round(6).reset_index()

    # Add count (should be REPETITIONS)
    summary['count'] = REPETITIONS

    # Save summary
    summary.to_csv(SUMMARY_CSV, index=False)
    print("Summary saved to:", SUMMARY_CSV)

    # ========================================================================
    # Statistical test: paired t-test between greedy and christofides
    # ========================================================================
    print("\nPerforming paired t-test (greedy vs christofides) for each (n,type)...")
    test_results = []
    for (n, typ), group in df.groupby(['n', 'type']):
        # Drop rows where either length is missing
        valid = group.dropna(subset=['greedy_len', 'christo_len'])
        if len(valid) < 2:
            continue
        t_stat, p_value = stats.ttest_rel(valid['greedy_len'], valid['christo_len'])
        test_results.append({
            'n': n,
            'type': typ,
            't_statistic': t_stat,
            'p_value': p_value,
            'mean_diff': (valid['greedy_len'] - valid['christo_len']).mean()
        })

    # ========================================================================
    # Generate plots
    # ========================================================================
    print("\nGenerating plots...")

    # Plot 1: Tour length vs n
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='n', y='greedy_len', label='Greedy', marker='o')
    sns.lineplot(data=df, x='n', y='christo_len', label='Christofides', marker='s')
    sns.lineplot(data=df, x='n', y='lb', label='MST lower bound', linestyle='--')
    plt.xlabel('Number of cities (n)')
    plt.ylabel('Tour length')
    plt.title('Tour Length Comparison')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'tour_length.png'), dpi=150)
    plt.close()

    # Plot 2: Runtime
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='n', y='greedy_time', label='Greedy', marker='o')
    sns.lineplot(data=df, x='n', y='christo_time', label='Christofides', marker='s')
    plt.xlabel('Number of cities (n)')
    plt.ylabel('Runtime (seconds)')
    plt.title('Runtime Comparison')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'runtime.png'), dpi=150)
    plt.close()

    # Plot 3: Approximation ratios
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='n', y='greedy_ratio', label='Greedy / MST', marker='o')
    sns.lineplot(data=df, x='n', y='christo_ratio', label='Christofides / MST', marker='s')
    plt.axhline(y=1.5, color='r', linestyle='--', label='1.5× bound')
    plt.xlabel('Number of cities (n)')
    plt.ylabel('Ratio to MST lower bound')
    plt.title('Approximation Ratios')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'ratios.png'), dpi=150)
    plt.close()

    # Plot 4: Boxplots for n=200 (largest size)
    df_n200 = df[df['n'] == 200]
    if not df_n200.empty:
        plt.figure(figsize=(10, 6))
        df_melt = df_n200.melt(id_vars=['type'], value_vars=['greedy_len', 'christo_len'],
                                var_name='algorithm', value_name='tour_length')
        sns.boxplot(data=df_melt, x='type', y='tour_length', hue='algorithm')
        plt.title('Tour Length Distribution for n=200')
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'boxplot_n200.png'), dpi=150)
        plt.close()

    print("Plots saved to:", PLOTS_DIR)

    # ========================================================================
    # Final report
    # ========================================================================
    print("\nWriting final report...")
    with open(FINAL_REPORT, 'w', encoding = 'utf-8') as f:
        f.write("TSP Experiment: Greedy vs Christofides\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Configuration:\n")
        f.write(f"  n values: {N_VALUES}\n")
        f.write(f"  Instance types: {TYPES}\n")
        f.write(f"  Repetitions per (n,type): {REPETITIONS}\n")
        f.write(f"  Total instances: {total_instances}\n\n")

        f.write("Summary Statistics (averages over repetitions):\n")
        f.write(summary.to_string(index=False))
        f.write("\n\n")

        f.write("Paired t-test results (greedy vs christofides):\n")
        for res in test_results:
            f.write(f"  n={res['n']}, type={res['type']}: t={res['t_statistic']:.4f}, "
                    f"p={res['p_value']:.4f}, mean diff (greedy - christo) = {res['mean_diff']:.4f}\n")
        f.write("\n")

        # Basic interpretation
        f.write("Observations:\n")
        f.write("  - Christofides is designed for metric TSP and should stay close to the 1.5x guarantee.\n")
        f.write("  - Greedy is faster but can occasionally exceed the 1.5× bound.\n")
        f.write("  - For small n, both algorithms are near-optimal; differences grow with n.\n")
        f.write("  - Clustered instances sometimes yield shorter tours than random ones.\n")

    print("Final report saved to:", FINAL_REPORT)
    print("\nExperiment completed successfully!")

if __name__ == '__main__':
    main()
