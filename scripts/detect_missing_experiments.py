#!/usr/bin/env python3
"""
Script to detect missing experimental runs by comparing expected vs actual files.
Usage: python detect_missing_experiments.py <timestamp>
Example: python detect_missing_experiments.py 20260113_002155
"""

import argparse
import os
from collections import defaultdict
from pathlib import Path


def get_expected_experiments(models, datasets, num_splits=10):
    """Generate list of expected experiment configurations."""
    expected = []
    for model in models:
        for dataset in datasets:
            for split in range(num_splits):
                expected.append({
                    'model': model,
                    'dataset': dataset,
                    'split': split,
                    'filename': f"{model}_{dataset}_split{split}.csv"
                })
    return expected


def get_actual_experiments(raw_dir):
    """Get list of actual CSV files in the raw directory."""
    actual = []
    if not os.path.exists(raw_dir):
        return actual
    
    for filename in os.listdir(raw_dir):
        if filename.endswith('.csv'):
            actual.append(filename)
    
    return actual


def find_missing_experiments(expected, actual_files):
    """Find experiments that are missing."""
    actual_set = set(actual_files)
    missing = []
    
    for exp in expected:
        if exp['filename'] not in actual_set:
            missing.append(exp)
    
    return missing


def group_by_model(experiments):
    """Group experiments by model type."""
    grouped = defaultdict(list)
    for exp in experiments:
        grouped[exp['model']].append(exp)
    return grouped


def generate_missing_summary(grouped_missing):
    """Generate a summary of missing experiments."""
    summary_lines = [
        "=" * 80,
        "SUMMARY OF MISSING EXPERIMENTS",
        "=" * 80,
        ""
    ]
    
    total_missing = 0
    for model, exps in sorted(grouped_missing.items()):
        summary_lines.append(f"\n{model}:")
        summary_lines.append(f"  Total missing: {len(exps)}")
        
        # Group by dataset
        dataset_counts = defaultdict(list)
        for exp in exps:
            dataset_counts[exp['dataset']].append(exp['split'])
        
        for dataset, splits in sorted(dataset_counts.items()):
            splits.sort()
            summary_lines.append(f"    {dataset}: {len(splits)} runs - splits {splits}")
        
        total_missing += len(exps)
    
    summary_lines.extend([
        "",
        "=" * 80,
        f"TOTAL MISSING EXPERIMENTS: {total_missing}",
        "=" * 80,
    ])
    
    return '\n'.join(summary_lines)


def main():
    parser = argparse.ArgumentParser(
        description='Detect missing experimental runs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python detect_missing_experiments.py 20260113_002155
  python detect_missing_experiments.py 20260113_002155 --models SparseGPR SparseTPR
        """
    )
    parser.add_argument('timestamp', help='Timestamp of the experiment run')
    parser.add_argument('--models', nargs='+', 
                       default=['SparseGPR', 'SparseTPR', 'XuSparseTPR'],
                       help='List of model names to check (default: SparseGPR SparseTPR XuSparseTPR)')
    parser.add_argument('--datasets', nargs='+',
                       default=[
                           'Taxi', 'Energy', 'Kin8nm', 'Protein', 
                           'Bike', 'Concrete', 'Elevators',
                           'Taxi_Outliers', 'Energy_Outliers', 'Kin8nm_Outliers', 
                           'Protein_Outliers', 'Bike_Outliers', 'Concrete_Outliers', 
                           'Elevators_Outliers'
                       ],
                       help='List of dataset names to check')
    parser.add_argument('--num_splits', type=int, default=10,
                       help='Number of splits per dataset (default: 10)')
    parser.add_argument('--output_file', type=str, default=None,
                       help='Output file to save the missing experiments list')
    
    args = parser.parse_args()
    
    # Check if raw directory exists
    raw_dir = f"results/{args.timestamp}/raw"
    if not os.path.exists(raw_dir):
        print(f"Error: Raw directory not found: {raw_dir}")
        return 1
    
    print(f"Checking for missing experiments in {raw_dir}")
    print(f"Models: {', '.join(args.models)}")
    print(f"Datasets: {len(args.datasets)} datasets")
    print(f"Splits per dataset: {args.num_splits}")
    print("=" * 80)
    
    # Generate expected experiments
    expected = get_expected_experiments(args.models, args.datasets, args.num_splits)
    total_expected = len(expected)
    print(f"\nTotal expected experiments: {total_expected}")
    
    # Get actual experiments
    actual_files = get_actual_experiments(raw_dir)
    total_actual = len(actual_files)
    print(f"Total actual experiments: {total_actual}")
    
    # Find missing experiments
    missing = find_missing_experiments(expected, actual_files)
    total_missing = len(missing)
    print(f"Total missing experiments: {total_missing}")
    
    if total_missing == 0:
        print("\n✓ All expected experiments are present!")
        return 0
    
    # Group by model
    grouped_missing = group_by_model(missing)
    
    # Generate summary
    summary = generate_missing_summary(grouped_missing)
    print("\n" + summary)
    
    # Save to file if requested
    if args.output_file:
        output_path = args.output_file
    else:
        output_path = f"scripts/missing_experiments_{args.timestamp}.txt"
    
    with open(output_path, 'w') as f:
        f.write(f"Missing experiments for timestamp: {args.timestamp}\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(summary)
        f.write("\n\nDetailed list of missing files:\n")
        f.write("=" * 80 + "\n")
        for exp in sorted(missing, key=lambda x: (x['model'], x['dataset'], x['split'])):
            f.write(f"{exp['filename']}\n")
    
    print(f"\nMissing experiments list saved to: {output_path}")
    
    # Generate instructions for using with generate_rerun_jobs.py
    print("\n" + "=" * 80)
    print("NOTE: Missing experiments cannot be processed by generate_rerun_jobs.py")
    print("      (which only detects incomplete runs from the summary file).")
    print("")
    print("To rerun these missing experiments, you'll need to:")
    print(f"  1. Review the missing experiments in: {output_path}")
    print(f"  2. Manually submit jobs for these configurations, or")
    print(f"  3. Re-run the full experiment suite: bash scripts/run_sparse.sh")
    print("=" * 80)
    
    return 0


if __name__ == '__main__':
    from datetime import datetime
    exit(main())
