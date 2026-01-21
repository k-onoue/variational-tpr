#!/usr/bin/env python3
"""
Script to automatically detect incomplete experimental runs and generate bash scripts to rerun them.
Usage: python generate_rerun_jobs.py <timestamp> [--min_epochs <value>]
Example: python generate_rerun_jobs.py 20260113_002155 --min_epochs 1000
"""

import argparse
import os
import re
from collections import defaultdict
from pathlib import Path
from datetime import datetime


def parse_summary_file(summary_file_path):
    """Parse the final_epochs_summary file and extract incomplete experiments."""
    incomplete_runs = []
    
    with open(summary_file_path, 'r') as f:
        lines = f.readlines()
    
    # Skip header lines (first 4 lines)
    data_lines = [line for line in lines if line.strip() and not line.startswith('--')]
    
    # Parse each data line
    for line in data_lines:
        parts = line.split()
        if len(parts) >= 7 and parts[0].endswith('.csv'):
            filename = parts[0]
            
            # Extract split from filename using regex instead of CSV column
            match = re.match(r"(\w+)_(.+)_split(\d+)\.csv", filename)
            if not match:
                continue
            
            model = match.group(1)
            dataset = match.group(2)
            split_num = int(match.group(3))
            
            # Get epoch from the parsed line
            epoch = parts[4]
            
            try:
                epoch_num = int(epoch)
                incomplete_runs.append({
                    'filename': filename,
                    'model': model,
                    'dataset': dataset,
                    'split': split_num,
                    'epoch': epoch_num
                })
            except ValueError:
                continue
    
    return incomplete_runs


def filter_incomplete_runs(runs, min_epochs=1000):
    """Filter runs that didn't reach minimum epochs."""
    return [run for run in runs if run['epoch'] < min_epochs]


def group_by_model(incomplete_runs):
    """Group incomplete runs by model type."""
    grouped = defaultdict(list)
    for run in incomplete_runs:
        grouped[run['model']].append(run)
    return grouped


def generate_rerun_script(model_name, runs, timestamp, python_script="experiments/sparse_v6.py", 
                          original_timestamp=None, consolidate=False):
    """Generate a bash script to rerun incomplete experiments for a specific model.
    
    Args:
        model_name: Name of the model
        runs: List of incomplete runs
        timestamp: Current timestamp being analyzed
        python_script: Path to the Python experiment script
        original_timestamp: Original experiment timestamp (for tracking lineage)
        consolidate: If True, save results to original timestamp directory
    """
    
    # Determine original timestamp for tracking
    if original_timestamp is None:
        original_timestamp = timestamp
    
    # Generate output timestamp for the rerun
    rerun_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Choose output directory based on consolidate flag
    if consolidate:
        output_dir = f"results/{original_timestamp}"
    else:
        output_dir = f"results/{rerun_timestamp}_rerun_{model_name}"
    
    # Group by dataset
    dataset_splits = defaultdict(list)
    for run in runs:
        dataset_splits[run['dataset']].append(run['split'])
    
    script_lines = [
        "#!/bin/bash",
        "",
        f"# Rerun script for incomplete {model_name} experiments",
        f"# Analyzed timestamp: {timestamp}",
        f"# Original timestamp: {original_timestamp}",
        f"# Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"# Consolidate mode: {'Yes (overwrites in original dir)' if consolidate else 'No (creates new dir)'}",
        "",
        "# --- Configuration ---",
        f'MODEL_NAME="{model_name}"',
        f'PYTHON_SCRIPT="{python_script}"',
        f'ORIGINAL_TIMESTAMP="{original_timestamp}"',
        "",
        "# Create output directories",
        f'OUTPUT_DIR="{output_dir}"',
        'mkdir -p "${OUTPUT_DIR}/snapshots"',
        'mkdir -p "${OUTPUT_DIR}/raw"',
        'mkdir -p "${OUTPUT_DIR}/logs"',
        "",
        "# Save lineage information",
        f'echo "{original_timestamp}" > "${{OUTPUT_DIR}}/ORIGINAL_TIMESTAMP"',
        f'echo "{timestamp}" >> "${{OUTPUT_DIR}}/RERUN_FROM"',
        'date >> "${OUTPUT_DIR}/RERUN_FROM"',
        "",
        "# Save snapshots",
        'cp "${PYTHON_SCRIPT}" "${OUTPUT_DIR}/snapshots/"',
        'cp "$0" "${OUTPUT_DIR}/snapshots/"',
        "",
        f'echo "Rerunning incomplete {model_name} experiments..."',
        f'echo "Original experiment: {original_timestamp}"',
        f'echo "Results will be saved in ${{OUTPUT_DIR}}"',
        'echo "=================================================="',
        "",
        "# Submit individual jobs for each incomplete run",
        "",
    ]
    
    job_count = 0
    for dataset, splits in sorted(dataset_splits.items()):
        splits.sort()
        script_lines.append(f"# Dataset: {dataset}")
        
        for split in splits:
            job_count += 1
            script_lines.extend([
                "",
                f'echo "Submitting job {job_count}: {model_name}, {dataset}, split {split}"',
                "sbatch << 'EOF'",
                "#!/bin/bash -l",
                f"#SBATCH --job-name={model_name}_{dataset}_s{split}",
                "#SBATCH --partition=gpu_short",
                "#SBATCH --time=4:00:00",
                "#SBATCH --gres=gpu:1",
                "#SBATCH --output=${OUTPUT_DIR}/logs/%x_%j.out",
                "#SBATCH --error=${OUTPUT_DIR}/logs/%x_%j.err",
                "#SBATCH --nodes=1",
                "#SBATCH --ntasks-per-node=1",
                "#SBATCH --cpus-per-task=4",
                "#SBATCH --mem=16G",
                "",
                'echo "Job started on $(hostname) at $(date)"',
                f'echo "Running: Model={model_name}, Dataset={dataset}, Split={split}"',
                "",
                f"python {python_script} \\",
                f'    --model "{model_name}" \\',
                f'    --dataset "{dataset}" \\',
                f'    --split {split} \\',
                f'    --output_dir "{output_dir}/raw"',
                "",
                'echo "Job finished at $(date)"',
                "EOF",
                "",
            ])
    
    script_lines.extend([
        "",
        f'echo "Submitted {job_count} jobs for {model_name}"',
        "",
    ])
    
    return '\n'.join(script_lines), job_count


def generate_summary(grouped_runs, min_epochs):
    """Generate a summary of incomplete runs."""
    summary_lines = [
        "=" * 80,
        "SUMMARY OF INCOMPLETE RUNS",
        f"Minimum required epochs: {min_epochs}",
        "=" * 80,
        ""
    ]
    
    total_incomplete = 0
    for model, runs in sorted(grouped_runs.items()):
        summary_lines.append(f"\n{model}:")
        summary_lines.append(f"  Total incomplete: {len(runs)}")
        
        # Group by dataset
        dataset_counts = defaultdict(list)
        for run in runs:
            dataset_counts[run['dataset']].append(run['split'])
        
        for dataset, splits in sorted(dataset_counts.items()):
            splits.sort()
            summary_lines.append(f"    {dataset}: {len(splits)} runs - splits {splits}")
        
        total_incomplete += len(runs)
    
    summary_lines.extend([
        "",
        "=" * 80,
        f"TOTAL INCOMPLETE RUNS: {total_incomplete}",
        "=" * 80,
    ])
    
    return '\n'.join(summary_lines)


def main():
    parser = argparse.ArgumentParser(
        description='Generate rerun scripts for incomplete experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_rerun_jobs.py 20260113_002155
  python generate_rerun_jobs.py 20260113_002155 --min_epochs 1000
  python generate_rerun_jobs.py 20260113_002155 --output_dir scripts/reruns
        """
    )
    parser.add_argument('timestamp', help='Timestamp of the experiment run')
    parser.add_argument('--min_epochs', type=int, default=1000,
                       help='Minimum number of epochs required (default: 1000)')
    parser.add_argument('--python_script', type=str, default='experiments/sparse_v6.py',
                       help='Python script to use for rerunning (default: experiments/sparse_v6.py)')
    parser.add_argument('--output_dir', type=str, default='scripts',
                       help='Directory to save generated scripts (default: scripts)')
    parser.add_argument('--original_timestamp', type=str, default=None,
                       help='Original experiment timestamp (for tracking rerun lineage)')
    parser.add_argument('--consolidate', action='store_true',
                       help='Save rerun results to original timestamp directory (overwrites)')
    
    args = parser.parse_args()
    
    # Find the summary file
    summary_file = f"results/{args.timestamp}/final_epochs_summary_{args.timestamp}.txt"
    
    if not os.path.exists(summary_file):
        print(f"Error: Summary file not found: {summary_file}")
        print(f"Please run check_final_epochs.sh {args.timestamp} first")
        return 1
    
    print(f"Reading summary file: {summary_file}")
    
    # Parse the summary file
    all_runs = parse_summary_file(summary_file)
    print(f"Found {len(all_runs)} total experimental runs")
    
    # Filter incomplete runs
    incomplete_runs = filter_incomplete_runs(all_runs, args.min_epochs)
    print(f"Found {len(incomplete_runs)} incomplete runs (< {args.min_epochs} epochs)")
    
    if not incomplete_runs:
        print("No incomplete runs found! All experiments completed successfully.")
        return 0
    
    # Group by model
    grouped_runs = group_by_model(incomplete_runs)
    
    # Generate summary
    summary = generate_summary(grouped_runs, args.min_epochs)
    print("\n" + summary)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save summary to file
    summary_output = os.path.join(args.output_dir, f"rerun_summary_{args.timestamp}.txt")
    with open(summary_output, 'w') as f:
        f.write(summary)
    print(f"\nSummary saved to: {summary_output}")
    
    # Determine original timestamp
    if args.original_timestamp:
        original_timestamp = args.original_timestamp
    else:
        # Check if current timestamp is itself a rerun
        original_ts_file = f"results/{args.timestamp}/ORIGINAL_TIMESTAMP"
        if os.path.exists(original_ts_file):
            with open(original_ts_file, 'r') as f:
                original_timestamp = f.read().strip()
            print(f"\nDetected rerun lineage: {args.timestamp} -> {original_timestamp}")
        else:
            original_timestamp = args.timestamp
    
    # Generate rerun scripts for each model
    print("\nGenerating rerun scripts...")
    if args.consolidate:
        print(f"WARNING: Consolidate mode enabled - results will OVERWRITE files in {original_timestamp}")
    total_jobs = 0
    
    for model, runs in sorted(grouped_runs.items()):
        script_content, job_count = generate_rerun_script(
            model, runs, args.timestamp, args.python_script,
            original_timestamp=original_timestamp, consolidate=args.consolidate
        )
        
        # Save script
        script_filename = f"rerun_{model.lower()}_{args.timestamp}.sh"
        script_path = os.path.join(args.output_dir, script_filename)
        
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Make script executable
        os.chmod(script_path, 0o755)
        
        print(f"  ✓ Generated {script_path} ({job_count} jobs)")
        total_jobs += job_count
    
    # Generate master rerun script
    master_script = [
        "#!/bin/bash",
        "",
        f"# Master script to rerun all incomplete experiments from {args.timestamp}",
        f"# Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
        "",
        'echo "=================================================="',
        f'echo "Rerunning all incomplete experiments from {args.timestamp}"',
        'echo "=================================================="',
        "",
    ]
    
    for model in sorted(grouped_runs.keys()):
        script_filename = f"rerun_{model.lower()}_{args.timestamp}.sh"
        master_script.extend([
            f'echo "Launching {model} jobs..."',
            f'bash "$SCRIPT_DIR/{script_filename}"',
            'sleep 1  # Small delay between model submissions',
            "",
        ])
    
    master_script.extend([
        'echo "=================================================="',
        f'echo "All {total_jobs} jobs have been submitted"',
        'echo "=================================================="',
    ])
    
    master_script_path = os.path.join(args.output_dir, f"rerun_all_{args.timestamp}.sh")
    with open(master_script_path, 'w') as f:
        f.write('\n'.join(master_script))
    os.chmod(master_script_path, 0o755)
    
    print(f"  ✓ Generated {master_script_path} (master script)")
    
    print("\n" + "=" * 80)
    print(f"SUCCESS: Generated rerun scripts for {total_jobs} incomplete experiments")
    print("=" * 80)
    print("\nTo rerun all incomplete experiments:")
    print(f"  bash {master_script_path}")
    print("\nOr to rerun specific models:")
    for model in sorted(grouped_runs.keys()):
        script_filename = f"rerun_{model.lower()}_{args.timestamp}.sh"
        print(f"  bash {os.path.join(args.output_dir, script_filename)}")
    
    return 0


if __name__ == '__main__':
    exit(main())