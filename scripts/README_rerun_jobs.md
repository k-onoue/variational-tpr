# Rerun Job Generator for Incomplete Experiments

This toolset automatically detects problematic experimental runs and generates bash scripts to rerun them. It handles two types of issues:
- **Incomplete runs**: Experiments that started but didn't reach the required number of epochs
- **Missing experiments**: Experiments that were never run or failed to generate output files

## Quick Start

**One-command workflow (recommended):**
```bash
bash scripts/analyze_and_generate_reruns.sh <timestamp>
```
Example: `bash scripts/analyze_and_generate_reruns.sh 20260113_002155`

This runs all analysis steps and generates rerun scripts automatically.

**Step-by-step workflow:**

1. **Check your experimental results:**
   ```bash
   bash scripts/check_final_epochs.sh <timestamp>
   ```
   Example: `bash scripts/check_final_epochs.sh 20260113_002155`

2. **Detect missing experiments (optional but recommended):**
   ```bash
   python scripts/detect_missing_experiments.py <timestamp>
   ```
   Example: `python scripts/detect_missing_experiments.py 20260113_002155`

3. **Generate rerun scripts for incomplete experiments:**
   ```bash
   python scripts/generate_rerun_jobs.py <timestamp>
   ```
   Example: `python scripts/generate_rerun_jobs.py 20260113_002155`

4. **Submit rerun jobs:**
   ```bash
   bash scripts/rerun_all_<timestamp>.sh
   ```
   Example: `bash scripts/rerun_all_20260113_002155.sh`

## Detailed Usage

### detect_missing_experiments.py

This script checks for experiments that were never run or failed to generate output files by comparing expected vs. actual files.

**Syntax:**
```bash
python scripts/detect_missing_experiments.py <timestamp> [OPTIONS]
```

**Required Arguments:**
- `<timestamp>`: Timestamp of the experiment run (e.g., `20260113_002155`)

**Optional Arguments:**
- `--models <list>`: List of model names to check (default: `SparseGPR SparseTPR XuSparseTPR`)
- `--datasets <list>`: List of dataset names to check  
- `--num_splits <value>`: Number of splits per dataset (default: 10)
- `--output_file <path>`: Custom output file path

**Examples:**
```bash
# Basic usage
python scripts/detect_missing_experiments.py 20260113_002155

# Check specific models only
python scripts/detect_missing_experiments.py 20260113_002155 --models XuSparseTPR

# Custom output location
python scripts/detect_missing_experiments.py 20260113_002155 --output_file results/missing.txt
```

**Output:**
- Console summary of missing experiments
- Text file listing all missing experiment configurations

### generate_rerun_jobs.py

This Python script analyzes the output from `check_final_epochs.sh` and generates scripts to rerun incomplete experiments.

**Syntax:**
```bash
python scripts/generate_rerun_jobs.py <timestamp> [OPTIONS]
```

**Required Arguments:**
- `<timestamp>`: Timestamp of the experiment run (e.g., `20260113_002155`)

**Optional Arguments:**
- `--min_epochs <value>`: Minimum number of epochs required (default: 1000)
- `--python_script <path>`: Python script to use for rerunning (default: `experiments/sparse_v6.py`)
- `--output_dir <path>`: Directory to save generated scripts (default: `scripts`)
- `--original_timestamp <timestamp>`: Original experiment timestamp for tracking lineage (auto-detected if available)
- `--consolidate`: Save rerun results directly to original timestamp directory (overwrites files)

**Examples:**
```bash
# Basic usage
python scripts/generate_rerun_jobs.py 20260113_002155

# Consolidate mode - saves reruns to original directory (overwrites)
python scripts/generate_rerun_jobs.py 20260113_002155 --consolidate

# Iterative rerun - analyzing a rerun directory
python scripts/generate_rerun_jobs.py 20260118_203034
# (auto-detects original timestamp from ORIGINAL_TIMESTAMP file)

# Custom minimum epochs
python scripts/generate_rerun_jobs.py 20260113_002155 --min_epochs 900

# Custom Python script
python scripts/generate_rerun_jobs.py 20260113_002155 --python_script experiments/sparse_v7.py

# Custom output directory
python scripts/generate_rerun_jobs.py 20260113_002155 --output_dir scripts/reruns
```

## Output Files

The script generates several files:

1. **Summary file**: `rerun_summary_<timestamp>.txt`
   - Overview of incomplete runs grouped by model and dataset

2. **Model-specific rerun scripts**: `rerun_<model>_<timestamp>.sh`
   - Individual scripts for each model type (e.g., `rerun_xusparsetpr_20260113_002155.sh`)
   - Contains SLURM job submissions for all incomplete runs of that model

3. **Master rerun script**: `rerun_all_<timestamp>.sh`
   - Convenience script that runs all model-specific scripts
   - Submits all incomplete jobs at once

## Understanding Missing vs. Incomplete Experiments

### Missing Experiments
- **Definition**: Experiment configurations that were supposed to run but never produced an output CSV file
- **Causes**: 
  - Job never submitted
  - Job failed immediately (before writing any output)
  - Job was cancelled before starting
- **Detection**: Compare expected file list vs. actual files in raw directory
- **Tool**: `detect_missing_experiments.py`

### Incomplete Experiments  
- **Definition**: Experiments that started and produced output but didn't reach the required number of epochs
- **Causes**:
  - Job timeout (exceeded time limit)
  - Out of memory error
  - Early stopping due to convergence issues
  - Manual cancellation during execution
- **Detection**: Parse CSV files to check final epoch count
- **Tool**: `generate_rerun_jobs.py` (processes output from `check_final_epochs.sh`)

**Important**: The two tools are complementary:
- `detect_missing_experiments.py` finds experiments that never ran
- `generate_rerun_jobs.py` finds experiments that started but didn't finish

For a complete picture, run both tools!

## Rerun Modes

### Non-Consolidate Mode (Default)
- Creates a new results directory with timestamp: `results/<timestamp>_rerun_<model>/`
- Original experiment data remains untouched
- Keeps separate history of rerun attempts
- **Use when**: You want to preserve all rerun attempts separately

### Consolidate Mode (`--consolidate`)
- Saves results directly to the original timestamp directory
- Overwrites incomplete CSV files with new results
- Maintains single authoritative results directory
- **Use when**: You want all final results in one place
- **Warning**: This will overwrite existing files!

## Iterative Rerun Workflow

Sometimes experiments may fail multiple times. The tools support iterative reruns with lineage tracking:

### Workflow with Lineage Tracking

```bash
# Initial experiment (some fail)
Original: results/20260113_002155/

# First rerun (some still incomplete)
bash scripts/check_final_epochs.sh 20260113_002155
python scripts/generate_rerun_jobs.py 20260113_002155
bash scripts/rerun_all_20260113_002155.sh
# Creates: results/20260118_120000_rerun_XuSparseTPR/
#   with ORIGINAL_TIMESTAMP file pointing to 20260113_002155

# Second rerun (check the first rerun results)
bash scripts/check_final_epochs.sh 20260118_120000
python scripts/generate_rerun_jobs.py 20260118_120000
# Automatically detects original timestamp from ORIGINAL_TIMESTAMP file
# Creates new rerun with proper lineage tracking

# Continue until all experiments complete...
```

### Consolidate Workflow (Recommended)

```bash
# Initial experiment
Original: results/20260113_002155/

# Rerun in consolidate mode (overwrites failed experiments)
bash scripts/check_final_epochs.sh 20260113_002155
python scripts/generate_rerun_jobs.py 20260113_002155 --consolidate
bash scripts/rerun_all_20260113_002155.sh
# Results saved back to: results/20260113_002155/raw/

# Check again and repeat if needed
bash scripts/check_final_epochs.sh 20260113_002155
python scripts/generate_rerun_jobs.py 20260113_002155 --consolidate
bash scripts/rerun_all_20260113_002155.sh

# All results end up in the original directory
```

## Workflow

### Complete Workflow Example

```bash
# Step 1: Check your experimental results
bash scripts/check_final_epochs.sh 20260113_002155

# Step 2: Detect missing experiments
python scripts/detect_missing_experiments.py 20260113_002155

# Step 3: Review the missing experiments
cat scripts/missing_experiments_20260113_002155.txt

# Step 4: Generate rerun scripts for incomplete experiments
python scripts/generate_rerun_jobs.py 20260113_002155

# Step 5: Review the incomplete runs summary
cat scripts/rerun_summary_20260113_002155.txt

# Step 6: Submit rerun jobs for incomplete experiments
bash scripts/rerun_all_20260113_002155.sh

# Note: Missing experiments need to be handled separately
# (either manually or by re-running the full experiment suite)
```

### Monitoring Reruns

After submitting jobs, you can monitor them using:

```bash
# Check job status
squeue -u $USER

# View logs
tail -f results/<rerun_timestamp>/logs/*.out
tail -f results/<rerun_timestamp>/logs/*.err

# After completion, check the rerun results
bash scripts/check_final_epochs.sh <rerun_timestamp>
```

## Understanding the Output

### Summary Format

```
================================================================================
SUMMARY OF INCOMPLETE RUNS
Minimum required epochs: 1000
================================================================================

XuSparseTPR:
  Total incomplete: 39
    Bike_Outliers: 5 runs - splits [1, 2, 3, 4, 5]
    Protein: 9 runs - splits [0, 1, 2, 3, 4, 5, 6, 7, 8]
    ...
================================================================================
TOTAL INCOMPLETE RUNS: 39
================================================================================
```

This shows:
- Which models have incomplete runs
- How many runs per dataset
- Which specific splits need to be rerun

### Generated Scripts

Each generated script:
- Creates a new results directory with timestamp
- Submits individual SLURM jobs for each incomplete run
- Uses the same configuration as original runs
- Preserves dataset and split information

## Important Notes

1. **Prerequisites**: Must run `check_final_epochs.sh` first to generate the summary file

2. **New timestamps**: Rerun results are saved with a new timestamp to avoid overwriting

3. **Job array vs. individual jobs**: Rerun scripts submit individual jobs (not arrays) for better control

4. **Resource allocation**: Default configuration:
   - Partition: `gpu_short`
   - Time: 4 hours
   - GPU: 1
   - CPUs: 4
   - Memory: 16G

5. **Script modification**: If you need different resource allocation, edit the generated scripts before submitting

## Troubleshooting

### Error: Summary file not found

```
Error: Summary file not found: results/<timestamp>/final_epochs_summary_<timestamp>.txt
Please run check_final_epochs.sh <timestamp> first
```

**Solution**: Run `bash scripts/check_final_epochs.sh <timestamp>` first

### No incomplete runs found

```
No incomplete runs found! All experiments completed successfully.
```

This is good news! All experiments completed the required epochs.

### Adjusting minimum epochs

If you want to rerun experiments with fewer epochs (e.g., for testing):

```bash
python scripts/generate_rerun_jobs.py 20260113_002155 --min_epochs 500
```

## Advanced Usage

### Custom Filtering

To rerun only specific datasets or models, you can:

1. Edit the generated scripts before submission
2. Create a custom version of `generate_rerun_jobs.py` with additional filters
3. Manually select which jobs to submit from the generated scripts

### Batch Processing Multiple Timestamps

```bash
# Check multiple experimental runs
for ts in 20260113_002155 20260114_120000 20260115_083000; do
    echo "Processing $ts..."
    bash scripts/check_final_epochs.sh $ts
    python scripts/generate_rerun_jobs.py $ts
done
```

## See Also

- `check_final_epochs.sh`: Script to check experimental completion
- `README_final_epochs.md`: Documentation for checking final epochs
- `run_sparse.sh`: Original experiment submission script
