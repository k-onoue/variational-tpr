# Final Epoch Checker Scripts

Two scripts to examine the final epoch recorded for each CSV file in experimental runs.

## Bash Script (Recommended for quick checks)

**Usage:**
```bash
./scripts/check_final_epochs.sh <timestamp>
```

**Example:**
```bash
./scripts/check_final_epochs.sh 20260113_002155
```

**Features:**
- Fast execution with no dependencies
- Shows detailed results for all CSV files
- Provides summary statistics by model
- Identifies incomplete runs (epoch < 1000)
- Saves results to `results/<timestamp>/final_epochs_summary_<timestamp>.txt`

**Output includes:**
- File-by-file listing with final epoch, RMSE, NLL
- Summary by model (file count, avg epoch, avg RMSE, avg NLL)
- Warning list for incomplete runs

## Python Script (For advanced analysis)

**Usage:**
```bash
python scripts/check_final_epochs.py <timestamp>
```

**Example:**
```bash
python scripts/check_final_epochs.py 20260113_002155
```

**Requirements:**
- pandas

**Features:**
- Detailed statistical analysis
- Saves results to CSV format
- Provides grouped statistics by model

## Output Format

Both scripts display:
```
File                                    Model      Dataset       Split  Epoch  RMSE        NLL
--------------------------------------------------------------------------------------------
SparseGPR_Bike_split0.csv              SparseGPR  Bike          0      1000   0.226852    4.883201
SparseTPR_Bike_split0.csv              SparseTPR  Bike          0      1000   0.218954    1.017264
...
```

Summary statistics:
```
Summary by Model:
--------------------------------------------------------------------------------------------
SparseGPR: 140 files, avg epoch: 1000, avg RMSE: 0.426, avg NLL: 1853.81
SparseTPR: 245 files, avg epoch: 934.7, avg RMSE: 0.408, avg NLL: 1.649
XuSparseTPR: 105 files, avg epoch: 847.7, avg RMSE: 0.342, avg NLL: 1.455
```

Incomplete runs warning:
```
Checking for incomplete runs (epoch < 1000):
--------------------------------------------------------------------------------------------
  ⚠️  XuSparseTPR_Protein_Outliers_split0.csv: epoch 644
  ⚠️  XuSparseTPR_Taxi_split1.csv: epoch 51
...
```
