# Closed-Form Coordinate Ascent Variational Inference for Student-t Process Regression with Student-t Likelihood

This work is accepted by AISTATS 2026.

## 1. Installation

### 1.1 Prerequisites
- OS: Ubuntu 22.04.5 LTS
- Python: 3.12.2
- Job Manager: Slurm 23.02.3

### 1.2 Installation Steps

Run the following command:

```
git clone https://github.com/k-onoue/variational-tpr.git
cd variational-tpr
pyenv local 3.12
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## 2. Directory Structure

```
.
├── src/student         # Python source code
├── experiments         # Scripts and definitions for reproducing experiments
├── datasets            # Datasets
├── scripts             # Scripts to run experiments
├── logs                # Include this in .gitignore
├── results             # Include this in .gitignore
└── requirements.txt    # List of dependencies
```

## 3. Usage: How to Configure Hyperparameter Optimization Method

The `SparseTPR` model's hyperparameters—**`lengthscale`**, **`outputscale`**, **`dof_func`**, and **`dof_lik`**—are configured through a single, powerful dictionary argument: **`hyper_settings`**. This approach gives you precise, per-parameter control over both initialization and optimization.

### 3.1 The `hyper_settings` Dictionary

To configure the model, you pass a dictionary to the `hyper_settings` argument. The keys are the names of the hyperparameters. The value for each key is another dictionary that can contain two settings: **`init`** and **`optim`**.

1.  **`init`**: Sets the **initial value**.
    * If you provide a value (e.g., **`"init": 1.0`**), that value is used as the starting point.
    * If you **omit `init`**, a starting value is automatically **sampled** from that hyperparameter's default prior distribution.

2.  **`optim`**: Sets the **optimization mode**.
    * **`'MLE'`** (Maximum Likelihood): This is the **default**. It optimizes the hyperparameter to best fit the data, without using a prior as a regularizer.
    * **`'MAP'`** (Maximum a Posteriori): Optimizes the hyperparameter using its prior as a regularizer. This can help prevent extreme values and improve model stability.
    * **`'FIX'`**: The hyperparameter is **not optimized**. It remains fixed at its initial value throughout training.


__Configuration Summary__

This table shows how different settings for a hyperparameter (e.g., `outputscale`) affect its behavior.

| `hyper_settings` Example                      | `"optim"` Mode                  | `"init"` Value                      | Resulting Behavior                      |
| :-------------------------------------------- | :------------------------------ | :---------------------------------- | :-------------------------------------- |
| `{'outputscale': {}}`                         | Omitted (defaults to `'MLE'`)   | Omitted (defaults to `sample`)      | Sampled initial value, optimized with **MLE**.    |
| `{'outputscale': {'init': 1.0}}`              | Omitted (defaults to `'MLE'`)   | `1.0`                               | Starts at 1.0, optimized with **MLE**.    |
| `{'outputscale': {'optim': 'MAP'}}`           | `'MAP'`                         | Omitted (defaults to `sample`)      | Sampled initial value, optimized with **MAP**.    |
| `{'outputscale': {'optim': 'FIX', 'init': 1.0}}` | `'FIX'`                         | `1.0`                               | Fixed at 1.0, **not optimized**.        |

If a hyperparameter is completely left out of the `hyper_settings` dictionary, it uses the default behavior (sampled `init`, `MLE` optimization).

### 3.2 Examples

__(a) Default Behavior: Sampled MLE__

If you don't provide the `hyper_settings` argument, all hyperparameters are sampled from their priors and optimized using MLE.

```python
# All hyperparameters are sampled and optimized via MLE.
model = SparseTPR(X_train, y_train, M=25)
```

__(b) Full MAP with Custom Initial Values__
To run MAP estimation starting from specific points for all hyperparameters:

```python
settings = {
    'lengthscale': {"optim": "MAP", "init": 1.5},
    'outputscale': {"optim": "MAP", "init": 0.5},
    'dof_func':    {"optim": "MAP", "init": 4.0},
    'dof_lik':     {"optim": "MAP", "init": 4.0}
}

model = SparseTPR(X_train, y_train, M=25, hyper_settings=settings)
```

__(c) Fixed Configuration__
This example shows how to combine different strategies, which is the primary strength of this API.

```python
settings = {
    # Use MAP for lengthscale, but sample the starting point.
    'lengthscale': {"optim": "MAP"},   
    
    # Fix the kernel variance; do not train it.
    'outputscale': {"optim": "FIX", "init": 1.0},
    
    # Use MLE for the functional DoF, starting from 5.0.
    'dof_func':    {"optim": "MLE", "init": 5.0},      
    
    # Use the default for likelihood DoF (sampled init, MLE optim).
    'dof_lik':     {} 
}

model = SparseTPR(X_train, y_train, M=25, hyper_settings=settings)
```


## 4. Experiments

### 4.1 Setup datasets

Downloads kaggle.json to setup Kaggle API.

```bash
mv ~/Downloads/kaggle.json ~/.kaggle/
```

Then, run the following command.

```bash
bash scripts/setup_datasets.sh
```

### 4.2 Hardware and Runtime Environment

All experiments in this paper were conducted under the following hardware and runtime conditions:

**For non-sparse models:**

- **CPU:** Intel Xeon Gold 6230R (4 cores allocated per run)
- **Memory:** 8 GB
- **Timeout:** 3,600 seconds (1 hour)

**For sparse models:**

- **CPU:** Intel Xeon Gold 6230R (4 cores allocated per run)
- **Memory:** 16 GB
- **GPU:** 1 x NVIDIA A100
- **Timeout:** 14,400 seconds (4 hours)

### 4.3 Running Experiments

The experiments reported in the paper can be started by running the shell scripts located in the `scripts` directory from the project root directory.

__Running the Main Experiments__
To reproduce the main results from the paper, run the following command:

```sh
bash scripts/run_dense.sh
bash scripts/run_sparse.sh
```

__Running the Ablation Studies__
For the ablation studies for `TangTPR` and `XuTPR`, run the following command:

```sh
bash scripts/run_dense_ablation.sh
```


## 5. Citation

Citation information is pending. We will add the BibTeX entry here once the paper is publicly available.