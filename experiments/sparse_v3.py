# experiments/sparse_v3.py

import logging
import os
import pandas as pd
import numpy as np
import torch
import json
import copy
import argparse

from student import SparseTPR, XuSparseTPR, SparseGPR


# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# --- Data Loading Function ---
def load_split_data(base_path, dataset_name, split_idx):
    """Loads data for a specific dataset and split with double precision."""
    split_path = os.path.join(base_path, dataset_name, f"split_{split_idx}")
    X_train = pd.read_csv(
        os.path.join(split_path, "train_features.csv"), header=None
    ).values.astype(np.float64)
    y_train = pd.read_csv(
        os.path.join(split_path, "train_target.csv"), header=None
    ).values.astype(np.float64)
    X_test = pd.read_csv(
        os.path.join(split_path, "test_features.csv"), header=None
    ).values.astype(np.float64)
    y_test = pd.read_csv(
        os.path.join(split_path, "test_target.csv"), header=None
    ).values.astype(np.float64)
    return X_train, y_train, X_test, y_test


# --- Single Experiment Function ---
def run_single_experiment(model_name, dataset_name, split_idx, config, output_base_dir):
    """
    Runs a single experiment for a given model, dataset, and split index.
    """
    base_path = config["data"]["base_path"]
    device = config["device"]

    logging.info(
        f"===== Running experiment for {model_name} on {dataset_name} (Split {split_idx}) on {device} ====="
    )

    run_filename = f"{model_name}_{dataset_name}_split{split_idx}.csv"
    run_filepath = os.path.join(output_base_dir, run_filename)

    if os.path.exists(run_filepath):
        logging.info(
            f"Skipping split {split_idx}, result file already exists: {run_filepath}"
        )
        return

    X_train, y_train, X_test, y_test = load_split_data(
        base_path, dataset_name, split_idx
    )

    X_train_t = torch.tensor(X_train, device=device)
    y_train_t = torch.tensor(y_train, device=device)
    X_test_t = torch.tensor(X_test, device=device)
    y_test_t = torch.tensor(y_test, device=device)

    model_config = config["models"][model_name]
    model_class = model_config["class"]
    hyper_settings = model_config["hyper_settings"]
    fit_params = model_config["fit_params"]
    init_settings = model_config["init_params"]
    M, inducing_init = init_settings["num_inducing"], init_settings["inducing_init"]

    model = model_class(
        X_train_t,
        y_train_t,
        M,
        inducing_init_method=inducing_init,
        hyper_settings=hyper_settings,
        device=device,
    )

    history = model.fit(X_test=X_test_t, y_test=y_test_t, **fit_params)

    num_epochs_trained = len(history["loss"])
    base_run_data = {
        "model": model_name,
        "dataset": dataset_name,
        "split": split_idx,
        "epoch": np.arange(1, num_epochs_trained + 1),
        "loss": history.get("loss", [np.nan] * num_epochs_trained),
        "elbo": history.get("elbo", [np.nan] * num_epochs_trained),
        "log_prior": history.get("log_prior", [np.nan] * num_epochs_trained),
    }
    run_df = pd.DataFrame(base_run_data)

    if history.get("eval_epochs") and len(history["eval_epochs"]) > 0:
        eval_df = pd.DataFrame(history["eval_metrics"])
        eval_df["epoch"] = history["eval_epochs"]
        fit_times = history.get("fit_times", [])
        if len(fit_times) == num_epochs_trained:
            run_df["time"] = fit_times
        run_df = pd.merge(run_df, eval_df, on="epoch", how="left")

    os.makedirs(output_base_dir, exist_ok=True)
    run_df.to_csv(run_filepath, index=False, float_format="%.6f")
    logging.info(f"Saved results to {run_filepath}")


# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a single instance of a sparse model experiment."
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Name of the model to run."
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help="Name of the dataset to use."
    )
    parser.add_argument(
        "--split", type=int, required=True, help="The index of the data split to use."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the raw CSV result.",
    )
    args = parser.parse_args()

    torch.set_default_dtype(torch.float64)

    EXPERIMENT_CONFIG = {
        "data": {
            "base_path": "./datasets/dataset_combined/",
        },
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "models": {
            "SparseGPR": {
                "class": SparseGPR,
                "init_params": {"num_inducing": 256, "inducing_init": "kmeans"},
                "fit_params": {
                    "epochs": 1000,
                    "eval_interval": 1,
                    "batch_size": 1024,
                    "hyper_lr": 0.01,
                    "var_lr": 0.1,
                },
                "hyper_settings": {
                    "lengthscale": {"optim": "MAP"},
                    "outputscale": {"optim": "MAP"},
                    "noisescale": {"optim": "MAP"},
                },
            },
            "SparseTPR": {
                "class": SparseTPR,
                "init_params": {"num_inducing": 256, "inducing_init": "kmeans"},
                "fit_params": {
                    "epochs": 1000,
                    "eval_interval": 1,
                    "batch_size": 1024,
                    "hyper_lr": 0.01,
                    "var_lr": 0.1,
                },
                "hyper_settings": {
                    "lengthscale": {"optim": "MAP"},
                    "outputscale": {"optim": "FIX", "init": 1.0},
                    "noisescale": {"optim": "MAP"},
                    "dof_func": {"optim": "MAP"},
                    "dof_lik": {"optim": "MAP"},
                },
            },
            "XuSparseTPR": {
                "class": XuSparseTPR,
                "init_params": {"num_inducing": 256, "inducing_init": "kmeans"},
                "fit_params": {
                    "epochs": 1000,
                    "eval_interval": 1,
                    "batch_size": 1024,
                    "lr": 0.01,
                    "num_samples": 1000,
                },
                "hyper_settings": {
                    "lengthscale": {"optim": "MAP"},
                    "outputscale": {"optim": "FIX", "init": 1.0},
                    "noisescale": {"optim": "MAP"},
                    "dof_func": {"optim": "MAP"},
                    "dof_lik": {"optim": "MAP"},
                },
            },
        },
    }

    run_single_experiment(
        args.model, args.dataset, args.split, EXPERIMENT_CONFIG, args.output_dir
    )
