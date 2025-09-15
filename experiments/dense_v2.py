import logging
import os
from datetime import datetime
import pandas as pd
import numpy as np
import torch
import json
import copy

from student import TPR, XuTPR, TangTPR, GPR

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- Data Loading Function ---
def load_split_data(base_path, dataset_name, split_idx):
    """Loads data for a specific dataset and split with double precision."""
    split_path = os.path.join(base_path, dataset_name, f"split_{split_idx}")
    X_train = pd.read_csv(os.path.join(split_path, "train_features.csv"), header=None).values.astype(np.float64)
    y_train = pd.read_csv(os.path.join(split_path, "train_target.csv"), header=None).values.astype(np.float64)
    X_test = pd.read_csv(os.path.join(split_path, "test_features.csv"), header=None).values.astype(np.float64)
    y_test = pd.read_csv(os.path.join(split_path, "test_target.csv"), header=None).values.astype(np.float64)
    return X_train, y_train, X_test, y_test

# --- Main Experiment Function ---
def run_experiment(config):
    """
    Runs experiments based on the provided configuration dictionary and returns detailed epoch-wise results.
    """
    all_results_dfs = []
    
    base_path = config['data']['base_path']
    dataset_names = config['data']['dataset_names']
    num_splits = config['data']['num_splits']
    device = config['device']

    for model_name, model_config in config['models'].items():
        logging.info(f"===== Running experiments for {model_name} model on {device} =====")
        
        for dataset_name in dataset_names:
            logging.info(f"--- Dataset: {dataset_name} ---")
            
            for i in range(num_splits):
                logging.info(f"  Running split {i}/{num_splits-1}...")
                
                X_train, y_train, X_test, y_test = load_split_data(base_path, dataset_name, i)
                
                X_train_t = torch.tensor(X_train, device=device)
                y_train_t = torch.tensor(y_train, device=device)
                X_test_t = torch.tensor(X_test, device=device)
                y_test_t = torch.tensor(y_test, device=device)

                model_class = model_config['class']
                hyper_settings = model_config['hyper_settings']
                fit_params = model_config['fit_params']
                
                model = model_class(X_train_t, y_train_t, hyper_settings=hyper_settings, device=device)
                
                history = model.fit(X_test=X_test_t, y_test=y_test_t, **fit_params)

                num_epochs_trained = len(history['loss'])
                base_run_data = {
                    'model': model_name,
                    'dataset': dataset_name,
                    'split': i,
                    'epoch': np.arange(1, num_epochs_trained + 1),
                    'loss': history.get('loss', [np.nan] * num_epochs_trained),
                    'elbo': history.get('elbo', [np.nan] * num_epochs_trained),
                    'log_prior': history.get('log_prior', [np.nan] * num_epochs_trained)
                }
                run_df = pd.DataFrame(base_run_data)
                
                if history.get('eval_epochs') and len(history['eval_epochs']) > 0:
                    eval_data_list = history['eval_metrics']
                    
                    if isinstance(eval_data_list, list) and all(isinstance(item, dict) for item in eval_data_list):
                        eval_df = pd.DataFrame(eval_data_list)
                        eval_df['epoch'] = history['eval_epochs']
                        
                        # Ensure 'fit_times' aligns with the number of recorded epochs
                        fit_times = history.get('fit_times', [])
                        if len(fit_times) == num_epochs_trained:
                            run_df['time'] = fit_times
                        
                        run_df = pd.merge(run_df, eval_df, on='epoch', how='left')
                
                all_results_dfs.append(run_df)
        
    return pd.concat(all_results_dfs, ignore_index=True) if all_results_dfs else pd.DataFrame()

# --- Results Saving Function ---
def save_results(detailed_df, summary_df, config, output_dir):
    """Saves the experiment results and configuration to the specified directory."""
    os.makedirs(output_dir, exist_ok=True)
    
    detailed_csv_path = os.path.join(output_dir, 'results_detailed.csv')
    summary_csv_path = os.path.join(output_dir, 'results_summary.csv')
    
    detailed_df.to_csv(detailed_csv_path, index=False, float_format='%.6f')
    summary_df.to_csv(summary_csv_path, index=False, float_format='%.4f')
    
    logging.info(f"\nDetailed epoch-wise results saved to {detailed_csv_path}")
    logging.info(f"Summary of final results saved to {summary_csv_path}")
    
    config_to_save = copy.deepcopy(config)
    for model_name, model_config in config_to_save['models'].items():
        if 'class' in model_config:
            model_config['class'] = model_config['class'].__name__

    config_json_path = os.path.join(output_dir, 'experiment_config.json')
    with open(config_json_path, 'w') as f:
        json.dump(config_to_save, f, indent=4)
        
    logging.info(f"Experiment configuration saved to {config_json_path}")


# --- Main Execution Block ---
if __name__ == '__main__':
    torch.set_default_dtype(torch.float64)

    EXPERIMENT_CONFIG = {
        'data': {
            'base_path': './datasets/dataset_combined/',
            'dataset_names': [
                'Boston', 'Diabetes', 'ELE', 'MPG',
                'Machine_CPU', 'Neal', 'Neal_XOutlier', 'Yacht'
            ],
            'num_splits': 10,
        },
        'device': 'cpu',
        'models': {
            'GPR': {
                'class': GPR,
                'fit_params': {
                    'epochs': 100,
                    'eval_interval': 1,
                    'lr': 0.01
                },
                'hyper_settings': {
                    "lengthscale": {"optim": "MAP"},
                    "outputscale": {"optim": "MAP"},
                    "noisescale":  {"optim": "MAP"},
                }
            },
            'TPR': {
                'class': TPR,
                'fit_params': {
                    'epochs': 100,
                    'eval_interval': 1,
                    'hyper_lr': 0.01
                },
                'hyper_settings': {
                    "lengthscale": {"optim": "MAP"},
                    "outputscale": {"optim": "FIX", "init": 1.0},
                    "noisescale":  {"optim": "MAP"},
                    "dof_func":    {"optim": "MAP"},
                    "dof_lik":     {"optim": "MAP"},
                }
            },
            'XuTPR': {
                'class': XuTPR,
                'fit_params': {
                    'epochs': 100,
                    'eval_interval': 1,
                    'lr': 0.01,
                    'num_samples': 1000
                },
                'hyper_settings': {
                    'lengthscale': {'optim': 'MAP'},
                    'outputscale': {'optim': 'FIX', 'init': 1.0 },
                    'noisescale':  {'optim': 'MAP'},
                    'dof_func':    {'optim': 'MAP'},
                    'dof_lik':     {'optim': 'MAP'},
                }
            },
            # ▼▼▼ MODIFICATION 2: Add TangTPR configuration ▼▼▼
            'TangTPR': {
                'class': TangTPR,
                'fit_params': {
                    'epochs': 100,
                    'eval_interval': 1,
                    'lr_hyper': 0.01, # Learning rate for hyperparameters
                    'lr_f': 0.1,      # Learning rate for latent mode f_hat
                    'f_steps': 10     # Inner optimization steps for f_hat
                },
                'hyper_settings': {
                    'lengthscale': {'optim': 'MAP'},
                    'outputscale': {'optim': 'FIX', 'init': 1.0 },
                    'noisescale':  {'optim': 'MAP'},
                    'dof_func':    {'optim': 'MAP'},
                    'dof_lik':     {'optim': 'MAP'},
                }
            }
            # ▲▲▲ END MODIFICATION 2 ▲▲▲
        }
    }

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_directory = os.path.join('./results', timestamp)
    
    detailed_results_df = run_experiment(EXPERIMENT_CONFIG)
    
    # --- Aggregate Results ---
    # Find the maximum epoch number across all models to ensure we select the final results correctly
    max_epoch = detailed_results_df['epoch'].max()
    logging.info(f"Aggregating results at final epoch: {max_epoch}")
    final_epoch_results = detailed_results_df[detailed_results_df['epoch'] == max_epoch].copy()
    
    summary_df = final_epoch_results.groupby(['model', 'dataset'])['rmse'].agg(['mean', 'std']).reset_index()
    summary_df.rename(columns={'mean': 'rmse_mean', 'std': 'rmse_std'}, inplace=True)
    
    print("\n\n" + "="*60)
    print(f"       SUMMARY OF FINAL RESULTS (RMSE at Epoch {max_epoch})")
    print("="*60)
    pd.set_option('display.float_format', '{:.4f}'.format)
    print(summary_df)
    
    save_results(detailed_results_df, summary_df, EXPERIMENT_CONFIG, output_directory)