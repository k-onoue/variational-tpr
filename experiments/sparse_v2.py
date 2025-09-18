# import logging
# import os
# from datetime import datetime
# import pandas as pd
# import numpy as np
# import torch
# import json
# import copy
# import glob
# from tqdm import tqdm # <--- IMPORT TQDM

# from student import SparseTPR, XuSparseTPR, SparseGPR

# # --- Logging Configuration ---
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# # --- Data Loading Function ---
# def load_split_data(base_path, dataset_name, split_idx):
#     """Loads data for a specific dataset and split with double precision."""
#     split_path = os.path.join(base_path, dataset_name, f"split_{split_idx}")
#     X_train = pd.read_csv(os.path.join(split_path, "train_features.csv"), header=None).values.astype(np.float64)
#     y_train = pd.read_csv(os.path.join(split_path, "train_target.csv"), header=None).values.astype(np.float64)
#     X_test = pd.read_csv(os.path.join(split_path, "test_features.csv"), header=None).values.astype(np.float64)
#     y_test = pd.read_csv(os.path.join(split_path, "test_target.csv"), header=None).values.astype(np.float64)
#     return X_train, y_train, X_test, y_test

# # --- Main Experiment Function (MODIFIED WITH TQDM) ---
# def run_experiment(config, output_dir):
#     """
#     Runs experiments based on the provided configuration, saving results for each run immediately.
#     """
#     detailed_run_dir = os.path.join(output_dir, 'detailed_per_run')
#     os.makedirs(detailed_run_dir, exist_ok=True)
    
#     base_path = config['data']['base_path']
#     dataset_names = config['data']['dataset_names']
#     num_splits = config['data']['num_splits']
#     device = config['device']

#     # --- TQDM WRAPPER FOR MODELS ---
#     for model_name, model_config in tqdm(config['models'].items(), desc="Total Model Progress"):
#         logging.info(f"===== Running experiments for {model_name} model on {device} =====")
        
#         # --- TQDM WRAPPER FOR DATASETS ---
#         for dataset_name in tqdm(dataset_names, desc=f"Datasets for {model_name}", leave=False):
#             logging.info(f"--- Dataset: {dataset_name} ---")
            
#             # --- TQDM WRAPPER FOR SPLITS ---
#             for i in tqdm(range(num_splits), desc=f"Splits for {dataset_name}", leave=False):
#                 # Using leave=False for inner loops makes the progress bar disappear after completion,
#                 # keeping the console output cleaner.
                
#                 logging.info(f"  Running split {i}/{num_splits-1}...")
                
#                 run_filename = f"{model_name}_{dataset_name}_split{i}.csv"
#                 run_filepath = os.path.join(detailed_run_dir, run_filename)

#                 if os.path.exists(run_filepath):
#                     logging.info(f"  Skipping split {i}, result file already exists: {run_filepath}")
#                     continue
                
#                 X_train, y_train, X_test, y_test = load_split_data(base_path, dataset_name, i)
                
#                 X_train_t = torch.tensor(X_train, device=device)
#                 y_train_t = torch.tensor(y_train, device=device)
#                 X_test_t = torch.tensor(X_test, device=device)
#                 y_test_t = torch.tensor(y_test, device=device)

#                 model_class = model_config['class']
#                 hyper_settings = model_config['hyper_settings']
#                 fit_params = model_config['fit_params']

#                 init_settings = model_config['init_params']
#                 M, inducing_init = init_settings['num_inducing'], init_settings['inducing_init']
                
#                 model = model_class(
#                     X_train_t, y_train_t, M,
#                     inducing_init_method=inducing_init,
#                     hyper_settings=hyper_settings, device=device)
                
#                 history = model.fit(X_test=X_test_t, y_test=y_test_t, **fit_params)

#                 num_epochs_trained = len(history['loss'])
#                 base_run_data = {
#                     'model': model_name,
#                     'dataset': dataset_name,
#                     'split': i,
#                     'epoch': np.arange(1, num_epochs_trained + 1),
#                     'loss': history.get('loss', [np.nan] * num_epochs_trained),
#                     'elbo': history.get('elbo', [np.nan] * num_epochs_trained),
#                     'log_prior': history.get('log_prior', [np.nan] * num_epochs_trained)
#                 }
#                 run_df = pd.DataFrame(base_run_data)
                
#                 if history.get('eval_epochs') and len(history['eval_epochs']) > 0:
#                     eval_data_list = history['eval_metrics']
                    
#                     if isinstance(eval_data_list, list) and all(isinstance(item, dict) for item in eval_data_list):
#                         eval_df = pd.DataFrame(eval_data_list)
#                         eval_df['epoch'] = history['eval_epochs']
                        
#                         fit_times = history.get('fit_times', [])
#                         if len(fit_times) == num_epochs_trained:
#                             run_df['time'] = fit_times
                        
#                         run_df = pd.merge(run_df, eval_df, on='epoch', how='left')
                
#                 run_df.to_csv(run_filepath, index=False, float_format='%.6f')
#                 logging.info(f"  Saved intermediate results to {run_filepath}")
    
#     logging.info("===== All experimental runs complete. ======")

# # --- NEW Function to Aggregate Results ---
# def load_and_aggregate_results(output_dir):
#     """Loads all individual run CSVs from the 'detailed_per_run' subdirectory and concatenates them."""
#     detailed_run_dir = os.path.join(output_dir, 'detailed_per_run')
#     csv_files = glob.glob(os.path.join(detailed_run_dir, '*.csv'))
    
#     if not csv_files:
#         logging.warning("No individual result files found to aggregate.")
#         return pd.DataFrame()
        
#     df_list = [pd.read_csv(f) for f in csv_files]
#     aggregated_df = pd.concat(df_list, ignore_index=True)
#     logging.info(f"Aggregated {len(df_list)} result files into a single DataFrame.")
#     return aggregated_df

# # --- Results Saving Function ---
# def save_results(detailed_df, summary_df, config, output_dir):
#     """Saves the final aggregated experiment results and configuration."""
#     os.makedirs(output_dir, exist_ok=True)
    
#     detailed_csv_path = os.path.join(output_dir, 'results_detailed_aggregated.csv')
#     detailed_df.to_csv(detailed_csv_path, index=False, float_format='%.6f')
    
#     summary_csv_path = os.path.join(output_dir, 'results_summary.csv')
#     summary_df.to_csv(summary_csv_path, index=False, float_format='%.4f')
    
#     logging.info(f"\nAggregated detailed results saved to {detailed_csv_path}")
#     logging.info(f"Summary of final results saved to {summary_csv_path}")
    
#     config_to_save = copy.deepcopy(config)
#     for model_name, model_config in config_to_save['models'].items():
#         if 'class' in model_config:
#             model_config['class'] = model_config['class'].__name__

#     config_json_path = os.path.join(output_dir, 'experiment_config.json')
#     with open(config_json_path, 'w') as f:
#         json.dump(config_to_save, f, indent=4)
        
#     logging.info(f"Experiment configuration saved to {config_json_path}")


# # --- Main Execution Block ---
# if __name__ == '__main__':
#     # Make sure to install tqdm: pip install tqdm
#     torch.set_default_dtype(torch.float64)

#     EXPERIMENT_CONFIG = {
#         # 'data': {
#         #     'base_path': './datasets/dataset_combined/',
#         #     'dataset_names': [
#         #         'Bike', 'Concrete', 'Concrete_Outliers', 'Elevators',
#         #         'Energy', 'Kin8nm', 'Kin8nm_Outliers', 'Protein'
#         #     ],
#         #     'num_splits': 10,
#         # },
#         'data': {
#             'base_path': './datasets/dataset_combined/',
#             'dataset_names': [
#                 'Taxi'
#             ],
#             'num_splits': 1,
#         },
#         'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',
#         'models': {
#             # 'SparseGPR': {
#             #     'class': SparseGPR,
#             #     'init_params': { 
#             #         'num_inducing': 256, 
#             #         'inducing_init': 'kmeans' 
#             #     },
#             #     'fit_params': { 
#             #         'epochs': 1000, 
#             #         'eval_interval': 1, 
#             #         'batch_size': 1024, 
#             #         'hyper_lr': 0.01, 
#             #         'var_lr': 0.1 
#             #     },
#             #     'hyper_settings': { 
#             #         "lengthscale": {"optim": "MAP"}, 
#             #         "outputscale": {"optim": "MAP"}, 
#             #         "noisescale":  {"optim": "MAP"}, 
#             #     }
#             # },
#             'SparseTPR': {
#                 'class': SparseTPR,
#                 'init_params': { 
#                     'num_inducing': 256, 
#                     'inducing_init': 'kmeans' 
#                 },
#                 'fit_params': { 
#                     'epochs': 1000, 
#                     'eval_interval': 1, 
#                     'batch_size': 1024, 
#                     'hyper_lr': 0.01, 
#                     'var_lr': 0.1 
#                 },
#                 'hyper_settings': { 
#                     "lengthscale": {"optim": "MAP"}, 
#                     "outputscale": {"optim": "FIX", "init": 1.0}, 
#                     "noisescale":  {"optim": "MAP"}, 
#                     "dof_func":    {"optim": "MAP"}, 
#                     "dof_lik":     {"optim": "MAP"}, 
#                 }
#             },
#             # 'XuSparseTPR': {
#             #     'class': XuSparseTPR,
#             #     'init_params': { 
#             #         'num_inducing': 256, 
#             #         'inducing_init': 'kmeans' 
#             #     },
#             #     'fit_params': { 
#             #         'epochs': 1000, 
#             #         'eval_interval': 1, 
#             #         'batch_size': 1024, 
#             #         'lr': 0.01, 
#             #         'num_samples': 1000 
#             #     },
#             #     'hyper_settings': { 
#             #         'lengthscale': {'optim': 'MAP'}, 
#             #         'outputscale': {'optim': 'FIX', 'init': 1.0 }, 
#             #         'noisescale':  {'optim': 'MAP'}, 
#             #         'dof_func':    {'optim': 'MAP'}, 
#             #         'dof_lik':     {'optim': 'MAP'}, 
#             #     }
#             # },
#         }
#     }

#     timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#     output_directory = os.path.join('./results', timestamp)
#     os.makedirs(output_directory, exist_ok=True)
    
#     run_experiment(EXPERIMENT_CONFIG, output_directory)
    
#     detailed_results_df = load_and_aggregate_results(output_directory)
    
#     if not detailed_results_df.empty:
#         max_epoch = detailed_results_df['epoch'].max()
#         logging.info(f"Aggregating results at final epoch: {max_epoch}")
#         final_epoch_results = detailed_results_df[detailed_results_df['epoch'] == max_epoch].copy()
        
#         summary_df = final_epoch_results.groupby(['model', 'dataset'])['rmse'].agg(['mean', 'std']).reset_index()
#         summary_df.rename(columns={'mean': 'rmse_mean', 'std': 'rmse_std'}, inplace=True)
        
#         print("\n\n" + "="*60)
#         print(f"       SUMMARY OF FINAL RESULTS (RMSE at Epoch {max_epoch})")
#         print("="*60)
#         pd.set_option('display.float_format', '{:.4f}'.format)
#         print(summary_df)
        
#         save_results(detailed_results_df, summary_df, EXPERIMENT_CONFIG, output_directory)
#     else:
#         logging.info("No results were generated, skipping final summary and save.")




import logging
import os
from datetime import datetime
import pandas as pd
import numpy as np
import torch
import json
import copy
import glob
from tqdm import tqdm

from student import SparseTPR, XuSparseTPR, SparseGPR

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

# --- MODIFIED Main Experiment Function ---
def run_experiment(config, output_dir):
    """
    Runs experiments based on the provided configuration, saving results for each run immediately.
    """
    detailed_run_dir = os.path.join(output_dir, 'detailed_per_run')
    os.makedirs(detailed_run_dir, exist_ok=True)
    
    base_path = config['data']['base_path']
    dataset_names = config['data']['dataset_names']
    num_splits = config['data']['num_splits']
    device = config['device']

    for model_name, model_config in tqdm(config['models'].items(), desc="Total Model Progress"):
        logging.info(f"===== Running experiments for {model_name} model on {device} =====")
        
        for dataset_name in tqdm(dataset_names, desc=f"Datasets for {model_name}", leave=False):
            logging.info(f"--- Dataset: {dataset_name} ---")
            
            for i in tqdm(range(num_splits), desc=f"Splits for {dataset_name}", leave=False):
                logging.info(f"  Running split {i}/{num_splits-1}...")
                
                run_filename = f"{model_name}_{dataset_name}_split{i}.csv"
                run_filepath = os.path.join(detailed_run_dir, run_filename)

                if os.path.exists(run_filepath):
                    logging.info(f"  Skipping split {i}, result file already exists: {run_filepath}")
                    continue
                
                X_train, y_train, X_test, y_test = load_split_data(base_path, dataset_name, i)
                
                X_train_t = torch.tensor(X_train, device=device)
                y_train_t = torch.tensor(y_train, device=device)
                X_test_t = torch.tensor(X_test, device=device)
                y_test_t = torch.tensor(y_test, device=device)

                model_class = model_config['class']
                hyper_settings = model_config['hyper_settings']
                fit_params = model_config['fit_params']

                init_settings = model_config['init_params']
                M, inducing_init = init_settings['num_inducing'], init_settings['inducing_init']
                
                model = model_class(
                    X_train_t, y_train_t, M,
                    inducing_init_method=inducing_init,
                    hyper_settings=hyper_settings, device=device)
                
                # --- FIX STARTS HERE ---
                # 1. Create a list to store the yielded results from each epoch.
                history_list = []
                
                # 2. Get the generator from model.fit().
                training_generator = model.fit(X_test=X_test_t, y_test=y_test_t, **fit_params)
                
                # 3. Iterate through the generator and append each epoch's results to the list.
                for epoch_results in training_generator:
                    history_list.append(epoch_results)
                
                # 4. If training produced results, convert the list of dictionaries to a DataFrame.
                if history_list:
                    run_df = pd.DataFrame(history_list)
                    run_df['model'] = model_name
                    run_df['dataset'] = dataset_name
                    run_df['split'] = i
                    
                    run_df.to_csv(run_filepath, index=False, float_format='%.6f')
                    logging.info(f"  Saved intermediate results to {run_filepath}")
                else:
                    logging.warning(f"  No history was generated for {model_name} on {dataset_name} split {i}.")
                # --- FIX ENDS HERE ---

    logging.info("===== All experimental runs complete. ======")

# --- Function to Aggregate Results ---
def load_and_aggregate_results(output_dir):
    """Loads all individual run CSVs from the 'detailed_per_run' subdirectory and concatenates them."""
    detailed_run_dir = os.path.join(output_dir, 'detailed_per_run')
    csv_files = glob.glob(os.path.join(detailed_run_dir, '*.csv'))
    
    if not csv_files:
        logging.warning("No individual result files found to aggregate.")
        return pd.DataFrame()
        
    df_list = [pd.read_csv(f) for f in csv_files]
    aggregated_df = pd.concat(df_list, ignore_index=True)
    logging.info(f"Aggregated {len(df_list)} result files into a single DataFrame.")
    return aggregated_df

# --- Results Saving Function ---
def save_results(detailed_df, summary_df, config, output_dir):
    """Saves the final aggregated experiment results and configuration."""
    os.makedirs(output_dir, exist_ok=True)
    
    detailed_csv_path = os.path.join(output_dir, 'results_detailed_aggregated.csv')
    detailed_df.to_csv(detailed_csv_path, index=False, float_format='%.6f')
    
    summary_csv_path = os.path.join(output_dir, 'results_summary.csv')
    summary_df.to_csv(summary_csv_path, index=False, float_format='%.4f')
    
    logging.info(f"\nAggregated detailed results saved to {detailed_csv_path}")
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
            'dataset_names': ['Taxi', 'Taxi_Outliers'],
            'num_splits': 10,
        },
        'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',
        'models': {
            #     'SparseGPR': {
            #     'class': SparseGPR,
            #     'init_params': { 
            #         'num_inducing': 256, 
            #         'inducing_init': 'kmeans' 
            #     },
            #     'fit_params': { 
            #         'epochs': 1000, 
            #         'eval_interval': 1, 
            #         'batch_size': 1024, 
            #         'hyper_lr': 0.01, 
            #         'var_lr': 0.1 
            #     },
            #     'hyper_settings': { 
            #         "lengthscale": {"optim": "MAP"}, 
            #         "outputscale": {"optim": "MAP"}, 
            #         "noisescale":  {"optim": "MAP"}, 
            #     }
            # },
            # 'SparseTPR': {
            #     'class': SparseTPR,
            #     'init_params': { 
            #         'num_inducing': 256, 
            #         'inducing_init': 'kmeans' 
            #     },
            #     'fit_params': { 
            #         'epochs': 1000, 
            #         'eval_interval': 1, 
            #         'batch_size': 1024, 
            #         'hyper_lr': 0.01, 
            #         'var_lr': 0.1 
            #     },
            #     'hyper_settings': { 
            #         "lengthscale": {"optim": "MAP"}, 
            #         "outputscale": {"optim": "FIX", "init": 1.0}, 
            #         "noisescale":  {"optim": "MAP"}, 
            #         "dof_func":    {"optim": "MAP"}, 
            #         "dof_lik":     {"optim": "MAP"}, 
            #     }
            # },
            'XuSparseTPR': {
                'class': XuSparseTPR,
                'init_params': { 
                    'num_inducing': 256, 
                    'inducing_init': 'kmeans' 
                },
                'fit_params': { 
                    'epochs': 1000, 
                    'eval_interval': 1, 
                    'batch_size': 1024, 
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
        }
    }

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_directory = os.path.join('./results', timestamp)
    os.makedirs(output_directory, exist_ok=True)
    
    run_experiment(EXPERIMENT_CONFIG, output_directory)
    
    detailed_results_df = load_and_aggregate_results(output_directory)
    
    if not detailed_results_df.empty:
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
    else:
        logging.info("No results were generated, skipping final summary and save.")