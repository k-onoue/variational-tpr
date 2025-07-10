import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')

class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 1)
        )

    def forward(self, x):
        return self.layers(x)
    

def load_data(dataset_path, dtype=torch.float64):
    """Loads training and testing data from a given path."""
    train_features_path = os.path.join(dataset_path, 'train_features.csv')
    train_target_path = os.path.join(dataset_path, 'train_target.csv')
    test_features_path = os.path.join(dataset_path, 'test_features.csv')
    test_target_path = os.path.join(dataset_path, 'test_target.csv')

    train_features = pd.read_csv(train_features_path, header=None).values
    train_target = pd.read_csv(train_target_path, header=None).values
    test_features = pd.read_csv(test_features_path, header=None).values
    test_target = pd.read_csv(test_target_path, header=None).values

    # Convert to torch tensors
    X_train = torch.tensor(train_features, dtype=dtype)
    y_train = torch.tensor(train_target, dtype=dtype)
    X_test = torch.tensor(test_features, dtype=dtype)
    y_test = torch.tensor(test_target, dtype=dtype)
    
    return X_train, y_train, X_test, y_test


def evaluate_datasets():
    base_path = '../datasets/dataset_tang_2017/'
    if not os.path.exists(base_path):
        base_path = 'datasets/dataset_tang_2017/'
        
    datasets = sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])
    results = defaultdict(list)
    
    for dataset in datasets:
        print(f"--- Evaluating dataset: {dataset} ---")
        dataset_rmses = []
        for i in range(10): # 10 splits
            split_path = os.path.join(base_path, dataset, f'split_{i}')
            
            try:
                X_train, y_train, X_test, y_test = load_data(split_path)
            except FileNotFoundError:
                print(f"  Split {i} not found for dataset {dataset}. Skipping.")
                continue

            input_dim = X_train.shape[1]
            model = MLP(input_dim)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.01)
            
            # Training loop
            epochs = 100
            for epoch in range(epochs):
                model.train()
                optimizer.zero_grad()
                outputs = model(X_train)
                loss = criterion(outputs, y_train)
                loss.backward()
                optimizer.step()

            # Evaluation
            model.eval()
            with torch.no_grad():
                predictions = model(X_test)
                rmse = np.sqrt(mean_squared_error(y_test.numpy(), predictions.numpy()))
                dataset_rmses.append(rmse)
                results[dataset].append(rmse)
                print(f"  Split {i}: RMSE = {rmse:.4f}")
        
        if dataset_rmses:
            avg_rmse = np.mean(dataset_rmses)
            std_rmse = np.std(dataset_rmses)
            print(f"  => Average RMSE for {dataset}: {avg_rmse:.4f} ± {std_rmse:.4f}")
            
    return results

results = evaluate_datasets()


results_df = pd.DataFrame(results)
print("\n--- Overall Results (RMSE) ---")
print(results_df.describe())