#!/bin/bash

# Dataset Setup Automation Script
# This script will clone datasets_regression, generate datasets, and move them to the correct location

set -e  # Exit on error

echo "╔═══════════════════════════════════════════════════════════════════╗"
echo "║              BENCHMARK DATASET SETUP AUTOMATION                   ║"
echo "╚═══════════════════════════════════════════════════════════════════╝"
echo ""

# Get the project root directory (parent of scripts/)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Clone datasets_regression if it doesn't exist
if [ ! -d "datasets_regression" ]; then
    echo "→ Cloning datasets_regression repository..."
    git clone https://github.com/k-onoue/datasets_regression.git
else
    echo "→ datasets_regression directory already exists, skipping clone..."
fi

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "⚠ uv is not installed."
    echo "Please install uv first:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo "Or visit: https://github.com/astral-sh/uv"
    exit 1
fi

# Create/update virtual environment
echo "→ Setting up virtual environment..."
if [ ! -d ".venv" ]; then
    uv venv --python 3.12 .venv
fi

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
echo "→ Installing dependencies..."
uv pip install -e .

# Navigate to datasets_regression and run the setup
echo ""
echo "→ Running dataset generation in datasets_regression..."
echo ""
cd datasets_regression

# Create a temporary venv for datasets_regression if needed
if [ ! -d ".venv-benchmark" ]; then
    uv venv --python 3.12 .venv-benchmark
fi

source .venv-benchmark/bin/activate
uv pip install pandas numpy scikit-learn ucimlrepo openpyxl kaggle beautifulsoup4 requests

# Run the dataset setup
python scripts/setup_datasets.py

# Deactivate the benchmark venv
deactivate

# Go back to project root
cd "$PROJECT_ROOT"

# Create datasets directory if it doesn't exist
mkdir -p datasets

# Move the generated dataset_combined to datasets/
echo ""
echo "→ Moving generated datasets to datasets/dataset_combined..."
if [ -d "datasets_regression/dataset_combined" ]; then
    # Remove old dataset_combined if it exists
    if [ -d "datasets/dataset_combined" ]; then
        echo "  Removing old datasets/dataset_combined..."
        rm -rf datasets/dataset_combined
    fi
    
    # Move the new dataset_combined
    mv datasets_regression/dataset_combined datasets/
    echo "  ✓ Datasets moved successfully"
else
    echo "  ⚠ Warning: datasets_regression/dataset_combined was not created"
    exit 1
fi

echo ""
echo "╔═══════════════════════════════════════════════════════════════════╗"
echo "║                      SETUP COMPLETE                               ║"
echo "╚═══════════════════════════════════════════════════════════════════╝"
echo ""
echo "✓ All datasets are ready in: datasets/dataset_combined/"
echo ""
