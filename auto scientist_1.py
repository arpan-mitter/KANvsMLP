import pandas as pd
import torch
import argparse
from kan import KAN
import os

def run_pipeline(csv_path, target_column):
    # 1. Load Data
    df = pd.read_csv(csv_path)
    X = torch.tensor(df.drop(columns=[target_column]).values, dtype=torch.float32)
    y = torch.tensor(df[target_column].values, dtype=torch.float32).reshape(-1, 1)
    
    input_dim = X.shape[1]
    print(f"--- Loaded {csv_path} with {input_dim} features ---")

    # 2. KAN Scouting (Feature Selection & Logic Discovery)
    # We use a small width [input, 2, 1] to force a bottleneck
    model = KAN(width=[input_dim, 2, 1], grid=5, k=3)
    
    print("Step 1: Training KAN (Sparsifying)...")
    model.train({'train_input': X, 'train_label': y}, steps=40, lamb=0.02, lamb_entropy=2.0)
    
    print("Step 2: Pruning non-contributing features...")
    model = model.prune()
    
    # 3. Symbolic Extraction
    print("Step 3: Fitting symbolic formulas...")
    lib = ['x', 'x^2', 'x^3', 'sin', 'cos', 'exp', 'log', 'sqrt']
    model.auto_symbolic(lib=lib)
    
    # 4. Final Output
    formula = model.symbolic_formula()[0][0]
    
    print("\n" + "="*30)
    print("SUCCESS: DISCOVERED LOGIC")
    print("="*30)
    print(f"Formula: y = {formula}")
    print("="*30)
    
    # Save the model for future use
    model_name = os.path.basename(csv_path).replace('.csv', '_model.pth')
    model.save(model_name)
    print(f"Model saved as: {model_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Discover the math in your CSV.")
    parser.add_argument("file", help="Path to the CSV file")
    parser.add_argument("--target", required=True, help="Name of the target column")
    
    args = parser.parse_args()
    run_pipeline(args.file, args.target)
