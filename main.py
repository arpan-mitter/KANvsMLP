import torch
from kan_model import train_kan
from mlp_model import train_mlp

# 1. Generate High-Dimensional Synthetic Data
# 100 features, 1000 samples
X = torch.rand((1000, 100)) 
# Target is a non-linear combination of features
y = torch.exp(torch.sin(X[:, 0]) + X[:, 1]**2).reshape(-1, 1)

# 2. Train and Compare
print("--- Training MLP ---")
mlp_model = train_mlp(X, y, 100, 1)

print("\n--- Training KAN ---")
kan_model = train_kan(X, y, 100, 1)

# 3. Accuracy Check (RMSE)
with torch.no_grad():
    mlp_preds = mlp_model(X)
    kan_preds = kan_model(X)
    
    mlp_rmse = torch.sqrt(torch.mean((mlp_preds - y)**2))
    kan_rmse = torch.sqrt(torch.mean((kan_preds - y)**2))
    
    print(f"\nMLP RMSE: {mlp_rmse.item():.4f}")
    print(f"KAN RMSE: {kan_rmse.item():.4f}")
