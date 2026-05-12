"""
depth_saturation.py
====================
Empirical investigation of loss saturation with increasing MLP depth.

Motivation:
    There is no principled rule for choosing MLP depth vs width.
    A network with 100 layers of 100 neurons and one with 10 layers
    of 1000 neurons have similar parameter counts but behave differently.

    This experiment asks: as we increase depth while holding parameter
    count approximately constant, does accuracy keep improving?

    Observation: loss improvement follows a diminishing returns curve —
    steep early improvement that gradually flattens, resembling an
    inverted diode current curve. Beyond a certain depth, adding more
    layers yields negligible gain while increasing training time and
    risk of vanishing gradients.

    This saturation behaviour motivated investigating KANs as an
    architecture with theoretically grounded function decomposition
    rather than arbitrary depth-width tradeoffs.
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

DEPTHS        = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20]  # Number of hidden layers
NEURONS       = 64       # Neurons per hidden layer (fixed)
INPUT_DIM     = 10       # Input features
OUTPUT_DIM    = 1
N_SAMPLES     = 500      # Training samples
EPOCHS        = 200      # Training epochs per model
LR            = 0.01
SEED          = 42


# ─────────────────────────────────────────────────────────────────────────────
# Data — non-linear function of first 3 features, rest are noise
# ─────────────────────────────────────────────────────────────────────────────

torch.manual_seed(SEED)
X = torch.rand((N_SAMPLES, INPUT_DIM))
y = torch.exp(
    torch.sin(X[:, 0]) + X[:, 1]**2 + torch.cos(X[:, 2])
).reshape(-1, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Dynamic MLP builder
# ─────────────────────────────────────────────────────────────────────────────

class DynamicMLP(nn.Module):
    """MLP with configurable depth and fixed width per layer."""
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers):
        super().__init__()
        layers = []

        # Input layer
        layers.extend([nn.Linear(input_dim, hidden_dim), nn.ReLU()])

        # Hidden layers
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])

        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# Training function
# ─────────────────────────────────────────────────────────────────────────────

def train_model(model, X, y, epochs=EPOCHS, lr=LR):
    """Train model and return final loss and training time."""
    optimiser  = torch.optim.Adam(model.parameters(), lr=lr)
    criterion  = nn.MSELoss()
    loss_curve = []

    t0 = time.time()
    for epoch in range(epochs):
        optimiser.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimiser.step()
        loss_curve.append(loss.item())

    training_time = time.time() - t0
    return loss_curve, training_time


# ─────────────────────────────────────────────────────────────────────────────
# Experiment — sweep across depths
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 55)
print("  DEPTH SATURATION EXPERIMENT")
print("  Fixed width per layer:", NEURONS, "neurons")
print("  Sweeping depths:", DEPTHS)
print("=" * 55)

results = []

for depth in DEPTHS:
    torch.manual_seed(SEED)
    model      = DynamicMLP(INPUT_DIM, NEURONS, OUTPUT_DIM, depth)
    n_params   = sum(p.numel() for p in model.parameters())

    loss_curve, t = train_model(model, X, y)
    final_loss    = loss_curve[-1]

    results.append({
        'depth':      depth,
        'final_loss': final_loss,
        'time':       t,
        'n_params':   n_params,
        'loss_curve': loss_curve,
    })

    print(f"  Depth {depth:>3} layers | "
          f"Params: {n_params:>7,} | "
          f"Final Loss: {final_loss:.6f} | "
          f"Time: {t:.2f}s")

print("=" * 55)


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(
    "MLP Depth Saturation — Loss vs Depth, Training Curves, Training Time",
    fontsize=13, fontweight='bold'
)

# ── Plot 1: Final loss vs depth (the inverted diode curve) ───────────────────
depths      = [r['depth'] for r in results]
final_losses = [r['final_loss'] for r in results]

axes[0].plot(depths, final_losses, 'b-o', linewidth=2, markersize=6)
axes[0].set_xlabel("Number of Hidden Layers (Depth)", fontsize=11)
axes[0].set_ylabel("Final Training Loss (MSE)", fontsize=11)
axes[0].set_title("Loss Saturation with Depth\n(Inverted Diode Curve)", fontsize=11)
axes[0].grid(True, alpha=0.3)
axes[0].annotate(
    "Saturation region",
    xy=(depths[-3], final_losses[-3]),
    xytext=(depths[-5], final_losses[-5] + 0.01),
    arrowprops=dict(arrowstyle='->', color='red'),
    color='red', fontsize=9
)

# ── Plot 2: Training loss curves per depth ───────────────────────────────────
cmap = plt.cm.viridis
for i, r in enumerate(results):
    color = cmap(i / len(results))
    axes[1].plot(
        r['loss_curve'],
        color=color,
        alpha=0.8,
        label=f"depth={r['depth']}"
    )
axes[1].set_xlabel("Training Epoch", fontsize=11)
axes[1].set_ylabel("MSE Loss", fontsize=11)
axes[1].set_title("Training Curves by Depth", fontsize=11)
axes[1].legend(fontsize=7, ncol=2)
axes[1].grid(True, alpha=0.3)

# ── Plot 3: Training time vs depth ───────────────────────────────────────────
times = [r['time'] for r in results]

axes[2].bar(depths, times, color='steelblue', alpha=0.7, width=0.6)
axes[2].set_xlabel("Number of Hidden Layers (Depth)", fontsize=11)
axes[2].set_ylabel("Training Time (seconds)", fontsize=11)
axes[2].set_title("Training Time vs Depth\n(Cost of Going Deeper)", fontsize=11)
axes[2].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig("depth_saturation_curves.png", dpi=150, bbox_inches='tight')
plt.show()

print("\nPlot saved as: depth_saturation_curves.png")
print("\nKey finding:")
print(f"  Depth 1  → Loss: {results[0]['final_loss']:.6f}")
print(f"  Depth 5  → Loss: {results[4]['final_loss']:.6f}")
print(f"  Depth 20 → Loss: {results[-1]['final_loss']:.6f}")
print(f"\n  Improvement from depth 1→5:  "
      f"{((results[0]['final_loss'] - results[4]['final_loss'])/results[0]['final_loss']*100):.1f}%")
print(f"  Improvement from depth 5→20: "
      f"{((results[4]['final_loss'] - results[-1]['final_loss'])/results[4]['final_loss']*100):.1f}%")
print("\n  If improvement 1→5 >> improvement 5→20: saturation confirmed.")
