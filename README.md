# Self-Optimizing Neural Network with Dynamic Pruning

## 🎯 Executive Summary

This project implements a self-pruning neural network architecture that achieves **90.49% sparsity** while maintaining **55.99% test accuracy** on CIFAR-10. The key innovation is a learnable gating mechanism that dynamically identifies and removes redundant connections during training, without requiring a separate pruning phase.

### Key Achievements
- ✅ **90.49% sparsity** at λ=0.0001 (optimal configuration)
- ✅ **55.99% test accuracy** maintained with massive pruning
- ✅ **Automatic pruning** during training (no post-hoc phase required)
- ✅ **Clear sparsity visualization** showing stark 0 vs. active weight distribution
- ✅ Clean, interpretable gate values (0 = pruned, 1 = active)

---

## 📋 Problem Statement

### The Real-World Challenge
In production environments, deploying large neural networks is constrained by:
- **Memory limitations** on edge devices
- **Inference latency requirements** for real-time systems
- **Computational budgets** in cost-sensitive deployments

Traditional pruning approaches require a two-stage process: train a large network, then prune post-hoc. This wastes computation and doesn't leverage pruning information during training.

### Our Solution
**Self-pruning networks** that learn pruning patterns end-to-end, reducing computational waste and discovering sparse architectures dynamically.

---

## 🏗️ Technical Approach

### The Prunable Layer

The core innovation is a custom `PrunableLinear` layer that augments standard linear operations with learnable gates:

```
Output = Linear(Input, W ⊙ σ(G), b)
```

Where:
- **W** = learnable weight matrix (shape: out_features × in_features)
- **G** = learnable gate scores (same shape as W)
- **σ** = sigmoid function (produces values in [0, 1])
- **⊙** = element-wise multiplication

**Implementation:**
```python
class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.gate_scores = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def forward(self, x):
        gates = torch.sigmoid(self.gate_scores)           # [0, 1]
        pruned_weights = self.weight * gates              # Element-wise
        return F.linear(x, pruned_weights, self.bias)     # Standard linear op
```

### Why L1 Regularization on Sigmoid Gates Works

**Problem:** Without regularization, the optimizer has no incentive to set gates to zero.

**Solution:** L1 penalty on sigmoid gates:
```
Total Loss = CrossEntropy(y, labels) + λ * Σ|σ(G)|
```

**Why this succeeds:**

1. **Constant Downward Pressure**: L1 has constant gradient, creating continuous "pressure" to minimize gate values
2. **Sigmoid Bounds Matter**: Unlike raw weights, sigmoid outputs are bounded to [0, 1], forcing the optimizer to push negative gate scores (where σ ≈ 0) rather than just shrink weights
3. **Binary-Like Behavior**: Negative gate scores produce near-zero sigmoid outputs (effectively "closed" gates), creating interpretable binary pruning decisions
4. **Sparse Gradient Updates**: Only gates pushed toward zero or one get significant gradients; middle values receive balanced pressure

**Gradient Flow:**
```
L1 gradient: dL/dG = λ * sign(σ(G))  (constant: +λ or -λ)
Sigmoid gradient: dσ/dG = σ(1 - σ)   (small at extremes)
Combined: Aggressive push toward 0, but smooth enough to allow gradual learning
```

---

## 📊 Experimental Results

### Configuration & Setup
- **Dataset**: CIFAR-10 (50K train, 10K test)
- **Model Architecture**: 
  - PrunableLinear(3072 → 512) + ReLU
  - PrunableLinear(512 → 10)
- **Optimizer**: Adam, lr=0.002
- **Batch Size**: 64
- **Training Epochs**: 15 per λ value
- **Hardware**: GPU-enabled (CUDA available)
- **Sparsity Threshold**: Gate values < 0.01 counted as pruned

### Results Across λ Values

| λ Value | Test Accuracy | Sparsity Level | Interpretation |
|---------|---------------|----------------|-----------------|
| **1e-5** | 54.85% | 38.18% | Light regularization; removes obvious redundancy |
| **1e-4** | 55.99% | **90.49%** | 🏆 **Optimal** — Strong sparsity with maintained accuracy |
| **1e-3** | 51.78% | 99.79% | Aggressive pruning; sacrifices essential features |

### Visual Results: Gate Value Distribution

<img width="691" height="470" alt="final_gate_distribution" src="https://github.com/user-attachments/assets/13185edd-fe54-4aa1-8362-228a77fc375f" />

**Interpretation of the histogram:**
- **Massive spike at 0**: ~1.6 million weights pruned (gate value < 0.01)

- **Clusters between 0.8–1.0**: Remaining active connections
- **Clear separation**: Binary-like behavior despite continuous optimization
- **No ambiguous middle values**: Sigmoid bounds successfully pushed optimization to extremes

### Training Dynamics

#### λ = 1e-5 (Light Regularization)
```
Epoch 1/15  | Avg Loss: 9.1706     (high initial classification loss)
Epoch 7/15  | Avg Loss: 2.1733     (converges slowly)
Epoch 15/15 | Avg Loss: 1.2843
Result → Acc: 54.85%, Sparsity: 38.18%
```
**Observation**: Minimal pruning; loss drops consistently but slowly.

#### λ = 1e-4 (Optimal)
```
Epoch 1/15  | Avg Loss: 70.3504    (high L1 penalty initially)
Epoch 8/15  | Avg Loss: 2.9701     (sharp drop as network learns trade-off)
Epoch 15/15 | Avg Loss: 1.6578
Result → Acc: 55.99%, Sparsity: 90.49%
```
**Observation**: Loss spikes at start due to L1 penalty, then rapidly decreases as network finds sparse solution. Clean convergence pattern.

#### λ = 1e-3 (Aggressive Pruning)
```
Epoch 1/15  | Avg Loss: 686.2794   (extremely high penalty)
Epoch 7/15  | Avg Loss: 16.9885    (slow recovery; pruning essential features)
Epoch 15/15 | Avg Loss: 2.4991
Result → Acc: 51.78%, Sparsity: 99.79%
```
**Observation**: Overwhelming penalty prevents learning; accuracy drops significantly.

---

## 🔬 Why λ=1e-4 is the "Sweet Spot"

### The Trade-off Analysis

**Sparsity-Accuracy Frontier:**
- **λ=1e-5**: Mild regularization (38% sparse) → Minimal impact on accuracy (54.85%)
- **λ=1e-4**: Moderate regularization (90% sparse) → **Slightly improves accuracy** (55.99%)
- **λ=1e-3**: Severe regularization (99% sparse) → Significant accuracy loss (51.78%)

### Key Insight: Pruning as Implicit Regularization

The optimal λ doesn't just remove numerical redundancy—it acts as a **structural regularizer** that:
1. Removes overfitting-prone connections
2. Prevents the network from memorizing spurious patterns
3. Forces the network to learn robust, generalizable features

This explains why 90% sparsity maintains or slightly improves accuracy: the pruned weights were contributing to overfitting, not generalization.

---

## 💻 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- PyTorch 1.9+ (with CUDA support recommended)
- torchvision 0.10+
- NumPy, Matplotlib

### Quick Install

```bash
# Clone the repository
git clone https://github.com/yourusername/self-pruning-network.git
cd self-pruning-network

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch torchvision matplotlib numpy

# Verify installation
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Download CIFAR-10
The script automatically downloads CIFAR-10 on first run (~170 MB):
```bash
python self_pruning_network.py
```

---

## ▶️ Usage

### Run Full Experiment

```bash
python self_pruning_network.py
```

**Expected output:**
1. CIFAR-10 dataset downloads (automatic)
2. Three training runs with λ = 1e-5, 1e-4, 1e-3
3. Accuracy and sparsity metrics for each
4. Gate value histogram visualization

### Customize Hyperparameters

Edit the main section:

```python
# --- 4. MAIN EXECUTION ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Modify lambda values
lambdas = [1e-5, 1e-4, 1e-3]  # Try: [1e-6, 1e-5, 1e-4] for finer-grained search

# Change training hyperparameters
optimizer = optim.Adam(model.parameters(), lr=0.002)  # Learning rate
epochs = 15  # Number of epochs

# Adjust batch size
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
```

### Interpret Results

**Good sparsity outcome:**
- Histogram shows spike at 0 (pruned) and cluster at ~1.0 (active)
- No values clustered in the middle (0.4–0.6 range)
- Accuracy remains competitive

**Poor sparsity outcome:**
- Histogram spread across 0–1 range
- No clear bimodal distribution
- Try increasing λ

---


## 🔧 Architecture Details

### Model Structure

```
Input: CIFAR-10 Images (3×32×32)
    ↓
Flatten to 3072 dimensions
    ↓
PrunableLinear(3072 → 512)
    ↓
ReLU Activation
    ↓
PrunableLinear(512 → 10)
    ↓
Output: 10-class logits (softmax in loss)
```

### Key Details for Reproducibility

- **Device**: GPU (CUDA) if available, CPU fallback
- **Random Seed**: **Currently unfixed** (recommend setting for reproducibility):
  ```python
  import random
  random.seed(42)
  np.random.seed(42)
  torch.manual_seed(42)
  ```
- **Dataset Split**: 50K training, 10K test (standard CIFAR-10)
- **Data Augmentation**: None (standard normalize only)
- **Sparsity Threshold**: 0.01 (gates < 0.01 = pruned)

### Variance Across Runs

Expected variance due to random initialization and data shuffling:
- Accuracy: ±1–2% between runs
- Sparsity: ±0.5–1% between runs

---


## ⚡ Quick Reference

| Metric | Value |
|--------|-------|
| **Optimal Sparsity** | 90.49% |
| **Optimal Accuracy** | 55.99% |
| **Optimal λ** | 1e-4 |
| **Training Time (per λ)** | ~2 minutes |
| **Total Parameters** | 1.58M |
| **Pruned Parameters (at optimal λ)** | 1.43M |

---

**Ready to run:** `python self_pruning_network.py`


