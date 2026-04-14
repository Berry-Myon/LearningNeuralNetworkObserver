# Neural Network Solver for Control Systems

A PyTorch-based neural network solver for control system stabilization using pole placement and LMI (Linear Matrix Inequality) optimization.

## Project Structure

```
Code/
├── config/                     # Configuration files
│   ├── __init__.py
│   └── config.py               # System matrices and hyperparameters
├── src/                        # Source code
│   ├── __init__.py
│   ├── model/                  # Neural network architectures
│   │   ├── __init__.py
│   │   └── Neural_Architecture.py
│   ├── train/                  # Training methods
│   │   ├── __init__.py
│   │   ├── Pretrain.py         # Pre-training stage
│   │   ├── Tuning.py           # Fine-tuning stage
│   │   └── Two_Stage_Training.py
│   └── utils/                  # Utility modules (MATLAB .m files)
│       ├── __init__.py
│       ├── Solver.py           # Python interface to MATLAB
│       ├── Pole_Placement.m    # Pole placement algorithm
│       ├── Solver.m            # LMI-based NN solver
│       ├── LMI_verify.m        # LMI verification
│       ├── Hurwitz.m           # Hurwitz stability check
│       └── NN_paras.m          # NN parameter computation
├── Weights/                    # Saved weights and outputs (auto-created)
├── main.py                     # Entry point
└── readme.md                   # This file
```

## Installation

### Requirements

- Python 3.8+
- PyTorch
- NumPy
- Pandas
- MATLAB with MATLAB Engine for Python

### Install Python Dependencies

```bash
pip install torch numpy pandas
```

### Install MATLAB Engine (Optional - required for Solver_Linear/Solver_NN)

```bash
cd $MATLABROOT/extern/engines/python
python setup.py install
```

**Note:** The MATLAB `.m` files are located in `src/utils/`. The solver automatically adds this directory to the MATLAB path at runtime.

## Usage

### Command Line

```bash
# Run with default configuration (solver_type from config/config.py)
python main.py

# Override solver type via command line
python main.py --solver_type Solver_Linear
python main.py --solver_type Solver_NN
python main.py --solver_type Train_Small_NN
python main.py --solver_type Train_Large_NN
```

## Configuration

Edit `config/config.py` to customize system matrices and hyperparameters:

### System Matrices (State-Space Model)

```python
A = np.array([...])  # System matrix (4x4)
B = np.array([...])  # Input matrix (4x3)
C = np.array([...])  # Output matrix (5x4)
epsilon = 0.01       # Gain coefficient
```

### Solver Parameters

```python
poles = np.array([-0.5, -1+0.2j, -1-0.2j, ...])  # Desired closed-loop poles
nodes = np.array([3, 3, 3])                       # NN hidden layer sizes
max_iter = 10                                     # Max iterations for LMI solver
```

### Training Hyperparameters

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
P = torch.tensor(np.diag((10, 1, 1, 1, 1, 1, 1, 1)), dtype=torch.float32) * 0.1
learning_rate = 0.01
epochs_pretrain = 50000
epochs_tuning = 10000
```

### Solver Selection

```python
solver_type = "Solver_Linear"  # Options:
# "Solver_Linear"     - Linear pole placement via MATLAB
# "Solver_NN"         - Neural network with MATLAB LMI
# "Train_Small_NN"    - Two-stage training (small architecture)
# "Train_Large_NN"    - Two-stage training (large architecture)
```

## Methods

### Solver_Linear / Solver_NN

Uses MATLAB for:
1. Pole placement to compute initial gain matrix L
2. LMI optimization for neural network weights (Solver_NN only)

### Train_Small_NN / Train_Large_NN

Two-stage training process:

1. **Pre-training** (`Pretrain.py`): Robust training with random perturbations
   - Optimizes one-step Lyapunov loss
   - Initializes network weights

2. **Fine-tuning** (`Tuning.py`): Direct LMI optimization
   - Loads pre-trained weights
   - Optimizes eigenvalue constraints
   - Saves final weights to `Weights/`

**Architecture Comparison:**

| Model | Layers | Hidden Units |
|-------|--------|--------------|
| Small | 4      | 32 → 64 → 32 → 32 |
| Large | 8      | 32 → 64 → 64 → 128 → 128 → 64 → 64 → 32 |

## Output

Weights are saved to `Weights/` (directory is auto-created if it doesn't exist):

| File | Description |
|------|-------------|
| `L.csv` | Gain matrix from pole placement (Solver_Linear/Solver_NN) |
| `Weight_1.csv` ~ `Weight_5.csv` | Small network weights (4 layers) |
| `Weight_0.csv` ~ `Weight_9.csv` | Large network weights (9 layers + L matrix) |
| `T.csv` | Lagrange multiplier diagonal matrix (from Tuning) |

**Note:** For `Train_Large_NN`, `Weight_0.csv` corresponds to the output layer L matrix, while `Weight_1.csv` through `Weight_9.csv` are the hidden layer weights.

## Citing

This code implements neural network-based stabilization for linear control systems using LMI constraints and Lyapunov stability theory.

## License

Research and educational use.
