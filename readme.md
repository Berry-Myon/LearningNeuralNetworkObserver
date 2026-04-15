# Neural Network Solver for Control Systems

A PyTorch and MATLAB based project for observer design and stabilization of uncertain dynamic systems.

## Project Structure

```text
Code/
|-- config/
|   |-- __init__.py
|   `-- config.py
|-- scripts/
|   |-- run_solver_linear.sh
|   |-- run_solver_nn.sh
|   |-- run_train_small.sh
|   `-- run_train_large.sh
|-- src/
|   |-- model/
|   |   `-- Neural_Architecture.py
|   |-- train/
|   |   |-- Pretrain.py
|   |   |-- Tuning.py
|   |   `-- Two_Stage_Training.py
|   `-- utils/
|       |-- Solver.py
|       |-- checkpoints.py
|       |-- Pole_Placement.m
|       |-- Solver.m
|       |-- LMI_verify.m
|       |-- Hurwitz.m
|       `-- NN_paras.m
|-- Checkpoints/
|-- main.py
`-- readme.md
```

## Requirements

- Python 3.8+
- PyTorch
- NumPy
- Pandas
- MATLAB
- MATLAB Engine for Python

## Installation

Install Python dependencies:

```bash
pip install torch numpy pandas
```

Install MATLAB Engine for Python if you want to use `Solver_Linear` or `Solver_NN`:

```bash
cd $MATLABROOT/extern/engines/python
python setup.py install
```

## Usage

Run with the solver type configured in `config/config.py`:

```bash
python main.py
```

Override the solver type from the command line:

```bash
python main.py --solver_type Solver_Linear
python main.py --solver_type Solver_NN
python main.py --solver_type Train_Small_NN
python main.py --solver_type Train_Large_NN
```

## Configuration

Edit `config/config.py` to set the system matrices and hyperparameters.

### System Matrices

```python
A = np.array([...])
B = np.array([...])
C = np.array([...])
epsilon = 0.01
```

### Solver Parameters

```python
poles = np.array([...])
nodes = np.array([3, 3, 3])
max_iter = 10
```

### Training Parameters

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
P = torch.tensor(...)
learning_rate = 0.01
epochs_pretrain = 50000
epochs_tuning = 10000
```

### Solver Type

```python
solver_type = "Solver_Linear"
```

Available options:

- `Solver_Linear`
- `Solver_NN`
- `Train_Small_NN`
- `Train_Large_NN`

## Methods

### `Solver_Linear`

Computes the observer gain matrix `L` with MATLAB pole placement.

### `Solver_NN`

1. Computes the observer gain matrix `L`
2. Uses MATLAB LMI solving to obtain neural network layer weights

### `Train_Small_NN` and `Train_Large_NN`

Training uses two stages:

1. `Pretrain.py`
   - trains the pointwise neural network with random perturbation samples
   - saves a pretraining checkpoint
2. `Tuning.py`
   - loads the pretraining checkpoint
   - optimizes the LMI related objective
   - saves a tuning checkpoint

## Architectures

| Model | Layers | Hidden Units |
|------|------|------|
| Small | 4 | 32 -> 64 -> 32 -> 32 |
| Large | 8 | 32 -> 64 -> 64 -> 128 -> 128 -> 64 -> 64 -> 32 |

## Outputs

Generated checkpoints are stored in `Checkpoints/`.

Typical files include:

| File | Description |
|------|------|
| `train_small_nn_pretrain.pt` | Small model pretraining result |
| `train_small_nn_tuning.pt` | Small model tuning result |
| `train_large_nn_pretrain.pt` | Large model pretraining result |
| `train_large_nn_tuning.pt` | Large model tuning result |
| `solver_linear.pt` | Pole placement result |
| `solver_nn.pt` | MATLAB solver result |

Training checkpoints store the model weights together with optimizer state, epoch, and loss.

## License

Research and educational use.
