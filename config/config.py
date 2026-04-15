import numpy as np
import torch

# System State Space
A = np.array([[-0.4272e-1, -0.8541e+1, -0.4451e+0, -0.3216e+2],
              [-0.7881e-3, -0.5291e+0, +0.9896e+0, +0.1439e-9],
              [+0.4010e-3, +0.3542e+1, -0.2228e+0, +0.6150e-8],
              [0., 0., 1., 0.]], dtype=np.float32)

B = np.array([[-0.3385e-1, -0.9386e-1, +0.4888e-2],
              [-0.1028e-2, -0.1297e-2, -0.4054e-3],
              [+0.2718e-1, -0.5744e-2, -0.1351e-1],
              [0., 0., 0.]], dtype=np.float32)

C = np.array([[1., 0., 0., 0.],
              [0., 57.3, 0., 0.],
              [0., 0., 57.3, 0.],
              [0., 0., 0., 57.3],
              [0.007063, 4.567, 0.09867, -0.3809e-4]], dtype=np.float32)
epsilon = 0.01

# Type of Solver
solver_type = "Train_Small_NN"  # "Solver_Linear" or "Solver_NN" or "Train_Small_NN" or "Train_Large_NN"

# Solver Parameters
poles = np.array([-0.5, -1+0.2j, -1-0.2j, -1.5+0.5j, -1.5-0.5j, -2+0.2j, -2-0.2j, -2.5])
nodes = np.array([3, 3, 3], dtype=np.int64)
connected_nodes = [len(A)] + list(nodes) + [len(C)]
max_iter = 10

# Training Parameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
P = torch.tensor(np.diag((10, 1, 1, 1, 1, 1, 1, 1)), dtype=torch.float32) * 0.1
epochs_pretrain = 50000
epochs_tuning = 10000
learning_rate = 0.01
