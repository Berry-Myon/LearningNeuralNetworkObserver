import os
import numpy as np
import matlab.engine
import torch

from utils.checkpoints import load_named_checkpoint, save_named_checkpoint

eng = matlab.engine.start_matlab()

# Add the utils folder to MATLAB path so it can find .m files
_utils_dir = os.path.dirname(os.path.abspath(__file__))
eng.addpath(_utils_dir)

class Solver:
    def __init__(self, a, b, c, poles, max_iter=10, epsilon=0.01):
        self.A = a
        self.B = b
        self.C = c
        self.L = None
        self.poles = poles
        self.max_iter = max_iter
        self.epsilon = epsilon

    def solve_short_cut(self):
        checkpoint = load_named_checkpoint('solver_linear')
        if checkpoint is not None:
            self.L = checkpoint['L'].detach().cpu().numpy().astype(np.float32)
            return

        self.L = np.array(eng.Pole_Placement(self.A, self.B, self.C, self.poles), dtype=np.float32)
        save_named_checkpoint(
            'solver_linear',
            {
                'solver_type': 'Solver_Linear',
                'L': torch.tensor(self.L, dtype=torch.float32),
                'poles': self.poles,
            },
        )

    def solver(self, nodes):
        self.solve_short_cut()
        connected_nodes = [len(self.A)] + list(nodes) + [len(self.C[0])]
        weights, time_use, flag = eng.Solver(self.A, self.B, self.C, self.L, self.epsilon, nodes, self.max_iter, nargout=3)
        if flag:
            weights = weights[0]
            layer_weights = {}
            for i, w in enumerate(weights):
                weight = np.ones((connected_nodes[i + 1], connected_nodes[i])) * w
                layer_weights[f'weights{i + 1}'] = torch.tensor(weight, dtype=torch.float32)
            save_named_checkpoint(
                'solver_nn',
                {
                    'solver_type': 'Solver_NN',
                    'L': torch.tensor(self.L, dtype=torch.float32),
                    'layer_weights': layer_weights,
                    'nodes': list(nodes),
                    'time_use': float(time_use),
                },
            )
            print(f"LMI Solved successfully, using {time_use} seconds.")
        else:
            print("Failed to solve!")
