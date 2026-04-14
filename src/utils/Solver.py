import os
import numpy as np
import pandas as pd
import matlab.engine

eng = matlab.engine.start_matlab()

# Add the utils folder to MATLAB path so it can find .m files
_utils_dir = os.path.dirname(os.path.abspath(__file__))
eng.addpath(_utils_dir)

# Project root directory (parent of src/utils)
_project_root = os.path.dirname(os.path.dirname(_utils_dir))
_weights_dir = os.path.join(_project_root, 'Weights')

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
        L_path = os.path.join(_weights_dir, 'L.csv')
        if not os.path.exists(L_path):
            eng.Pole_Placement(self.A, self.B, self.C, self.poles)
        self.L = np.array(pd.read_csv(L_path), dtype=np.float32)

    def solver(self, nodes):
        self.solve_short_cut()
        connected_nodes = [len(self.A)] + list(nodes) + [len(self.C[0])]
        weights, time_use, flag = eng.Solver(self.A, self.B, self.C, self.L, self.epsilon, nodes, self.max_iter, nargout=3)
        if flag:
            weights = weights[0]
            for i, w in enumerate(weights):
                weight = np.ones((connected_nodes[i + 1], connected_nodes[i])) * w
                df = pd.DataFrame(weight)
                df.to_csv(os.path.join(_weights_dir, f'Weight_{i + 1}.csv'), index=False, header=False)
            print(f"LMI Solved successfully, using {time_use} seconds.")
        else:
            print("Failed to solve!")
