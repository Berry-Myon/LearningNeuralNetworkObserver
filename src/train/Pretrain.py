from model.Neural_Architecture import NN_Small_Point, NN_Large_Point
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import os

# Project root directory (parent of src)
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_weights_dir = os.path.join(_project_root, 'Weights')

def one_step_loss_function(eta_error, v_error, A_epsilon, P, var_epsilon, rho, M, epsilon):
    eta_error = eta_error.reshape(4 + 4, 1)
    v_error = v_error.reshape(4 + 4, 1)
    one_step_loss = (eta_error.T @ (A_epsilon.T @ P + P @ A_epsilon + rho * P) @ eta_error / epsilon
                     + 2 * (v_error.T @ P @ eta_error) / epsilon
                     + 2 * M * np.linalg.norm(eta_error.T @ P)
                     + var_epsilon)
    return one_step_loss[0, 0]

class Pretrain:
    def __init__(self, a, c, epsilon, lr, P, epochs=50000, robust=10.0):
        self.A = torch.tensor(a)
        self.C = torch.tensor(c)
        self.epsilon = epsilon
        self.input_size = len(a)
        self.output_size = len(c)
        self.A_epsilon = torch.cat([
                                        torch.cat([epsilon * self.A, torch.eye(self.input_size, dtype=torch.float32)], dim=1),
                                        torch.cat([torch.zeros(self.input_size, self.input_size * 2, dtype=torch.float32)], dim=1)
                                    ], dim=0)
        self.C_epsilon = torch.cat([self.C, torch.zeros(self.output_size, self.input_size, dtype=torch.float32)], dim=1)
        self.lr = lr
        self.P = P
        self.epochs = epochs
        self.robust = robust
        self.model = None
        self.weight_len = 6
        self.optimizer = None

    def init_model(self, model=""):
        if model == "Train_Small_NN":
            self.model = NN_Small_Point(self.input_size, self.output_size, self.P)
        else:
            self.model = NN_Large_Point(self.input_size, self.output_size, self.P)
            self.weight_len = 10
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def pre_train(self):
        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            errors = torch.tensor(np.random.normal(0, 1, size=(5, self.input_size * 2)), dtype=torch.float32)
            total_loss = torch.tensor(0.0, dtype=torch.float32)
            weight_loss = torch.tensor(0.0, dtype=torch.float32)
            for error in errors:
                eta_error = torch.tensor(list(error[:self.input_size] / self.epsilon) + list(error[self.input_size:]), dtype=torch.float32)
                L = self.model.get_paras_L()
                P = self.model.get_paras_P()
                v_error = self.model(eta_error, self.C_epsilon)
                total_loss += max(
                    one_step_loss_function(eta_error, v_error, self.A_epsilon, P, self.robust, rho=0.1, M=0.5, epsilon=self.epsilon),
                    0)
                for ws in self.model.get_paras():
                    weight_loss += torch.norm(ws)
                weight_loss += torch.norm(L)
            total_loss /= len(errors)
            total_loss += 0.01 * weight_loss
            total_loss.backward()
            self.optimizer.step()
            if epoch % (self.epochs // 20) == 0:
                print(f'Epoch {epoch}/{self.epochs}, Loss: {total_loss.item()}')
        for i in range(1, self.weight_len):
            weight = getattr(self.model, f'weights{i}').detach().numpy()
            df = pd.DataFrame(weight)
            df.to_csv(os.path.join(_weights_dir, f'Weight_{i}.csv'), index=False, header=False)
        # For large NN, also save weights0 (the L matrix)
        if self.weight_len == 10:
            weight0 = getattr(self.model, 'weights0').detach().numpy()
            df = pd.DataFrame(weight0)
            df.to_csv(os.path.join(_weights_dir, 'Weight_0.csv'), index=False, header=False)
