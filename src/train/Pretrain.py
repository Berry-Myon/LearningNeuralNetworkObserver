from model.Neural_Architecture import NN_Small_Point, NN_Large_Point
from utils.checkpoints import save_checkpoint
import torch
import torch.optim as optim
import numpy as np

def one_step_loss_function(eta_error, v_error, A_epsilon, P, var_epsilon, rho, M, epsilon):
    eta_error = eta_error.reshape(4 + 4, 1)
    v_error = v_error.reshape(4 + 4, 1)
    one_step_loss = (eta_error.T @ (A_epsilon.T @ P + P @ A_epsilon + rho * P) @ eta_error / epsilon
                     + 2 * (v_error.T @ P @ eta_error) / epsilon
                     + 2 * M * np.linalg.norm(eta_error.T @ P)
                     + var_epsilon)
    return one_step_loss[0, 0]


def positive_part(loss):
    return torch.maximum(loss, torch.zeros((), dtype=loss.dtype, device=loss.device))

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
        self.model_name = None
        self.weight_len = 6
        self.optimizer = None

    def init_model(self, model=""):
        self.model_name = model
        if model == "Train_Small_NN":
            self.model = NN_Small_Point(self.input_size, self.output_size, self.P)
        else:
            self.model = NN_Large_Point(self.input_size, self.output_size, self.P)
            self.weight_len = 10
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def pre_train(self):
        total_loss = None
        reported_loss = None
        last_epoch = -1
        for epoch in range(self.epochs):
            last_epoch = epoch
            self.optimizer.zero_grad()
            errors = torch.tensor(np.random.normal(0, 1, size=(5, self.input_size * 2)), dtype=torch.float32)
            total_loss = torch.tensor(0.0, dtype=torch.float32)
            reported_loss = torch.tensor(0.0, dtype=torch.float32)
            weight_loss = torch.tensor(0.0, dtype=torch.float32)
            for error in errors:
                eta_error = torch.tensor(list(error[:self.input_size] / self.epsilon) + list(error[self.input_size:]), dtype=torch.float32)
                L = self.model.get_paras_L()
                P = self.model.get_paras_P()
                v_error = self.model(eta_error, self.C_epsilon)
                corrected_step_loss = one_step_loss_function(
                    eta_error, v_error, self.A_epsilon, P, self.robust, rho=0.1, M=0.5, epsilon=self.epsilon
                )
                raw_step_loss = one_step_loss_function(
                    eta_error, v_error, self.A_epsilon, P, self.robust, rho=0.0, M=0.5, epsilon=self.epsilon
                )
                total_loss += positive_part(corrected_step_loss)
                reported_loss += positive_part(raw_step_loss)
                for ws in self.model.get_paras():
                    weight_loss += torch.norm(ws)
                weight_loss += torch.norm(L)
            total_loss /= len(errors)
            reported_loss /= len(errors)
            total_loss += 0.01 * weight_loss
            reported_loss += 0.01 * weight_loss
            total_loss.backward()
            self.optimizer.step()
            if epoch % (self.epochs // 20) == 0:
                print(f'Epoch {epoch}/{self.epochs}, Loss: {reported_loss.item()}')
        checkpoint_path = save_checkpoint(
            self.model_name,
            'pretrain',
            self.model,
            optimizer=self.optimizer,
            epoch=last_epoch,
            loss=reported_loss,
        )
        print(f'Pretrain checkpoint saved to {checkpoint_path}')
