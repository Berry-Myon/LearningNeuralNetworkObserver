from model.Neural_Architecture import NN_Small_Tuning, NN_Large_Tuning
from utils.checkpoints import load_pretrained_parameters, save_checkpoint
import torch
import torch.optim as optim


def loss_function(A, T_lambda=None, weights=None, device=None):
    A_minors = torch.real(torch.linalg.eigvalsh(A))
    loss = sum(-torch.log(torch.abs(minor)) for minor in A_minors)
    penalty1 = sum(
        torch.tensor(-1e+5, device=device) * minor for minor in A_minors if minor < torch.tensor(0., device=device))
    total_loss = loss + penalty1
    if T_lambda is not None:
        penalty2 = sum(torch.tensor(-1e+5, device=device) * T for T in T_lambda if T < torch.tensor(0., device=device))
        total_loss += penalty2
        total_loss += 0.01 * torch.norm(T_lambda)
    if weights is not None:
        for w in weights:
            total_loss += 0.01 * torch.norm(w)
    return total_loss


class Tuning:
    def __init__(self, a, c, epsilon, lr, P, epochs=10000, device=None):
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
        self.model = None
        self.weight_len = 6
        self.optimizer = None
        self.device = device

    def init_model(self, model=""):
        if model == "Train_Small_NN":
            self.model = NN_Small_Tuning(self.input_size, self.output_size, self.device)
        else:
            self.model = NN_Large_Tuning(self.input_size, self.output_size, self.device)
            self.weight_len = 10
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        checkpoint_source = load_pretrained_parameters(self.model, model, self.device)
        print(f'Loaded pretrain weights from {checkpoint_source}')

    def tuning(self):
        loss = None
        reported_loss = None
        last_epoch = -1
        for epoch in range(self.epochs):
            last_epoch = epoch
            self.optimizer.zero_grad()
            A0 = self.model(self.A_epsilon, self.C_epsilon, self.P)
            if epoch >= 8000 and min(torch.real(torch.linalg.eigvals(A0.detach().cpu())).numpy()) > 0.:
                break
            A0_with_margin = A0 - torch.eye(len(A0), device=A0.device, dtype=A0.dtype) * 1e-2
            T_l = self.model.get_T_lambda()
            weights = self.model.get_paras()
            loss = loss_function(A0_with_margin, T_l, weights, device=self.device)
            reported_loss = loss_function(A0, T_l, weights, device=self.device)
            loss.backward()
            self.optimizer.step()
            if epoch % 100 == 0:
                print(f'Epoch {epoch}/{self.epochs}, Loss: {reported_loss.item()}')
        checkpoint_path = save_checkpoint(
            'Train_Small_NN' if self.weight_len == 6 else 'Train_Large_NN',
            'tuning',
            self.model,
            optimizer=self.optimizer,
            epoch=last_epoch,
            loss=reported_loss,
        )
        print(f'Tuning checkpoint saved to {checkpoint_path}')
