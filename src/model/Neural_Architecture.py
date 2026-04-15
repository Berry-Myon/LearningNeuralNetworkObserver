import numpy as np
import torch
import torch.nn as nn


def _random_parameter(shape, device=None):
    return nn.Parameter(torch.tensor(np.random.normal(0, 1, size=shape), dtype=torch.float32, device=device))


class NN_Small_Point(nn.Module):
    def __init__(self, input_size, output_size, P):
        super(NN_Small_Point, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.P = P
        self.weights1 = _random_parameter((32, output_size))
        self.weights2 = _random_parameter((64, 32))
        self.weights3 = _random_parameter((32, 64))
        self.weights4 = _random_parameter((input_size * 2, 32))
        self.weights5 = _random_parameter((input_size * 2, output_size))

    def forward(self, input_data, C):
        input_data = C @ input_data.reshape(self.input_size * 2, 1)
        mid1 = torch.mm(self.weights1, input_data)
        layer1 = torch.tanh(mid1)
        mid2 = torch.mm(self.weights2, layer1)
        layer2 = torch.tanh(mid2)
        mid3 = torch.mm(self.weights3, layer2)
        layer3 = torch.tanh(mid3)
        mid4 = torch.mm(self.weights4, layer3)
        out = mid4 + torch.mm(self.weights5, input_data)
        return out.flatten()

    def get_paras(self):
        return self.weights1, self.weights2, self.weights3, self.weights4

    def get_paras_L(self):
        return self.weights5

    def get_paras_P(self):
        return self.P


class NN_Large_Point(nn.Module):
    def __init__(self, input_size, output_size, P):
        super(NN_Large_Point, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.P = P
        self.weights1 = _random_parameter((32, output_size))
        self.weights2 = _random_parameter((64, 32))
        self.weights3 = _random_parameter((64, 64))
        self.weights4 = _random_parameter((128, 64))
        self.weights5 = _random_parameter((128, 128))
        self.weights6 = _random_parameter((64, 128))
        self.weights7 = _random_parameter((64, 64))
        self.weights8 = _random_parameter((32, 64))
        self.weights9 = _random_parameter((input_size * 2, 32))
        self.weights0 = _random_parameter((input_size * 2, output_size))

    def forward(self, input_data, C):
        input_data = C @ input_data.reshape(self.input_size * 2, 1)
        mid1 = torch.mm(self.weights1, input_data)
        layer1 = torch.tanh(mid1)
        mid2 = torch.mm(self.weights2, layer1)
        layer2 = torch.tanh(mid2)
        mid3 = torch.mm(self.weights3, layer2)
        layer3 = torch.tanh(mid3)
        mid4 = torch.mm(self.weights4, layer3)
        layer4 = torch.tanh(mid4)
        mid5 = torch.mm(self.weights5, layer4)
        layer5 = torch.tanh(mid5)
        mid6 = torch.mm(self.weights6, layer5)
        layer6 = torch.tanh(mid6)
        mid7 = torch.mm(self.weights7, layer6)
        layer7 = torch.tanh(mid7)
        mid8 = torch.mm(self.weights8, layer7)
        layer8 = torch.tanh(mid8)
        mid9 = torch.mm(self.weights9, layer8)
        out = mid9 + torch.mm(self.weights0, input_data)
        return out.flatten()

    def get_paras(self):
        return self.weights1, self.weights2, self.weights3, self.weights4, self.weights5, self.weights6, self.weights7, self.weights8, self.weights9

    def get_paras_L(self):
        return self.weights0

    def get_paras_P(self):
        return self.P



off_side_small = [0,
                  32,
                  32 + 64,
                  32 + 64 + 32]
off_side_large = [0,
                  32,
                  32 + 64,
                  32 + 64 + 64,
                  32 + 64 + 64 + 128,
                  32 + 64 + 64 + 128 + 128,
                  32 + 64 + 64 + 128 + 128 + 64,
                  32 + 64 + 64 + 128 + 128 + 64 + 64,
                  32 + 64 + 64 + 128 + 128 + 64 + 64 + 32]

class NN_Small_Tuning(nn.Module):
    def __init__(self, input_size, output_size, device):
        super(NN_Small_Tuning, self).__init__()
        self.input_size = input_size
        self.device = device if device is not None else torch.device('cpu')
        self.weights1 = _random_parameter((32, output_size), self.device)
        self.weights2 = _random_parameter((64, 32), self.device)
        self.weights3 = _random_parameter((32, 64), self.device)
        self.weights4 = _random_parameter((input_size * 2, 32), self.device)
        self.weights5 = _random_parameter((input_size * 2, output_size), self.device)
        self.T_lambda = _random_parameter((32 + 64 + 32,), self.device)

    def forward(self, A0, C, P):
        N_piomega = torch.zeros((self.input_size * 2, off_side_small[3]), device=self.device)
        N_xix = torch.zeros((off_side_small[3], self.input_size * 2), device=self.device)
        N_xiomega = torch.zeros((off_side_small[3], off_side_small[3]), device=self.device)

        T = torch.diag(self.T_lambda)
        L = self.weights5
        N_pix = L @ C
        N_piomega[:, off_side_small[2]:] = self.weights4
        N_xix[:off_side_small[1], :] = self.weights1 @ C
        N_xiomega[off_side_small[1]:off_side_small[2], off_side_small[0]:off_side_small[1]] = self.weights2
        N_xiomega[off_side_small[2]:off_side_small[3], off_side_small[1]:off_side_small[2]] = self.weights3
        A = A0 + N_pix
        H = torch.cat(
            (torch.cat((torch.transpose(A, 0, 1) @ P + P @ A, -P @ N_piomega + torch.transpose(N_xix, 0, 1) @ T), dim=1),
             torch.cat((-torch.transpose(N_piomega, 0, 1) @ P + T @ N_xix, T @ N_xiomega + torch.transpose(N_xiomega, 0, 1) @ T - 2 * T), dim=1)), dim=0)
        return -H

    def get_paras(self):
        return self.weights1, self.weights2, self.weights3, self.weights4, self.weights5

    def get_T_lambda(self):
        return self.T_lambda


class NN_Large_Tuning(nn.Module):
    def __init__(self, input_size, output_size, device):
        super(NN_Large_Tuning, self).__init__()
        self.input_size = input_size
        self.device = device if device is not None else torch.device('cpu')
        self.weights1 = _random_parameter((32, output_size), self.device)
        self.weights2 = _random_parameter((64, 32), self.device)
        self.weights3 = _random_parameter((64, 64), self.device)
        self.weights4 = _random_parameter((128, 64), self.device)
        self.weights5 = _random_parameter((128, 128), self.device)
        self.weights6 = _random_parameter((64, 128), self.device)
        self.weights7 = _random_parameter((64, 64), self.device)
        self.weights8 = _random_parameter((32, 64), self.device)
        self.weights9 = _random_parameter((input_size * 2, 32), self.device)
        self.weights0 = _random_parameter((input_size * 2, output_size), self.device)
        self.T_lambda = _random_parameter((32 + 64 + 64 + 128 + 128 + 64 + 64 + 32,), self.device)

    def forward(self, A0, C, P):
        N_piomega = torch.zeros((self.input_size * 2, off_side_large[8]), device=self.device)
        N_xix = torch.zeros((off_side_large[8], self.input_size * 2), device=self.device)
        N_xiomega = torch.zeros((off_side_large[8], off_side_large[8]), device=self.device)

        T = torch.diag(self.T_lambda)
        L = self.weights0
        N_pix = L @ C
        N_piomega[:, off_side_large[7]:] = self.weights9
        N_xix[:off_side_large[1], :] = self.weights1 @ C
        N_xiomega[off_side_large[1]:off_side_large[2], off_side_large[0]:off_side_large[1]] = self.weights2
        N_xiomega[off_side_large[2]:off_side_large[3], off_side_large[1]:off_side_large[2]] = self.weights3
        N_xiomega[off_side_large[3]:off_side_large[4], off_side_large[2]:off_side_large[3]] = self.weights4
        N_xiomega[off_side_large[4]:off_side_large[5], off_side_large[3]:off_side_large[4]] = self.weights5
        N_xiomega[off_side_large[5]:off_side_large[6], off_side_large[4]:off_side_large[5]] = self.weights6
        N_xiomega[off_side_large[6]:off_side_large[7], off_side_large[5]:off_side_large[6]] = self.weights7
        N_xiomega[off_side_large[7]:off_side_large[8], off_side_large[6]:off_side_large[7]] = self.weights8
        A = A0 + N_pix
        H = torch.cat(
            (torch.cat((torch.transpose(A, 0, 1) @ P + P @ A, P @ N_piomega + torch.transpose(N_xix, 0, 1) @ T), dim=1),
             torch.cat((torch.transpose(N_piomega, 0, 1) @ P + T @ N_xix, T @ N_xiomega + torch.transpose(N_xiomega, 0, 1) @ T - 2 * T), dim=1)), dim=0)
        return -H

    def get_paras(self):
        return self.weights1, self.weights2, self.weights3, self.weights4, self.weights5, self.weights6, self.weights7, self.weights8, self.weights9, self.weights0

    def get_T_lambda(self):
        return self.T_lambda
