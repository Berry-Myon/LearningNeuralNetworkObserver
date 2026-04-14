import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os

# Project root directory (parent of src)
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_weights_dir = os.path.join(_project_root, 'Weights')


class NN_Small_Point(nn.Module):
    def __init__(self, input_size, output_size, P):
        super(NN_Small_Point, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.P = P
        self.weights1 = nn.Parameter(torch.tensor(np.random.normal(0, 1, size=(32, output_size)), dtype=torch.float32))
        self.weights2 = nn.Parameter(torch.tensor(np.random.normal(0, 1, size=(64, 32)), dtype=torch.float32))
        self.weights3 = nn.Parameter(torch.tensor(np.random.normal(0, 1, size=(32, 64)), dtype=torch.float32))
        self.weights4 = nn.Parameter(torch.tensor(np.random.normal(0, 1, size=(input_size * 2, 32)), dtype=torch.float32))
        self.weights5 = nn.Parameter(torch.tensor(np.random.normal(0, 1, size=(input_size * 2, output_size)), dtype=torch.float32))

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
        self.weights1 = nn.Parameter(torch.tensor(np.random.normal(0, 1, size=(32, output_size)), dtype=torch.float32))
        self.weights2 = nn.Parameter(torch.tensor(np.random.normal(0, 1, size=(64, 32)), dtype=torch.float32))
        self.weights3 = nn.Parameter(torch.tensor(np.random.normal(0, 1, size=(64, 64)), dtype=torch.float32))
        self.weights4 = nn.Parameter(torch.tensor(np.random.normal(0, 1, size=(128, 64)), dtype=torch.float32))
        self.weights5 = nn.Parameter(torch.tensor(np.random.normal(0, 1, size=(128, 128)), dtype=torch.float32))
        self.weights6 = nn.Parameter(torch.tensor(np.random.normal(0, 1, size=(64, 128)), dtype=torch.float32))
        self.weights7 = nn.Parameter(torch.tensor(np.random.normal(0, 1, size=(64, 64)), dtype=torch.float32))
        self.weights8 = nn.Parameter(torch.tensor(np.random.normal(0, 1, size=(32, 64)), dtype=torch.float32))
        self.weights9 = nn.Parameter(torch.tensor(np.random.normal(0, 1, size=(input_size * 2, 32)), dtype=torch.float32))
        self.weights0 = nn.Parameter(torch.tensor(np.random.normal(0, 1, size=(input_size * 2, output_size)), dtype=torch.float32))

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
    def __init__(self, input_size, device):
        super(NN_Small_Tuning, self).__init__()
        self.input_size = input_size
        self.device = device
        self.weights1 = nn.Parameter(torch.tensor(np.array(pd.read_csv(os.path.join(_weights_dir, 'Weight_1.csv'), header=None)), dtype=torch.float32).to(device))
        self.weights2 = nn.Parameter(torch.tensor(np.array(pd.read_csv(os.path.join(_weights_dir, 'Weight_2.csv'), header=None)), dtype=torch.float32).to(device))
        self.weights3 = nn.Parameter(torch.tensor(np.array(pd.read_csv(os.path.join(_weights_dir, 'Weight_3.csv'), header=None)), dtype=torch.float32).to(device))
        self.weights4 = nn.Parameter(torch.tensor(np.array(pd.read_csv(os.path.join(_weights_dir, 'Weight_4.csv'), header=None)), dtype=torch.float32).to(device))
        self.weights5 = nn.Parameter(torch.tensor(np.array(pd.read_csv(os.path.join(_weights_dir, 'Weight_5.csv'), header=None)), dtype=torch.float32).to(device))
        T_lambda_init = torch.tensor(np.random.normal(0, 1, size=32 + 64 + 32), dtype=torch.float32).to(device)
        self.T_lambda = nn.Parameter(T_lambda_init)

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
    def __init__(self, input_size, device):
        super(NN_Large_Tuning, self).__init__()
        self.input_size = input_size
        self.device = device
        self.weights1 = nn.Parameter(torch.tensor(np.array(pd.read_csv(os.path.join(_weights_dir, 'Weight_1.csv'), header=None)), dtype=torch.float32).to(device))
        self.weights2 = nn.Parameter(torch.tensor(np.array(pd.read_csv(os.path.join(_weights_dir, 'Weight_2.csv'), header=None)), dtype=torch.float32).to(device))
        self.weights3 = nn.Parameter(torch.tensor(np.array(pd.read_csv(os.path.join(_weights_dir, 'Weight_3.csv'), header=None)), dtype=torch.float32).to(device))
        self.weights4 = nn.Parameter(torch.tensor(np.array(pd.read_csv(os.path.join(_weights_dir, 'Weight_4.csv'), header=None)), dtype=torch.float32).to(device))
        self.weights5 = nn.Parameter(torch.tensor(np.array(pd.read_csv(os.path.join(_weights_dir, 'Weight_5.csv'), header=None)), dtype=torch.float32).to(device))
        self.weights6 = nn.Parameter(torch.tensor(np.array(pd.read_csv(os.path.join(_weights_dir, 'Weight_6.csv'), header=None)), dtype=torch.float32).to(device))
        self.weights7 = nn.Parameter(torch.tensor(np.array(pd.read_csv(os.path.join(_weights_dir, 'Weight_7.csv'), header=None)), dtype=torch.float32).to(device))
        self.weights8 = nn.Parameter(torch.tensor(np.array(pd.read_csv(os.path.join(_weights_dir, 'Weight_8.csv'), header=None)), dtype=torch.float32).to(device))
        self.weights9 = nn.Parameter(torch.tensor(np.array(pd.read_csv(os.path.join(_weights_dir, 'Weight_9.csv'), header=None)), dtype=torch.float32).to(device))
        self.weights0 = nn.Parameter(torch.tensor(np.array(pd.read_csv(os.path.join(_weights_dir, 'Weight_0.csv'), header=None)), dtype=torch.float32).to(device))
        T_lambda_init = torch.tensor(np.random.normal(0, 1, size=32 + 64 + 64 + 128 + 128 + 64 + 64 + 32), dtype=torch.float32).to(device)
        self.T_lambda = nn.Parameter(T_lambda_init)

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
