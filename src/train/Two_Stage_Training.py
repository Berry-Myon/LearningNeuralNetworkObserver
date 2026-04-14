from .Pretrain import Pretrain
from .Tuning import Tuning


class Two_Stage_Training:
    def __init__(self, a, c, epsilon, lr, model, P, pre_epochs=50000, tun_epochs=10000, device=None):
        self.A = a
        self.C = c
        self.epsilon = epsilon
        self.lr = lr
        self.model = model
        self.P = P
        self.pre_epochs = pre_epochs
        self.tun_epochs = tun_epochs
        self.device = device

    def train(self):
        pre = Pretrain(self.A, self.C, self.epsilon, self.lr, self.P, self.pre_epochs)
        tun = Tuning(self.A, self.C, self.epsilon, self.lr, self.P, self.tun_epochs, self.device)
        print("Start Pretraining")
        pre.init_model(self.model)
        pre.pre_train()
        print("Pretraining End")
        print("Start Finetuning")
        tun.init_model(self.model)
        tun.tuning()
        print("Finetuning End")
