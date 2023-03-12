
from matplotlib import pyplot as plt
import torch
from torch import nn
from utils import Module, DataModule, Trainer, HyperParameters

class Data(DataModule):
    def __init__(self, num_train, nom_val, num_inputs, batch_size):
        self.save_hyperparameters()
        n = self.num_train + self.nom_val
        self.X = torch.randn(n, self.num_inputs)
        noice = torch.randn(n, 1) * 0.01
        w, b = torch.ones(self.num_inputs, 1) * 0.01, 0.05
        self.y = torch.matmul(self.X, w) + b + noice
    
    def get_dataloader(self, train):
        i = slice(0, self.num_train) if train else slice(self.num_train, None)
        return self.get_tensorloader([self.X, self.y], train, i)
    

def l2_penalty(w):
    return (w**2).sum() / 2

class SGD(HyperParameters):
    """Defined in :numref:`sec_linear_scratch`"""
    def __init__(self, params, lr):
        """Minibatch stochastic gradient descent."""
        self.save_hyperparameters()

    def step(self):
        for param in self.params:
            param -= self.lr * param.grad

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()

class LinearRegression(Module):
    """Defined in :numref:`sec_linear_scratch`"""
    def __init__(self, num_inputs, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.w = torch.normal(0, sigma, (num_inputs, 1), requires_grad=True)
        self.b = torch.zeros(1, requires_grad=True)

    def forward(self, X):
        """The linear regression model.
    
        Defined in :numref:`sec_linear_scratch`"""
        return torch.matmul(X, self.w) + self.b

    def loss(self, y_hat, y):
        """Defined in :numref:`sec_linear_scratch`"""
        l = (y_hat - y) ** 2 / 2
        return torch.mean(l)

    def configure_optimizers(self):
        """Defined in :numref:`sec_linear_scratch`"""
        return SGD([self.w, self.b], self.lr)

class WeigthDecay(LinearRegression):
    def __init__(self, num_inputs, lambd, lr, sigma=0.01):
        super().__init__(num_inputs, lr, sigma)
        self.save_hyperparameters()
        
    def loss(self, y_hat, y):
        return super().loss(y_hat, y) + self.lambd * l2_penalty(self.w)
    
data = Data(num_train=20, nom_val=100, num_inputs=200, batch_size=5)  
trainer = Trainer(max_epochs=100)
lambd = 0
model = WeigthDecay(num_inputs=200, lambd=3, lr=0.01)
model.board.y_scaler = 'log'
trainer.fit(model, data)
