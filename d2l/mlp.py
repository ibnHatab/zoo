
from matplotlib import pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F

from utils import Classifier, FashionMNIST, Trainer
#%cd src

class MLPScrach(Classifier):
    def __init__(self, num_inputs, num_outputs, num_hiddens, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens) * sigma)
        self.b1 = nn.Parameter(torch.zeros(num_hiddens))
        
        self.W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs) * sigma)
        self.b2 = nn.Parameter(torch.zeros(num_outputs))
        
    def forward(self, X):
        X = X.reshape(-1, self.num_inputs)
        H = X @ self.W1 + self.b1
        H = torch.relu(H)
        O = H @ self.W2 + self.b2
        return O

class MLP(Classifier):
    def __init__(self, num_outputs, num_hiddens, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(nn.Flatten(), 
                                 nn.LazyLinear(num_hiddens), 
                                 nn.ReLU(), 
                                 nn.LazyLinear(num_outputs))
        

model = MLPScrach(num_inputs=784, num_outputs=10, num_hiddens=256, lr=0.1)
model = MLP(num_outputs=10, num_hiddens=256, lr=0.1)
data = FashionMNIST(batch_size=256)        
traine = Trainer(max_epochs=10, num_gpus=1)
        
        
traine.fit(model, data)        

plt.show()
        
        # self.net = nn.Sequential(nn.Flatten(), 
        #                          nn.LazyLinear(hidden_sizes[0]), 
        #                          nn.ReLU(), 
        #                          nn.LazyLinear(hidden_sizes[1]), 
        #                          nn.ReLU(), 
        #                          nn.LazyLinear(num_outputs))