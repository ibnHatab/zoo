from matplotlib import pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
# %cd src

from utils import Classifier, FashionMNIST, Trainer


class SoftmaxRegression(Classifier):                        
    def __init__(self, num_outputs, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(nn.Flatten(), 
                                 nn.LazyLinear(num_outputs))
    def forward(self, X):
        return self.net(X)
    
data = FashionMNIST(batch_size=256)
model = SoftmaxRegression(num_outputs=10, lr=0.1)
trainer = Trainer(max_epochs=10, num_gpus=1)
trainer.fit(model, data)

