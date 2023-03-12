from matplotlib import pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F

# %cd src
from utils import Classifier, FashionMNIST, Module, Trainer, graph_backward, graph_forward

def conv_block(num_channels):
    return nn.Sequential(
        nn.LazyBatchNorm2d(),
        nn.ReLU(),
        nn.LazyConv2d(num_channels, kernel_size=3, padding=1),
    )

class DenseBlock(Module):
    def __init__(self, num_convs, num_channels, **kwargs):
        super().__init__(**kwargs)
        layers = []
        for i in range(num_convs):
            layers.append(conv_block(num_channels))
        self.net = nn.Sequential(*layers)

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            X = torch.cat((X, Y), dim=1)
        return X

blk = DenseBlock(2, 10)
X = torch.rand(4, 3, 8, 8)
Y = blk(X)

def transition_block(num_channels):
    return nn.Sequential(
        nn.LazyBatchNorm2d(),
        nn.ReLU(),
        nn.LazyConv2d(num_channels, kernel_size=1),
        nn.AvgPool2d(kernel_size=2, stride=2)
    )

blk = transition_block(10)
blk(Y)
graph_forward(blk, Y)

class DenseNet(Classifier):
    @staticmethod
    def init_cnn(module):
        if type(module) in (nn.Conv2d, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)


    def __init__(self, num_channels, growth_rate=32, arch=(4, 4, 4, 4), lr=0.1, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(self.b1())

        for i, num_convs in enumerate(arch):
            self.net.add_module(f"block{i+1}", DenseBlock(num_convs, growth_rate))
            num_channels += num_convs * growth_rate
            if i != len(arch) - 1:
                num_channels = num_channels // 2
                self.net.add_module(f"transition{i+1}", transition_block(num_channels))
        self.net.add_module("last", nn.Sequential(
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.LazyLinear(num_classes)
        ))
        self.net.apply(self.init_cnn)

    def b1(self):
        return nn.Sequential(
            nn.LazyConv2d(64, kernel_size=7, stride=2, padding=3),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

model = DenseNet(64, lr=0.01)
X = torch.rand(1, 1, 224, 224)
y = model(X)
graph_forward(model, X, depth=4)
trainer = Trainer(max_epochs=10, num_gpus=1)
data = FashionMNIST(batch_size=128, resize=96)
trainer.fit(model, data)


