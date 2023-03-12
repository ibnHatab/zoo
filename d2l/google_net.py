
from matplotlib import pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F

# %cd src
from utils import Classifier, FashionMNIST, Module, Trainer, graph_backward, graph_forward


class Inception(Module):
    def __init__(self, c1, c2, c3, c4, **kwargs):
        super().__init__(**kwargs)
        self.b1_1 = nn.LazyConv2d(c1, kernel_size=1)

        self.b2_1 = nn.LazyConv2d(c2[0], kernel_size=1)
        self.b2_2 = nn.LazyConv2d(c2[1], kernel_size=3, padding=1)

        self.b3_1 = nn.LazyConv2d(c3[0], kernel_size=1)
        self.b3_2 = nn.LazyConv2d(c3[1], kernel_size=5, padding=2)

        self.b4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.b4_2 = nn.LazyConv2d(c4, kernel_size=1)

    def forward(self, X):
        y1 = F.relu(self.b1_1(X))
        y2 = F.relu(self.b2_2(F.relu(self.b2_1(X))))
        y3 = F.relu(self.b3_2(F.relu(self.b3_1(X))))
        y4 = F.relu(self.b4_2(self.b4_1(X)))
        return torch.cat((y1, y2, y3, y4), dim=1)

class GoogleNet(Classifier):

    @staticmethod
    def init_cnn(module):
        if type(module) in (nn.Conv2d, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def __init__(self, lr=0.1, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(
            self.b1(),
            self.b2(),
            self.b3(),
            self.b4(),
            self.b5(),
            nn.LazyLinear(num_classes)
        )
        self.apply(GoogleNet.init_cnn)

    def b1(self):
        return nn.Sequential(
            nn.LazyConv2d(64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    def b2(self):
        return nn.Sequential(
            nn.LazyConv2d(64, kernel_size=1),
            nn.LazyConv2d(192, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    def b3(self):
        return nn.Sequential(
            Inception(64, (96, 128), (16, 32), 32),
            Inception(128, (128, 192), (32, 96), 64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    def b4(self):
        return nn.Sequential(
            Inception(192, (96, 208), (16, 48), 64),
            Inception(160, (112, 224), (24, 64), 64),
            Inception(128, (128, 256), (24, 64), 64),
            Inception(112, (144, 288), (32, 64), 64),
            Inception(256, (160, 320), (32, 128), 128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    def b5(self):
        return nn.Sequential(Inception(256, (160, 320), (32, 128), 128),
                             Inception(384, (192, 384), (48, 128), 128),
                             nn.AdaptiveAvgPool2d((1, 1)),
                             nn.Flatten())

model = GoogleNet()
model.layer_summary((1, 1, 96, 96))

model = GoogleNet(lr=0.01)
trainer = Trainer(max_epochs=10, num_gpus=1)
data = FashionMNIST(batch_size=128, resize=(96, 96))
model.apply_init([next(iter(data.get_dataloader(True)))[0]], GoogleNet.init_cnn)
trainer.fit(model, data)