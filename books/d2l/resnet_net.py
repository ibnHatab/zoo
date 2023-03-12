from matplotlib import pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F

# %cd src
from utils import Classifier, FashionMNIST, Module, Trainer, graph_backward, graph_forward


class Residual(Module):
    def __init__(self, num_channels, use_1x1conv=False, stride=1):
        super().__init__()
        self.save_hyperparameters()
        self.conv1 = nn.LazyConv2d(num_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.LazyConv2d(num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.LazyConv2d(num_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.LazyBatchNorm2d()
        self.bn2 = nn.LazyBatchNorm2d()

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        return F.relu(self.bn2(self.conv2(Y))) + (self.conv3(X) if self.conv3 else X)

class ResNet(Classifier):

    @staticmethod
    def init_cnn(module):
        if type(module) in (nn.Conv2d, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def __init__(self, arch, lr=0.1, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(self.b1())
        for i, b in enumerate(arch):
            self.net.add_module(f"block{i+2}", self.block(*b, first_block=i==0))
        self.net.add_module("last", nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.LazyLinear(num_classes)))
        self.net.apply(ResNet.init_cnn)


    def b1(self):
        return nn.Sequential(
            nn.LazyConv2d(64, kernel_size=7, stride=2, padding=3),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    def block(self, num_channels, num_residuals, first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(Residual(num_channels, use_1x1conv=True, stride=2))
            else:
                blk.append(Residual(num_channels))
        return nn.Sequential(*blk)

class ResNet18(ResNet):
    def __init__(self, lr=0.1, num_classes=10):
        super().__init__([(64, 2), (128, 2), (256, 2), (512, 2)], lr, num_classes)

ResNet18().layer_summary((1, 1, 224, 224))

model = ResNet18()
X = torch.randn(1, 1, 224, 224)
y = model(X)
graph_forward(model, X, depth=4)
data = FashionMNIST(batch_size=256, resize=(96, 96))
trainer = Trainer(max_epochs=10, num_gpus=1)
model = ResNet18()
model.apply_init([next(iter(data.get_dataloader(True)))[0]], ResNet.init_cnn)
trainer.fit(model,0 data)


class ResNeXtBlock(nn.Module):  #@save
    """The ResNeXt block."""
    def __init__(self, num_channels, groups, bot_mul, use_1x1conv=False,
                 strides=1):
        super().__init__()
        bot_channels = int(round(num_channels * bot_mul))
        self.conv1 = nn.LazyConv2d(bot_channels, kernel_size=1, stride=1)
        self.conv2 = nn.LazyConv2d(bot_channels, kernel_size=3,
                                   stride=strides, padding=1,
                                   groups=bot_channels//groups)
        self.conv3 = nn.LazyConv2d(num_channels, kernel_size=1, stride=1)
        self.bn1 = nn.LazyBatchNorm2d()
        self.bn2 = nn.LazyBatchNorm2d()
        self.bn3 = nn.LazyBatchNorm2d()
        if use_1x1conv:
            self.conv4 = nn.LazyConv2d(num_channels, kernel_size=1,
                                       stride=strides)
            self.bn4 = nn.LazyBatchNorm2d()
        else:
            self.conv4 = None

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = F.relu(self.bn2(self.conv2(Y)))
        Y = self.bn3(self.conv3(Y))
        if self.conv4:
            X = self.bn4(self.conv4(X))
        return F.relu(Y + X)