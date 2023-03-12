
from matplotlib import pyplot as plt
import torch
from torch import nn

# %cd src
from utils import Classifier, FashionMNIST, Trainer, graph_backward, graph_forward

def init_cnn(module):
    if type(module) in (nn.Conv2d, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

def nin_block(out_channels, kernel_size, stride, padding):
    blk = nn.Sequential(
        nn.LazyConv2d(out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.ReLU(),
        nn.LazyConv2d(out_channels, kernel_size=1, stride=1, padding=0),
        nn.ReLU(),
        nn.LazyConv2d(out_channels, kernel_size=1, stride=1, padding=0),
        nn.ReLU(),
    )
    return blk

class NiN(Classifier):
    def __init__(self, lr=0.1, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(
            nin_block(96, kernel_size=11, stride=4, padding=0),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nin_block(256, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nin_block(384, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Dropout(p=0.5),
            nin_block(num_classes, kernel_size=3, stride=1, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

model = NiN(lr=0.5)
X = torch.randn(1, 1, 224, 224)
y = model(X)
# graph_forward(model, X)
# graph_backward(model, X)
train = Trainer(max_epochs=10, num_gpus=1)
data = FashionMNIST(batch_size=128, resize=(224, 224))
model.apply_init([next(iter(data.get_dataloader(True)))[0]], init_cnn)
train.fit(model, data)
