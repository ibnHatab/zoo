
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

class AlexNet(Classifier):
    def __init__(self, lr=0.1, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(
            nn.LazyConv2d(96, kernel_size=11, stride=4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LazyConv2d(256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LazyConv2d(384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.LazyConv2d(384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.LazyConv2d(256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Flatten(),
            nn.LazyLinear(4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.LazyLinear(4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.LazyLinear(num_classes),
        )
        self.net.apply(init_cnn)

model = AlexNet()
X = torch.randn(1, 1, 224, 224)
y = model(X)
y.shape

# graph_forward(model, X)
# graph_backward(model, X)
data = FashionMNIST(batch_size=256, resize=(224, 224))
trainer = Trainer(max_epochs=10, num_gpus=1)
trainer.fit(model, data)