
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

def vgg_block(num_convs, out_channels):
    blk = []
    for _ in range(num_convs):
        blk.append(nn.LazyConv2d(out_channels, kernel_size=3, padding=1))
        blk.append(nn.ReLU())
    blk.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*blk)

class VGG(Classifier):
    def __init__(self, arch, lr=0.1, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        conv_blks = []
        for (num_convs, out_channels) in arch:
            conv_blks.append(vgg_block(num_convs, out_channels))

        self.net = nn.Sequential(
            *conv_blks,
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

model = VGG(arch=[(1, 64), (1, 128),  (2, 256), (2, 512), (2, 512)])
X = torch.randn(1, 1, 224, 224)
y = model(X)
graph_forward(model, X)
graph_backward(model, X)
data = FashionMNIST(batch_size=128, resize=(32, 32))
trainer = Trainer(max_epochs=10, num_gpus=1)
trainer.fit(model, data)
plt.show()