
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

class LeNet5(Classifier):

    def __init__(self, lr=0.1, num_classes=10):
        super().__init__()
        self.save_hyperparameters()

        self.net = nn.Sequential(
            nn.LazyConv2d(6, kernel_size=5, padding=2),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.LazyConv2d(16, kernel_size=5),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.LazyLinear(120),
            nn.Sigmoid(),
            nn.LazyLinear(84),
            nn.Sigmoid(),
            nn.LazyLinear(num_classes)
        )

    def layer_summary(self, X_shape):
        X = torch.randn(*X_shape)
        for layer in self.net:
            X = layer(X)
            print(layer.__class__.__name__, 'output shape:\t', X.shape)

# graph_forward(model, X)
# graph_backward(model, X)
# plt.imshow(X[0][0])
# plt.show()
model = LeNet5()
data = FashionMNIST(batch_size=256)
trainer = Trainer(max_epochs=40, num_gpus=1)
model.apply_init([next(iter(data.get_dataloader(True)))[0]], init_cnn)
trainer.fit(model, data)

ll = next(iter(data.get_dataloader(True)))[0]
ll.device
pic = model.net[0:3](ll.to('cuda:0'))
pic = pic.detach().cpu().numpy()
plt.imshow(pic[0][0])
plt.show(block=False)