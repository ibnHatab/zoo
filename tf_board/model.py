
import torch
import torch.nn as nn
from torch.nn import functional as F

from utils import graph_backward, graph_forward

class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# net = Net()
# X = torch.rand(1, 1, 28, 28)
# y = net(X)

# p = torch.softmax(y, dim=1)
# idx = torch.argmax(p, dim=1)
# #

# graph_forward(net, X)
# graph_backward(net, X)
