
import torch
from torch import nn
from torch.nn import functional as F
from torchviz import make_dot
from torchview import draw_graph

net = nn.Sequential(nn.LazyLinear(256), nn.ReLU(), nn.LazyLinear(10))
X = torch.rand(2, 20)
y = net(X)
net(X).shape

print(net)

make_dot(y, params=dict(list(net.named_parameters()))).render("rnn_torchviz", format="png", view=True)

mg = draw_graph(net, input_size=(2, 20), expand_nested=True)
mg.visual_graph.render("rnn_torchview", format="png", view=True)

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden = nn.LazyLinear(256)  # hidden layer
        self.out = nn.LazyLinear(10)  # output layer

    def forward(self, X):
        return self.out(F.relu(self.hidden(X)))

net = MLP()
net(X).shape
net

class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # Random weight parameters that will not compute gradients and
        # therefore keep constant during training
        self.rand_weight = torch.rand((20, 20))
        self.linear = nn.LazyLinear(20)

    def forward(self, X):
        X = self.linear(X)
        X = F.relu(X @ self.rand_weight + 1)
        # Reuse the fully connected layer. This is equivalent to sharing
        # parameters with two fully connected layers
        X = self.linear(X)
        # Control flow
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()

net = FixedHiddenMLP()
net(X)

class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.LazyLinear(64), nn.ReLU(),
                                 nn.LazyLinear(32), nn.ReLU())
        self.linear = nn.LazyLinear(16)

    def forward(self, X):
        return self.linear(self.net(X))

chimera = nn.Sequential(NestMLP(), nn.LazyLinear(20))
chimera(X)

net = chimera

torch.save(net.state_dict(), 'net.pt')

chimera.load_state_dict(torch.load('net.pt'))