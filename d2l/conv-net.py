from matplotlib import pyplot as plt
import torch
from torch import nn

def corr2d(X, K):
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i+h, j:j+w] * K).sum()
    return Y

X = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]]).float()
K = torch.tensor([[0, 1], [2, 3]]).float()

corr2d(X, K)

class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super(Conv2D, self).__init__()
        self.weight = nn.Parameter(torch.randn(kernel_size))
        self.bias = nn.Parameter(torch.randn(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias

X = torch.ones(6, 8)
X[:, 2:6] = 0


K = torch.tensor([[1, -1]]).float()
Y = corr2d(X, K)

# plt.imshow(Y, cmap='gray')
# plt.show()

net = nn.LazyConv2d(1, kernel_size=(1, 2), bias=False)
X = X.reshape((1, 1, 6, 8))
Y = Y.reshape((1, 1, 6, 7))
lr = 0.03

for i in range(10):
    Y_hat = net(X)
    l = ((Y_hat - Y) ** 2).sum()
    l.backward()
    net.weight.data[:] -= lr * net.weight.grad
    net.weight.grad.zero_()
    #if (i+1) % 2 == 0:
    print('batch %d, loss %.3f' % (i + 1, l.item()))

net.weight.data
