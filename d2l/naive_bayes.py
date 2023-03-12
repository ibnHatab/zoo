import math
from matplotlib import pyplot as plt
import torch
import torchvision

data_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    lambda x: torch.floor(x * 255 / 128).squeeze(dim=0)])

mnist_train = torchvision.datasets.MNIST(root='./data/MNIST', train=True, download=True, transform=data_transform)
mnist_test = torchvision.datasets.MNIST(root='./data/MNIST', train=False, download=True, transform=data_transform)

image, label = mnist_train[2]

image.shape, label

images = torch.stack([mnist_train[i][0] for i in range(10,38)], dim=0)
labels = torch.tensor([mnist_train[i][1] for i in range(10,38)]);

from utils import show_images
# show_images(images, 2, 9)
# plt.show()

X = torch.stack([mnist_train[i][0] for i in range(len(mnist_train))], dim=0)
Y = torch.tensor([mnist_train[i][1] for i in range(len(mnist_train))])

n_y = torch.zeros(10)
for y in range(10):
    n_y[y] = (Y == y).sum()
P_y = n_y / n_y.sum()

n_x = torch.zeros(10, 28, 28)
for y in range(10):
    n_x[y] = torch.tensor(X.numpy()[Y == y].sum(axis=0))

P_xy = (n_x+1) / (n_y+2).reshape(10, 1, 1)
# plt.imshow(n_x[3])
# show_images(p_xy, 2, 5)
# plt.show()

x = image
def bayes_predict(X):
    x = x.unsqueeze(dim=0)
    p_xy = P_xy * x + (1-P_xy) * (1-x)
    p_xy = p_xy.reshape(10, -1).prod(axis=1)
    return p_xy * P_y

log_P_xy = torch.log(P_xy)
log_P_xy_neg = torch.log(1-P_xy)
log_P_y = torch.log(P_y)

def bayes_predict_log(x):
    x = x.unsqueeze(dim=0)
    log_xy = log_P_xy * x + log_P_xy_neg * (1-x)
    log_xy = log_xy.reshape(10, -1).sum(axis=1)
    return log_xy + log_P_y

py = bayes_predict_log(image)
py.argmax(dim=0) == label

def predict(X):
    return [bayes_predict_log(x).argmax(dim=0).type(torch.int32).item() for x in X]

Xt = torch.stack([mnist_test[i][0] for i in range(len(mnist_test))], dim=0)
yt = torch.tensor([mnist_test[i][1] for i in range(len(mnist_test))])
preds = predict(Xt)

show_images(Xt[:18], 2, 9, titles = [str(d) for d in preds[:18]])

preds = torch.tensor(preds, dtype=torch.int32)
float((preds == yt).sum()) / len(yt)

