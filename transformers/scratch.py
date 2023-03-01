from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


d = {
    "apple": 10,
    "banana": 5,
    "chair": 2,
}

k=np.array(list(d.keys()))
v=np.array(list(d.values()))


0.6 * d["apple"] + 0.4 * d["banana"] + 0.0 * d["chair"]
np.array([0.6, 0.4, 0.0]) @ np.array([d["apple"], d["banana"], d["chair"]])


'apple', 'banana', 'chair'
'vegetable'

d = {
    "apple": [0.9, 0.2, -0.5, 1.0],
    "banana": [1.2, 2.0, 0.1, 0.2],
    "chair": [-1.2, -2.0, 1.0, -0.2],
}

d.keys()
d.values()
d.items()

0.6 * np.array(d["apple"]) + 0.4 * np.array(d["banana"]) + 0.0 * np.array(d["chair"])

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

np.sum(softmax(np.array([4., -1., 2.1, 6.])))

def get_word_vector(word, d_k=8):
    return np.random.normal(size=(d_k,))

def attention(Q, K, V):
    d_k = K.shape[-1]
    scores = np.dot(Q, K.T) / np.sqrt(d_k)
    a = softmax(scores)
    return np.dot(a, V)

def kv_lookup(query, keys, values):
    return attention(
        Q=np.stack([get_word_vector(qi) for qi in query])
        ,
        K=np.array([get_word_vector(ki)  for ki in keys])
        ,
        V=np.array(values)
        ,
        )

kv_lookup(query=['fruit', 'vegetable'], keys=list(d.keys()), values=list(d.values()))


def gausian(x): return torch.exp(-x**2/2)
def boxcar(x): return (x.abs() < 1).float()
def constant(x): return 0.5 * torch.ones_like(x)
def enapechikov(x): return torch.max(1 - torch.abs(x), torch.zeros_like(x))

kernels = [gausian, boxcar, constant, enapechikov]
names = ['gausian', 'boxcar', 'constant', 'enapechikov']

fig, ax = plt.subplots(1, 4)
x = torch.arange(-3, 3, 0.1)
for kernel, name, ax in zip(kernels, names, ax):
    ax.plot(x, kernel(x))
    ax.set_title(name)
plt.show()

def f(x): return 2*torch.sin(x)+x

n = 40
x_train, _ = torch.sort(torch.randn(n)*5)
y_train = f(x_train) + torch.normal(0, 0.5, size=(n,))

x_val = torch.arange(0, 5, 0.1)
y_val= f(x_val)

# plt.plot(x_train, y_train, 'o')
# plt.plot(x_val, y_val)
# plt.show()


def nadaraya_watson(x_train, y_train, x_val, kernel, h):
    y_pred = torch.zeros_like(x_val)
    for i, x in enumerate(x_val):
        y_pred[i] = torch.sum(kernel((x - x_train)/h) * y_train) / torch.sum(kernel((x - x_train)/h))
    return y_pred, None

def nadaraya_watson_ex(x_train, y_train, x_val, kernel):
    dists = x_train.reshape(-1, 1) - x_val.reshape(1, -1)
    k = kernel(dists).float()
    attention = k / torch.sum(k, dim=0)
    y_hat =y_train@attention
    return y_hat, attention


def plot_attention(x_train, y_train, x_val, kernels, names):
    fig, axs = plt.subplots(2, 4)
    for kernel, name, (ax1, ax2) in zip(kernels, names, axs.T):
        pass
        y_pred, attention_W = nadaraya_watson(x_train, y_train, x_val, kernel=kernel, h=0.5)
        y_pred_ex, attention_W = nadaraya_watson_ex(x_train, y_train, x_val, kernel=kernel)

        ax1.plot(x_train, y_train, 'o')
        ax1.plot(x_val, y_val)
        ax1.plot(x_val, y_pred)
        ax1.plot(x_val, y_pred_ex, '--')

        if attention_W is not None:
            ax2.imshow(attention_W, cmap='Reds')
    plt.show()

sigmas = (0.1, 0.2, 0.5, 1)
names = ['Sigma ' + str(sigma) for sigma in sigmas]

def gaussian_with_width(sigma):
    return (lambda x: torch.exp(-x**2 / (2*sigma**2)))
kernels = [gaussian_with_width(sigma) for sigma in sigmas]
plot_attention(x_train, y_train, x_val, kernels, names)

