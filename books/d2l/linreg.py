
import math
import numpy as np
import torch
from torch import nn 
from src.utils import plot, HyperParameters


def normal(x, mu, sigma):
    p1 = 1 / (sigma * math.sqrt(2 * math.pi))
    p2 = -0.5 / sigma**2 * (x - mu)**2
    return p1 * np.exp(p2)

x = np.arange(-7, 7, 0.01)
mu, sigma = 0, 1
normal(x, 0, 1)

params = [(0, 1), (0, 2), (3, 1)]
plot(x, [normal(x, mu, sigma) for mu, sigma in params], xlabel='x',
         ylabel='p(x)', figsize=(4.5, 2.5),
         legend=[f'mean {mu}, std {sigma}' for mu, sigma in params])


class B(HyperParameters):
    def __init__(self, a, b, c) -> None:
        self.save_hyperparameters(ignore=[c])
        
b = B(1,2,3)        
b.hparams
