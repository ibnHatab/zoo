import torch
import numpy as np
from torch import tensor
from numpy import array

x = torch.arange(4.0)   
x.requires_grad_(True)
x.grad
y = 2 * torch.dot(x, x)
y
y.backward()
x, y, x.grad

x.grad == 4*x

x.grad.zero_()
y = x*x
y.backward(torch.ones_like(x)/10)
x.grad


################################

# x=1 y=x^2
x=tensor(1.0, requires_grad=True)
y=x**2
y.backward()
x.grad

x = x.detach().numpy()
J = array([[2*x]])
vt = array([[1.0, ]])
vt@J

def f(x): return torch.sin(x)

x = torch.arange(0., 10., .01, requires_grad=True)
y = f(x)
y.backward(torch.ones_like(x))

from src.utils import *
plot(x.detach().numpy(), [y.detach().numpy(), x.grad.detach().numpy()], 'x', 'f(x)')
y = x.pow(2)
y.grad_fn._saved_self