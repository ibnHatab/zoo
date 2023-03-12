
import torch
import numpy as np
            

x = torch.arange(12, dtype=torch.float32)
x.shape

xr = x.reshape(3, 4)

np.allclose(x.view(3,4), xr)

torch.zeros_like(xr)
torch.rand(3, 4)

X = torch.tensor([[2, 0, 1, 3], 
                  [1, 2, 3, 4], 
                  [4, 3, 2, 1]], dtype=torch.float32)

X[:,-1]
X[:2, :] = 12

torch.exp(X)
Y = torch.arange(12, dtype=torch.float32).reshape(3, 4)
X.shape, Y.shape
torch.cat((X, Y), dim=1)

X == Y
Y.sum()

a = torch.arange(3, dtype=torch.float32).reshape(3, 1)
b = torch.arange(2, dtype=torch.float32).reshape(1, 2)
a+b

Z = Y.clone()
Z = Z + X
Z =+ X

Z = torch.zeros_like(Y)
id(Y), id(Z)
Z[:] = X + Y
Z

A = X.numpy()
B = torch.from_numpy(A)
type(A), type(B)

a = torch.tensor([3.5])
a, a.item(), float(a), int(a)

X < Y

