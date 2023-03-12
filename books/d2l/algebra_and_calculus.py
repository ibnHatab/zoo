
import torch

from src.utils import *

x = torch.tensor(3.0)
y = torch.tensor(4.0)

x+y, x*y, x/y, 
z = (x**y)
z.item()

x = torch.arange(4, dtype=torch.float32)
len(x), x.shape, x.numel(), x.dim()

x.sum(), x.mean(), x.std()

A = torch.arange(20).reshape(5, 4)
A, A.T

torch.arange(24).reshape(2, 3, -1).shape

B = A.clone()
A * B
A.shape, B.T.shape, (A @ B.T).shape

A.T @ B

A.shape, A.sum(axis=0), A.sum(axis=1)
A = A.float()
A.mean(axis=0), A.sum(axis=1) / A.shape[1]
A.mean(axis=1, keepdims=True)
A, A.cumsum(axis=1)
A*x, A@x
torch.mv(A, x)

B = torch.ones(4, 3)
torch.mm(A, B), A @ B

u = torch.tensor([3.0, -4.0])
u.norm()
u.abs().sum()

torch.norm(torch.ones((4, 9)))
T = torch.rand((4, 9, 3))
T
torch.linalg.norm(T, dim=(1, 2))

def norm(X):
    return (X**2).sum().sqrt()

norm(X)

def binomial(n, k, p):
    return torch.distributions.binomial.Binomial(n, p).probs


B= binomial(10, 2, 0.5)
B.sample((10,))

def multinomial(n, p):
    return torch.distributions.multinomial.Multinomial(n, p).probs

multinomial(10, torch.tensor([0.1, 0.2, 0.3, 0.4]))

def f(x): return 3*x**2 - 4*x

for h in 10.**np.arange(-1, -6, -1):
    print(f"{h} {(f(1+h) - f(1))/h}")
    
    
x = np.arange(-2, 3, 0.1)
plot(x, [f(x), 2 * x - 3], 'x', 'f(x)', legend=['f(x)', 'Tangent line (x=1)'])

x = np.arange(0.1, 3, 0.1)
def f(x): return x**3 - 1/x
f"Derivative at x=1: {(f(1+h) - f(1))/h}"
x=1
plot(x, [f(x), 3*(x-1)**2 + 1/((x-1)**2), 4*x-4], 'x', 'f(x)', legend=['f(x)', 'Tangent line (x=1)'])

# for function x**3 - 1/x find the tangent line at x=1
