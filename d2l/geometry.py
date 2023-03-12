
import torch
import torchvision
from torchvision import transforms
import math

def angle(v, w):
    cos = torch.dot(v, w) / (torch.norm(v) * torch.norm(w))
    return torch.acos(torch.clamp(cos, -1, 1))

v = torch.tensor([1, 2, 3], dtype=torch.float32)
w = torch.tensor([4, 5, 6], dtype=torch.float32)
v@w

alpha = angle(v, w).item()
math.degrees(alpha)

v = torch.eye(4)
w = torch.randn(4, 4)
torch.matmul(v, w)

A = torch.tensor([[1, 2], [3, 4]]).float()
B = torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]).float()
v = torch.tensor([1, 2]).float()

# Print out the shapes
A.shape, B.shape, v.shape
B.T@A@v
torch.einsum('ij,j->i', A, v)

torch.einsum("ijk,il,j->kl", B, A, v)

A = torch.tensor([[2, 0], [0, -1]]).float()
A@torch.tensor([1, 0]).float()
A@torch.tensor([0, 1]).float()

torch.eig(torch.tensor([[2, 1], [2, 3]], dtype=torch.float64), eigenvectors=True)
