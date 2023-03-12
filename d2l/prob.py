
from matplotlib import pyplot as plt
import torch
import random
from torch.distributions import Multinomial


num_toses = 1000
fair_probs = torch.tensor([0.5, 0.5])
counts = Multinomial(1, fair_probs).sample((num_toses,)) 
cum_counts = counts.cumsum(dim=0)
estimate_probs = cum_counts / cum_counts.sum(dim=1, keepdim=True)
estimatea = estimate_probs.numpy()

plt.plot(estimatea[:, 0], label='heads')
plt.plot(estimatea[:, 1], label='tails')
plt.axhline(y=0.5, color='black', linestyle='dashed')
plt.show(block=False)


counts = Multinomial(1, fair_probs).sample((10000,))
cum_counts = counts.cumsum(dim=0)
estimates = cum_counts / cum_counts.sum(dim=1, keepdims=True)
estimates = estimates.numpy()

plt.plot(estimates[:, 0], label=("P(coin=heads)"))
plt.plot(estimates[:, 1], label=("P(coin=tails)"))
plt.axhline(y=0.5, color='black', linestyle='dashed')
plt.gca().set_xlabel('Samples')
plt.gca().set_ylabel('Estimated probability')
plt.legend();

