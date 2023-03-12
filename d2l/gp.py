import math
import numpy as np
from scipy.spatial import distance_matrix
from scipy import optimize
import gpytorch
import torch

# %cd src
from matplotlib import pyplot as plt

def rbfkernel(x, y, ls=4.):
    dist = distance_matrix(np.expand_dims(x, 1), np.expand_dims(y, 1))
    return np.exp(-dist**2 / (2 * ls**2))

x_points = np.linspace(0, 5, 50)
meanvec = np.zeros(50)
covmat = rbfkernel(x_points, x_points,1)

prior_samples = np.random.multivariate_normal(meanvec, covmat, 5)
plt.plot(x_points, prior_samples.T, alpha=0.5)
plt.show(block=False)

def data_maker1(x, sig):
    return  np.sin(x) + 0.5*np.sin(4.*x) + np.random.randn(x.shape[0]) * sig

sig = 0.25
train_x, test_x = np.linspace(0, 5, 50), np.linspace(0, 5, 500)
train_y, test_y = data_maker1(train_x, sig), data_maker1(test_x, sig=0.)

plt.scatter(train_x, train_y, label='train')
plt.plot(test_x, test_y, label='test')
plt.show(block=False)

mean = np.zeros(test_x.shape[0])
cov = rbfkernel(test_x, test_x, ls=0.2)
# plt.imshow(cov)

prior_samples = np.random.multivariate_normal(mean, cov, 5)
plt.plot(test_x, prior_samples.T, alpha=0.5)
plt.plot(test_x, mean, 'k')

ell_est = 0.4
post_sig_est = 0.5

def neg_log_marginal_likelihood(params, x, y, sig):
    ell, post_sig = params
    cov = rbfkernel(x, x, ls=ell)
    cov += post_sig**2 * np.eye(cov.shape[0])
    return 0.5 * np.log(np.linalg.det(cov)) + 0.5 * y.T @ np.linalg.inv(cov) @ y + 0.5 * x.shape[0] * np.log(2 * np.pi)

def neg_MLL(pars):
    K = rbfkernel(train_x, train_x, ls=pars[0])
    kernel_term = -0.5 * train_y @ \
        np.linalg.inv(K + pars[1]**2 * np.eye(train_x.shape[0])) @ train_y
    logdet = -0.5 * np.log(np.linalg.det(K + pars[1]**2 * np.eye(train_x.shape[0])))
    const = -train_x.shape[0] * 0.5 * np.log(2 * np.pi)
    return -(kernel_term + logdet + const)

learned_hypers = optimize.minimize(neg_MLL, [ell_est, post_sig_est], method='L-BFGS-B', bounds=((0.01, 10), (0.01, 10)))

ell = learned_hypers.x[0]
post_sig_est = learned_hypers.x[1]

K_x_xstar = rbfkernel(train_x, test_x, ls=ell)
K_x_x = rbfkernel(train_x, train_x, ls=ell)
K_xstar_xstar = rbfkernel(test_x, test_x, ls=ell)

post_mean = K_x_xstar.T @ np.linalg.inv((K_x_x + \
                post_sig_est ** 2 * np.eye(train_x.shape[0]))) @ train_y
post_cov = K_xstar_xstar - K_x_xstar.T @ np.linalg.inv((K_x_x + \
                post_sig_est ** 2 * np.eye(train_x.shape[0]))) @ K_x_xstar

lw_bd = post_mean - 2 * np.sqrt(np.diag(post_cov))
up_bd = post_mean + 2 * np.sqrt(np.diag(post_cov))

plt.scatter(train_x, train_y)
plt.plot(test_x, test_y, linewidth=2.)
plt.plot(test_x, post_mean, linewidth=2.)
plt.fill_between(test_x, lw_bd, up_bd, alpha=0.25)
plt.legend(['Observed Data', 'True Function', 'Predictive Mean', '95% Set on True Func'])
plt.show()

lw_bd_observed = post_mean - 2 * np.sqrt(np.diag(post_cov) + post_sig_est ** 2)
up_bd_observed = post_mean + 2 * np.sqrt(np.diag(post_cov) + post_sig_est ** 2)

post_samples = np.random.multivariate_normal(post_mean, post_cov, size=20)
plt.scatter(train_x, train_y)
plt.plot(test_x, test_y, linewidth=2.)
plt.plot(test_x, post_mean, linewidth=2.)
plt.plot(test_x, post_samples.T, color='gray', alpha=0.25)
plt.fill_between(test_x, lw_bd, up_bd, alpha=0.25)
plt.legend(['Observed Data', 'True Function', 'Predictive Mean', 'Posterior Samples'])
plt.show()

train_x = torch.from_numpy(train_x).float()
train_y = torch.from_numpy(train_y).float()
test_y = torch.from_numpy(test_y).float()

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = ExactGPModel(train_x, train_y, likelihood)
train_iter = 100
model.train()
likelihood.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

for i in range(train_iter):
    # Zero gradients from previous iteration
    optimizer.zero_grad()
    # Output from model
    output = model(train_x)
    # Calc loss and backprop gradients
    loss = -mll(output, train_y)
    loss.backward()
    if i % 10 == 0:
        print(f'Iter {i+1:d}/{train_iter:d} - Loss: {loss.item():.3f} '
              f'squared lengthscale: '
              f'{model.covar_module.base_kernel.lengthscale.item():.3f} '
              f'noise variance: {model.likelihood.noise.item():.3f}')
    optimizer.step()

test_x = torch.from_numpy(test_x).float()
model.eval()
likelihood.eval()
observed_pred = likelihood(model(test_x))
with torch.no_grad():
    f, ax = plt.subplots(1, 1, figsize=(4, 3))
    lower, upper = observed_pred.confidence_region()
    ax.scatter(train_x.numpy(), train_y.numpy())
    ax.plot(test_x.numpy(), test_y.numpy(), linewidth=2.)
    ax.plot(test_x.numpy(), observed_pred.mean.numpy(), linewidth=2.)
    ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.25)
plt.show(block=False)