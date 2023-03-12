
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# %cd src
tfb = tfp.bijectors
tfd = tfp.distributions

xs = np.linspace(0., 500, 500)
postm = tfd.Gamma((143,85), (1.1, 0.5))
post = tfd.MixtureSameFamily(
    mixture_distribution=tfd.Categorical(probs=[0.45, 0.55]),
    components_distribution=postm)

plt.plot(xs, post.prob(xs)); plt.show()

mu_sigma = tf.Variable([263., 34.], dtype=tf.float32)
vari = tfd.Normal(loc=mu_sigma[0], scale=mu_sigma[1])
optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)

@tf.function
def train():
    with tf.GradientTape() as tape:
        vari = tfd.Normal(loc=mu_sigma[0], scale=mu_sigma[1])
        qs = vari.sample(1000)
        H = tf.clip_by_value(vari.prob(qs) / post.prob(qs), 1e-10, 1e10)
        loss = tf.reduce_mean(H)
    grads = tape.gradient(loss, [mu_sigma])
    optimizer.apply_gradients(zip(grads, [mu_sigma]))
    return loss

for i in range(30000):
    loss = train()
    if i % 1000 == 999:
        print(i, loss)

vari = tfd.Normal(loc=mu_sigma[0], scale=mu_sigma[1])
vari.mean(), vari.stddev()

plt.plot(xs, vari.prob(xs));
plt.plot(xs, post.prob(xs)); plt.show()

num = 4
sigma = 3.
bmin=-10;bmax=8
amin=-3;amax=8
xmin = -5.; xmax = 5.; ymin = -40.; ymax = 40.

x = np.linspace(-2, 2, num).reshape(-1, 1)
y = 2*x[:,0] -1 + 1.*np.random.normal(0, sigma, num)
plt.scatter(x, y, color='k', label='data'); plt.show()

## base model
from sklearn.linear_model import LinearRegression
regr = LinearRegression(fit_intercept=True)
regr.fit(x, y)
regr.coef_, regr.intercept_

def plot_regression(x, y, coef, intercept, xmin, xmax, ymin, ymax, label):
    plt.plot([xmin, xmax], [intercept + xmin*coef, intercept + xmax*coef], label=label)
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.scatter(x, y, color='k', label='data')
    return plt

plt = plot_regression(x, y, regr.coef_, regr.intercept_, xmin, xmax, ymin, ymax, 'base model')
plt.scatter(x, y, color='k', label='data'); plt.show()

 ## analytical solution
 def make_design_matrix(x):
    return np.hstack((np.ones((len(x), 1)), x))

def posterior(x, y, one_over_var0, var):
    """
        x vector with training x data
        y vector with training y values
        one_over_var0 1/𝜎0^2 the variances of the prior distribution
        var is the assumed to be known variance of data
        @returns mean vector mu and covariance Matrix Sig
    """
    X = make_design_matrix(x)
    Sig_inv = one_over_var0 * np.eye(X.shape[1]) + X.T.dot(X) / var
    Sig = np.linalg.inv(Sig_inv)
    mu = Sig.dot(X.T).dot(y) / var
    return mu, Sig

def posterior_predictive(x_test, mu, Sig, var):
    """
        x_test the positions, where the posterior is to be evaluated
        mu the mean values of the weight-posterior
        Sig the covariance matrix
        var is the assumed to be known variance of data
        Computes mean and variances of the posterior predictive distribution of y
    """
    X_test = make_design_matrix(x_test)
    y = X_test.dot(mu)
    # Only compute variances (diagonal elements of covariance matrix)
    y_var = var + np.sum(X_test.dot(Sig) * X_test, axis=1)
    return y, y_var

m, Sigma = posterior(x, y, 1./1.0**2, sigma**2)
print('Analytical Solution:\n mean (b,a):',m, '\n sigma:\n',Sigma)

mu_b_ana,mu_a_ana = m
sig_b_ana = Sigma[0,0]
sig_a_ana = Sigma[1,1]
mu_b_ana, mu_a_ana
sig_a_ana, sig_b_ana

plot_regression(x, y, mu_a_ana, mu_b_ana, xmin, xmax, ymin, ymax, 'analytical solution'); plt.show()

## variational inference
mu = 2.
s = 3.
tfd.Normal(loc=mu, scale=s).kl_divergence(tfd.Normal(loc=0., scale=1.)).numpy()
-(1.+np.log(s**2) - (mu**2 + s**2))/2.

## with sample
n23 = tfd.Normal(loc=mu, scale=s)

import scipy.stats as stats
n23_rep_sample = mu + s*tfd.Normal(loc=0., scale=1.).sample(1000)
stats.probplot(n23_rep_sample, dist=stats.norm, plot=plt); plt.show()
np.mean(n23_rep_sample), np.std(n23_rep_sample)

class Logger:
    """
        Writes out the weights, gradient, and losses. To be used later in e.g. R. n is the numbver of weights
    """
    def __init__(self, steps, num_weights = 4):
        self.steps = steps
        self.num_weights = num_weights
        self.X = np.zeros((steps, 12))
        self.header = 'epoch,w0,w1,w2,w3,wg0,wg1,wg2,wg3,loss,loss_kl,loss_mse'

    def log(self, step, epoch, w, w_grad, loss, loss_kl, loss_mse):
        n = self.num_weights
        self.X[step,0] = epoch
        self.X[step,1:(n+1)] = w.numpy()
        self.X[step,(n+1):((2*n)+1)] = w_grad.numpy()
        self.X[step,((2*n)+1)] = loss.numpy()
        self.X[step,((2*n)+2)] = loss_kl.numpy()
        self.X[step,((2*n)+3)] = loss_mse.numpy()

    def write4r(self, filename):
        np.savetxt(filename, self.X, delimiter=',', header=self.header,comments="",fmt='%.4e');

    def getX(self):
        return self.X

epochs=10000
logger = Logger(epochs)
lr = 0.001
optimizer = tf.keras.optimizers.SGD(lr)

# 10          20       30        40       50         55
#123456789012345678901234567890123456789012345678901234
w_0=(1.,1.,1.,1.)  #A
log = tf.math.log
w = tf.Variable(w_0)
e = tfd.Normal(loc=0., scale=1.) #B
ytensor = y.reshape([len(y),1]) #A
for i in range(epochs):
    with tf.GradientTape() as tape:

        mu_a = w[0] #C
        sig_a = tf.math.softplus(w[1]) #D

        mu_b= w[2]  #E
        sig_b= tf.math.softplus(w[3]) #F

        l_kl = -0.5*(1.0 + #G
            log(sig_a**2) - sig_a**2 - mu_a**2 + #G
            1.0 + log(sig_b**2) - sig_b**2 - mu_b**2)#G

        a =  mu_a + sig_a * e.sample()  #H
        b =  mu_b + sig_b * e.sample()  #I

        y_prob = tfd.Normal(loc=x*a+b, scale=sigma)
        l_nll = -tf.reduce_sum(y_prob.log_prob(ytensor)) #J

        loss = l_nll + l_kl
    grads = tape.gradient(loss, w)
    logger.log(i, i, w, grads, loss, l_kl, l_nll)
    w = tf.Variable(w - lr*grads)  # G

    if i % 200 == 0 or i < 2:
        print(i, " loss ", loss.numpy(), " lr ", lr)
        print('a_mu', w[0].numpy(),'b_mu', w[2].numpy())


W = logger.getX()
loss_history = W[:,9]
loss_history_nll = W[:,10]
loss_history_kl = W[:,11]
plt.plot(loss_history)
plt.plot(loss_history_kl)
plt.plot(loss_history_nll)
plt.legend(('total', 'nll', 'kl'))
plt.ylim(0,60)
plt.show()
print('b = ', tf.math.softplus(w[0]).numpy(), ' ', tf.math.softplus(w[1]).numpy(),
      ' a = ', w[2].numpy(), ' ',  tf.math.softplus(w[3]).numpy())

W = logger.getX()
weights = W[:,1:5]
epochs = W.shape[0]
from matplotlib.pyplot import figure

#slope a
plt.plot(weights[:,0],color='r',linestyle='-.')
plt.plot([0, epochs], [mu_a_ana, mu_a_ana], linewidth=1,color='r', linestyle='-')
plt.xlabel('Epochs')
plt.ylabel('Parameter Values')

plt.plot(weights[:,2],color='b', linestyle='-.')
plt.plot([0, epochs], [mu_b_ana, mu_b_ana], linewidth=2, color='b', linestyle='-')
plt.legend(('mu_a [VI]', 'mu_a [Analytical]', 'mu_b [VI]','mu_b [Analytical]'))

plt.title('Convergence to Analytical Solution')
plt.show()

model = tf.keras.Sequential([
    tfp.layers.DenseReparametrization(1, input_shape=(None,1)),
    tfp.layers.DenseReparametrization(2),
    tfp.layers.DenseReparametrization(3),
    ])

model = tf.keras.Sequential([
  tfp.layers.DenseReparameterization(1, input_shape=(None,1)),
  tfp.layers.DenseReparameterization(2),
  tfp.layers.DenseReparameterization(3),
])
model.summary()

def NLL(y_true, y_pred):
    return -y_pred.log_prob(y_true)

def my_dist(mu):
    return tfd.Normal(loc=mu[:,0:1], scale=sigma)

kl = tfp.distributions.kl_divergence
divergence_fn = lambda q, p, _: kl(q, p) / (num * 1.0)

model = tf.keras.Sequential([
    tfp.layers.DenseReparameterization(1,
                                       kernel_divergence_fn=divergence_fn,
                                       bias_divergence_fn=divergence_fn,
                                       bias_prior_fn=tfp.layers.default_multivariate_normal_fn,
                                       bias_posterior_fn=tfp.layers.default_mean_field_normal_fn(),
    ),
    tfp.layers.DistributionLambda(my_dist),])

sgd = tf.keras.optimizers.SGD(lr=0.005)
model.compile(optimizer=sgd, loss=NLL)
hist = model.fit(x, y, epochs=1000, batch_size=num, verbose=0)
hist.history['loss']
plt.plot(hist.history['loss']); plt.show()

mu_a, sig_a, mu_b, sig_b = model.get_weights()
mu_a, sig_a, mu_b, sig_b = mu_a.flatten()[0], sig_a.flatten()[0], mu_b.flatten()[0], sig_b.flatten()[0]

