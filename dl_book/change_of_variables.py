
import os
from urllib.request import urlretrieve

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
from tqdm import tqdm

tfd = tfp.distributions
tfb = tfp.bijectors

import feather
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam


N = 10000
d = tfd.Uniform(low=0., high=2.)
zs = d.sample(N)

zd = np.linspace(-1., 3., 1000)
plt.hist(zs, bins=100, density=True);
plt.plot(zd, d.prob(zd)); plt.show()

x = zs**2
xs = np.linspace(0., 4., 1000)
plt.plot(xs, 1./np.sqrt(xs));
plt.hist(x, bins=100, density=True); plt.show()

g = tfb.Square()
g.forward(2).numpy()
g.inverse(4.).numpy()

base_dist = tfd.Uniform(low=0., high=2.)
dist = tfd.TransformedDistribution(distribution=base_dist, bijector=tfb.Square())
dist.prob(4.).numpy()
dist.sample(10).numpy()

xs = np.linspace(0., 4., 1000)
px = dist.prob(xs)

plt.plot(xs, px); plt.show()

chain = tfb.Chain([tfb.Square(), tfb.Square()])
chain.forward(2.).numpy()

## Afiine transformation

N = 10000
X = tfd.Normal(loc=5., scale=.2).sample(N)
X = np.reshape(X, (-1, 1))
plt.hist(X, bins=100, density=True); plt.show()

b = tf.Variable(0.)
a = tf.Variable(1.)

bijector = tfb.Shift(b)(tfb.Scale(a))

dist = tfd.TransformedDistribution(distribution=tfd.Normal(loc=0., scale=1.), bijector=bijector)

def nll(y, rv_y):
    return -tf.reduce_mean(rv_y.log_prob(y))

optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)

for i in range(1000):
    with tf.GradientTape() as tape:
        loss = -tf.reduce_mean(dist.log_prob(X))
    grads = tape.gradient(loss, [a, b])
    optimizer.apply_gradients(zip(grads, [a, b]))
    if i % 100 == 0:
        print(loss.numpy(), a.numpy(), b.numpy())

xx = np.linspace(4., 6., 1000)
plt.hist(X, bins=100, density=True);
plt.plot(xx, dist.prob(xx)); plt.show()

b.numpy(), a.numpy()
dist.mean().numpy(), dist.stddev().numpy()
dist.prob(4.6).numpy()
dist.quantile(.2).numpy()
