

import os
from urllib.request import urlretrieve

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
from tqdm import tqdm

from utils import get_if_not_there

tfd = tfp.distributions
tfb = tfp.bijectors

import feather
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam

""" Old Faithful Geyser Data

     Waiting time between eruptions and the duration of the eruption
     for the Old Faithful geyser in Yellowstone National Park, Wyoming,
     USA.

     A data frame with 272 observations on 2 variables.

eruptions  numeric  Eruption time in mins
waiting    numeric  Waiting time to next eruption

 """


data_file = get_if_not_there('https://raw.githubusercontent.com/tensorchiefs/dl_book/master/data', 'faithful.csv')

df = pd.read_csv(data_file)
df.head()

X = df.iloc[:,2].to_numpy(dtype='float32')
plt.hist(X, bins=50); plt.show()

# zs = np.linspace(-5., 5., 100, dtype='float32')
# plt.plot(zs, tfb.SinhArcsinh(skewness=0., tailweight=1.).forward(zs));
# plt.plot(zs, tfb.SinhArcsinh(skewness=1., tailweight=1.).forward(zs));
# plt.plot(zs, tfb.SinhArcsinh(skewness=-1., tailweight=1.4).forward(zs));
# plt.show()

num_bijectors = 5
bs = []
for i in range(num_bijectors):
    shift = tf.Variable(0., dtype='float32')
    scale = tf.Variable(1., dtype='float32')
    bs.append(tfb.Shift(shift)(tfb.Scale(scale)))
    skewness = tf.Variable(0., dtype='float32')
    tailweight = tf.Variable(1., dtype='float32')
    bs.append(tfb.SinhArcsinh(skewness=skewness,
                              tailweight=tailweight))

bijector = tfb.Chain(bs)
dist = tfd.TransformedDistribution(distribution=tfd.Normal(loc=0., scale=1.), bijector=bijector)
optimizer = Adam(learning_rate=0.01)

@tf.function
def train_step(X):
    with tf.GradientTape() as tape:
        loss = -tf.reduce_mean(dist.log_prob(X))
    grads = tape.gradient(loss, dist.trainable_variables)
    optimizer.apply_gradients(zip(grads, dist.trainable_variables))
    return loss

for i in range(20000):
    loss = train_step(X)
    if i % 1000 == 0:
        print(i, loss)

xs = np.linspace(30., 120., 1000)
plt.hist(X, bins=50, density=True)
plt.hist(dist.sample(len(X)), bins=50, density=True, alpha=0.5)
plt.plot(xs, dist.prob(xs))
plt.show()

z = np.linspace(-5., 5., 1000)
plt.plot(z, bijector.forward(z)); plt.show()