
import os
from urllib.request import urlretrieve

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from utils import get_if_not_there

tfd = tfp.distributions
tfb = tfp.bijectors

from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam

def sample(num):
    return np.array(np.random.uniform(-1, 1, (num, 2)), dtype=np.float32)

def sample_2(batch_size=500):
    x2_dist = tfd.Normal(loc=0., scale=4.)
    x2_samples = x2_dist.sample(batch_size)
    x1 = tfd.Normal(loc=.25*tf.square(x2_samples),
                    scale=tf.ones(batch_size, dtype=tf.float32))
    x1_samples = x1.sample()
    x_samples = tf.stack([x1_samples, x2_samples], axis=1)
    return x_samples.numpy() / 40.

X = sample_2(1500)
plt.scatter(X[:, 0], X[:, 1], s=1); plt.show()

class RealNVP(tf.keras.models.Model):
    def __init__(self, *, output_dim, num_masked, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.nets = []
        bijectors = []
        num_blocks = 5
        h = 32
        for i in range(num_blocks):
            net = tfb.real_nvp_default_template(hidden_layers=[h, h])
            bijectors.append(tfb.RealNVP(num_masked=num_masked,
                                         shift_and_log_scale_fn=net))
            bijectors.append(tfb.Permute(permutation=[1, 0]))
            self.nets.append(net)
        bijector = tfb.Chain(list(reversed(bijectors[:-1])))
        self.flow = tfd.TransformedDistribution(
            distribution=tfd.MultivariateNormalDiag(loc=[0., 0.]),
            bijector=bijector)

    def call(self, *inputs):
        return self.flow.bijector.forward(*inputs)

    # def log_prob(self, *inputs):
    #     return self.flow.log_prob(*inputs)

model = RealNVP(output_dim=2, num_masked=1)
y = model(X)
model.summary()

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
-tf.reduce_mean(model.flow.log_prob(y)).numpy()

@tf.function
def train_step(X):
    with tf.GradientTape() as tape:
        loss = -tf.reduce_mean(model.flow.log_prob(X))
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

for i in range(1000):
    loss = train_step(X)
    if i % 100 == (100-1):
        print(i, loss)

Z = np.random.normal(0,1,(5000,2))
cols = []
for i in range(5000):
    if (Z[i,0] > 0 and Z[i,1] > 0):
        cols.append('r')
    elif (Z[i,0] < 0 and Z[i,1] > 0):
        cols.append('b')
    elif (Z[i,0] < 0 and Z[i,1] < 0):
        cols.append('y')
    else:
        cols.append('g')

plt.figure(figsize=(14,6))
params = {'mathtext.default': 'regular' }
plt.rcParams.update(params)

plt.subplot(1,2,1)
plt.scatter(Z[:, 0], Z[:, 1], s=5,c=cols)
plt.title('$Z \sim N(0,1)$')
plt.xlabel('$z_1$')
plt.ylabel('$z_2$')
Xs = model(Z)

plt.subplot(1,2,2)
plt.title('Transformed distribution')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.scatter(Xs[:,0], Xs[:, 1], s=5, c=cols)
plt.xlim(-0.1,1)
plt.ylim(-0.35,0.35)
plt.show()



class Mask_AR(tf.keras.models.Model):

    def __init__(self, *, output_dim, num_masked, **kwargs): #** additional arguments for the super class
        super().__init__(**kwargs)
        self.output_dim = output_dim
        # We need to keep track of the nets
        self.nets = []

        # Defining the bijector
        bijectors=[]

        h = 32
        for i in range(3):

            ##### Here is some difference to RealNVP
            net = tfb.masked_autoregressive_default_template(hidden_layers=[h, h])
            # masked_autoregressive_default_template constructs a special network,
            # which preserves the autoregressive property, called MADE.

            bijectors.append(tfb.MaskedAutoregressiveFlow(shift_and_log_scale_fn=net))
            #bijectors.append(tfb.Invert(tfb.MaskedAutoregressiveFlow(shift_and_log_scale_fn=net)))
            #Uncommentung the line above (and commenting two lines above), would create a so-called inverse
            #autoregressive flow which is faster in prediction, but slower in training

            ##### End of difference

            bijectors.append(tfb.Permute([1,0]))
            self.nets.append(net)
        bijector = tfb.Chain(list(reversed(bijectors[:-1])))

        self.flow = tfd.TransformedDistribution(
            distribution=tfd.MultivariateNormalDiag(loc=[0., 0.]),
            bijector=bijector)

    def call(self, *inputs):
        return self.flow.bijector.forward(*inputs)


model = Mask_AR(output_dim=2, num_masked=1)
_ = model(X)
print(model.summary())

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
##### Important
#Needs to be called other-wise @tf.function has problem
-tf.reduce_mean(model.flow.log_prob(X))

@tf.function #Adding the tf.function makes it about 10-50 times faster!!!
def train_step(X):
    with tf.GradientTape() as tape:
        loss = -tf.reduce_mean(model.flow.log_prob(X))
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

from time import time
start = time()
for i in range(1001):
    #Xs = sample(1000) #Creat new training data
    loss = train_step(X)
    if (i % 100 == 0):
        print(i, " ",loss.numpy(), (time()-start))
        start = time()