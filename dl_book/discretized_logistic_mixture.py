
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

import matplotlib.pyplot as plt

tfd = tfp.distributions
tfb = tfp.bijectors

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

vals = np.linspace(-10, 10, 1000)
for scale in [0.1, 0.5, 1., 2., 5.,2.5]:
    dist = tfd.Logistic(loc=1., scale=scale)
    plt.subplot(1,2,1)
    plt.plot(vals, dist.prob(vals), label=f'scale={scale}')
    plt.subplot(1,2,2)
    plt.plot(vals, dist.cdf(vals), label=f'scale={scale}')
plt.legend()
plt.show()

bits = 4
discretized = tfd.QuantizedDistribution(dist, low=-0., high=2**bits-1.)
n_vals = np.linspace(0, 15, 16)

def show_logi_discr_cdf(vals, scale, dist, discretized, n_vals):
    plt.subplot(1,2,1)
    plt.plot(vals, dist.prob(vals), label=f'scale={scale}')
    plt.stem(n_vals, discretized.prob(n_vals), linefmt='C1--', markerfmt='C1o', label='discrete')
    plt.subplot(1,2,2)
    plt.plot(vals, dist.cdf(vals), label=f'scale={scale}')
    plt.plot(vals, discretized.cdf(vals), label=f'scale={scale}')
    plt.show()

show_logi_discr_cdf(vals, scale, dist, discretized, n_vals)

dist.sample(10).numpy(), discretized.sample(10).numpy()

## shift and scale
scale = 2.5
dist = tfd.Logistic(loc=1., scale=scale)
logis = tfd.TransformedDistribution(distribution=dist, bijector=tfb.Shift(-2.5))
discr = tfd.QuantizedDistribution(logis, low=-0., high=2**bits-1.)
show_logi_discr_cdf(vals, scale, logis, discr, n_vals)

def quantize(inner, low=0., bits=4):
    high = 2**bits - 1
    trans = tfd.TransformedDistribution(distribution=inner,bijector=tfb.Shift(-0.5))
    return tfd.QuantizedDistribution(trans, low=low, high=high)

logi = tfd.Logistic(loc=1, scale=0.25)
discretized_logistic_dist = quantize(logi)

vals = np.linspace(-10,10,1000)
n_vals = np.linspace(0,15,16) #0,1,2,3,...,15 discrete values

show_logi_discr_cdf(vals, scale, logi, discretized_logistic_dist, n_vals)

locs = (4.,10,)
scales = (0.25,.5)
probs = (0.8,0.2)

digits = tfd.Logistic(loc=locs, scale=scales)
quant = quantize(digits)
mixture = tfd.MixtureSameFamily(mixture_distribution=tfd.Categorical(probs=probs), components_distribution=digits)
mixture_quant = tfd.MixtureSameFamily(mixture_distribution=tfd.Categorical(probs=probs), components_distribution=quant)

vals = np.linspace(0,15.1,1000)
n_vals = np.linspace(0,15,16)
show_logi_discr_cdf(vals, scale, mixture, mixture_quant, n_vals)
digits.sample(10).numpy(), quant.sample(10).numpy()
mixture.sample(10).numpy(), mixture_quant.sample(10).numpy()


## Mixture of discretized logistic distributions
def quant_mixture_logistic(out, bits=8, num=3):
    loc, un_scale, logits = tf.split(out,  # A
                                     num_or_size_splits=num,
                                     axis=-1)
    scale = tf.nn.softplus(un_scale)  # B
    discretized_logistic_dist = tfd.QuantizedDistribution(
        distribution=tfd.TransformedDistribution(  # C
            distribution=tfd.Logistic(loc=loc, scale=scale),
            bijector=tfb.Shift(shift=-0.5)),
        low=0.,
        high=2**bits - 1.)
    mixture_dist = tfd.MixtureSameFamily(  # D
        mixture_distribution=tfd.Categorical(logits=logits),
        components_distribution=discretized_logistic_dist)
    return mixture_dist


inputs = tf.keras.layers.Input(shape=(100,))
h1 = Dense(10, activation='tanh')(inputs)
out = Dense(6)(h1) #E
p_y = tfp.layers.DistributionLambda(quant_mixture_logistic)(out)

model = Model(inputs=inputs, outputs=p_y)
model.summary()
