
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


def get_if_not_there(filename='deer_train.feather'):
    if not os.path.exists(filename):
        print('Downloading dear data...')
        url = 'https://raw.githubusercontent.com/tensorchiefs/dl_book/master/data/{}'.format(filename)
        urlretrieve(url, filename)


get_if_not_there('deer_train.feather')
get_if_not_there('deer_test.feather')

df_train = feather.read_dataframe('deer_train.feather')
df_train.head()
df_test = feather.read_dataframe('deer_test.feather')

df_train['daytime'].unique()
df_train['weekday'].unique()

y_train = df_train.iloc[:,0].to_numpy(dtype='float32')
y_test = df_test.iloc[:,0].to_numpy(dtype='float32')
X_train = pd.get_dummies(df_train.iloc[:,2:])
X_test = pd.get_dummies(df_test.iloc[:,2:])

X_train.iloc[:,0] = X_train.iloc[:,0] / X_train.iloc[:,0].max()
X_test.iloc[:,0] = X_test.iloc[:,0] / X_test.iloc[:,0].max()

X_train.shape, y_train.shape, X_test.shape, y_test.shape
del df_train, df_test

X_train = X_train.to_numpy(dtype='float32')
X_test = X_test.to_numpy(dtype='float32')

vals, count = np.unique(y_train, return_counts=True)

plt.stem(vals, count); plt.show()

## Linear Regression
model_lr = Sequential()
model_lr.add(Input(shape=(X_train.shape[1],)))
model_lr.add(Dense(1, activation='linear'))
model_lr.compile(loss='mse', optimizer=Adam(learning_rate=0.01))
model_lr.summary()
history = model_lr.fit(X_train, y_train, validation_data=[X_test, y_test], epochs=10, verbose=1)

from utils import show_train_history

show_train_history(history)

n = len(y_train)
y_hat_train = model_lr.predict(X_train).reshape(n)
sigma_hat_2 = (n-1.)/(n-2.) * np.var(y_train - y_hat_train, ddof=1)
n = len(y_test)
y_hat = model_lr.predict(X_test).reshape(n)
NLL_lr = 0.5 * np.log(2 * np.pi * sigma_hat_2) + 0.5 * np.mean((y_test - y_hat)**2) / sigma_hat_2

def predicted_vs_actual(y_hat, y_test, sigma_hat_2):
    plt.scatter(y_hat, y_test, alpha=0.5);
    sorrt_idx = np.argsort(y_hat, axis=0)
    plt.plot(y_hat[sorrt_idx], y_hat[sorrt_idx], color='black')
    plt.plot(y_hat[sorrt_idx], y_hat[sorrt_idx] + 2*np.sqrt(sigma_hat_2), color='red')
    plt.plot(y_hat[sorrt_idx], y_hat[sorrt_idx] - 2*np.sqrt(sigma_hat_2), color='red')
    plt.show()

predicted_vs_actual(y_hat, y_test, sigma_hat_2)

## Poisson Regression

imput = Input(shape=(X_train.shape[1],))
x = Dense(100, activation='relu')(imput)
x = Dense(100, activation='relu')(x)
x = Dense(10, activation='relu')(x)
rate = Dense(1, activation='linear')(x)
p_y = tfp.layers.DistributionLambda(lambda t: tfd.Poisson(rate=tf.exp(t)))(rate)
model_p = Model(inputs=imput, outputs=p_y)
model_p.compile(loss=lambda y, p_y: -p_y.log_prob(y), optimizer=Adam(learning_rate=0.01))
model_p.summary()

history = model_p.fit(X_train, y_train, validation_data=[X_test, y_test], epochs=10, verbose=1)

show_train_history(history)

n = len(y_train)
y_hat_train = model_p.predict(X_train).reshape(n)
sigma_hat_2 = (n-1.)/(n-2.) * np.var(y_train - y_hat_train, ddof=1)

n = len(y_test)
y_hat_test = model_p.predict(X_test).reshape(n)
predicted_vs_actual(y_hat_test, y_test, sigma_hat_2)

NLL_lr = 0.5 * np.log(2 * np.pi * sigma_hat_2) + 0.5 * np.mean((y_test - y_hat_test)**2) / sigma_hat_2
NLL_train = model_p.evaluate(X_train, y_train,verbose=0)
NLL_test = model_p.evaluate(X_test, y_test,verbose=0)

from scipy.stats import poisson
lower=poisson.ppf(0.025, y_hat_test)
upper=poisson.ppf(0.975, y_hat_test)

plt.scatter(y_hat_test, y_test, alpha=0.3)
plt.plot(y_hat_test[np.argsort(y_hat_test,axis=0)].flatten(), lower[np.argsort(y_hat_test,axis=0)],linestyle='dashed',c="black")
plt.plot(y_hat_test[np.argsort(y_hat_test,axis=0)].flatten(), upper[np.argsort(y_hat_test,axis=0)],linestyle='dashed',c="black")
plt.plot(y_hat_test, y_hat_test, c="black")
plt.title('Comparison on the testset')
plt.xlabel('predicted average of deers killed')
plt.ylabel('observed number of deers killed')
plt.show()


## Zero Inflated Poisson Regression

def zero_inf(out):
    rate = tf.squeeze(tf.math.exp(out[:, 0:1]))
    s = tf.math.sigmoid(out[:, 1:2])
    prob = tf.concat([1-s, s], axis=1)
    return tfd.Mixture(cat=tfd.Categorical(probs=prob),
                       components=[
        tfd.Deterministic(loc=tf.zeros_like(rate)),
        tfd.Poisson(rate=rate)
    ])

inputs = Input(shape=(X_train.shape[1],))
out = Dense(2)(inputs)
p_y_zi = tfp.layers.DistributionLambda(zero_inf)(out)
model_zi = Model(inputs=inputs, outputs=p_y_zi)
model_zi.compile(loss=lambda y, p_y: -p_y.log_prob(y), optimizer=Adam(learning_rate=0.01))
model_zi.summary()

history = model_zi.fit(X_train, y_train, validation_data=[X_test, y_test], epochs=10, verbose=1)

show_train_history(history)

NLL_train = model_zi.evaluate(X_train, y_train,verbose=0)
NLL_test = model_zi.evaluate(X_test, y_test,verbose=0)
NLL_train, NLL_test

n = len(y_train)
y_hat_train = model_zi.predict(X_train).reshape(n)
sigma_hat_2 = (n-1.)/(n-2.) * np.var(y_train - y_hat_train, ddof=1)

y_hat_test = model_zi.predict(X_test).reshape(n)
predicted_vs_actual(y_hat_test, y_test, sigma_hat_2)

NLL_lr = 0.5 * np.log(2 * np.pi * sigma_hat_2) + 0.5 * np.mean((y_test - y_hat_test)**2) / sigma_hat_2
NLL_train = model_p.evaluate(X_train, y_train,verbose=0)
NLL_test = model_p.evaluate(X_test, y_test,verbose=0)

from scipy.stats import poisson
lower=poisson.ppf(0.025, y_hat_test)
upper=poisson.ppf(0.975, y_hat_test)

plt.scatter(y_hat_test, y_test, alpha=0.3)
plt.plot(y_hat_test[np.argsort(y_hat_test,axis=0)].flatten(), lower[np.argsort(y_hat_test,axis=0)],linestyle='dashed',c="black")
plt.plot(y_hat_test[np.argsort(y_hat_test,axis=0)].flatten(), upper[np.argsort(y_hat_test,axis=0)],linestyle='dashed',c="black")
plt.plot(y_hat_test, y_hat_test, c="black")
plt.title('Comparison on the testset')
plt.xlabel('predicted average of deers killed')
plt.ylabel('observed number of deers killed')
plt.show()

## Regression with a Discrete Logit Likelihood

def quant_mixture_logistic(out, bits=8, num=3):
    loc, un_scale, logits = tf.split(out, num_or_size_splits=num, axis=-1)
    scale = tf.math.softplus(un_scale)
    descret = tfd.QuantizedDistribution(
        distribution=tfd.TransformedDistribution(
            distribution=tfd.Logistic(loc=loc, scale=scale),
            bijector=tfb.Shift(-.5)),
        low=0.,
        high=2**bits - 1.)
    mixture = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(logits=logits),
        components_distribution=descret)
    return mixture

inputs = Input(shape=(X_train.shape[1],))
out = Dense(9)(inputs)
p_y = tfp.layers.DistributionLambda(quant_mixture_logistic)(out)
model = Model(inputs=inputs, outputs=p_y)
model.compile(loss=lambda y, p_y: -p_y.log_prob(y), optimizer=Adam(learning_rate=0.01))
model.summary()

history = model.fit(X_train, y_train, validation_data=[X_test, y_test], epochs=10, verbose=1)
show_train_history(history)

print(-np.mean(model(X_train).log_prob(y_train)))
print(-np.mean(model(X_test).log_prob(y_test)))

NLL_train = model.evaluate(X_train, y_train,verbose=0)
NLL_test = model.evaluate(X_test, y_test,verbose=0)

print('NLL on training:', NLL_train)
print('NLL on test:', NLL_test)

preds = np.zeros((1000,len(y_test.flatten())))
for i in tqdm(range(0,1000)):
  preds[i,:] = model(X_test).sample().numpy()# sample from the QuantizedDistribution
y_hat_test=np.average(preds,axis=0)

upper=[]
lower=[]
for i in tqdm(range(0,np.int32(len(X_test)/10))):
  samples_tmp=model(X_test[(i*10):(i*10)+10]).sample(5000).numpy()
  upper=np.append(upper,np.quantile(samples_tmp,0.975,axis=0))
  lower=np.append(lower,np.quantile(samples_tmp,0.025,axis=0))

plt.scatter(y_hat_test, y_test, alpha=0.3)
plt.plot(y_hat_test[np.argsort(y_hat_test,axis=0)].flatten(), lower[np.argsort(y_hat_test,axis=0)],linestyle='dashed',c="black")
plt.plot(y_hat_test[np.argsort(y_hat_test,axis=0)].flatten(), upper[np.argsort(y_hat_test,axis=0)],linestyle='dashed',c="black")
plt.plot(y_hat_test, y_hat_test, c="black")
plt.title('Comparison on the testset')
plt.xlabel('predicted average of deers killed')
plt.ylabel('observed number of deers killed')
plt.show()