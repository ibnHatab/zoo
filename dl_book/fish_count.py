
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

import tensorflow as tf
import tensorflow_probability as tfp

from utils import show_train_history

tfd = tfp.distributions
tfb = tfp.bijectors

np.random.seed(42)
tf.random.set_seed(42)

dist = tfd.poisson.Poisson(rate=2.0)
vals = np.linspace(0, 10, 11)
p = dist.prob(vals)
dist.mean().numpy(), dist.stddev().numpy()

plt.xticks(vals)
plt.stem(vals, p)
plt.show()

# dist.sample(5).numpy()

dat = np.loadtxt('https://raw.githubusercontent.com/tensorchiefs/dl_book/master/data/fish.csv', delimiter=',', skiprows=1)
X = dat[...,1:5] #"livebait","camper","persons","child
y = dat[...,7] # "fish"
X=np.array(X,dtype="float32")
y=np.array(y,dtype="float32")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,shuffle=True)
d = X_train.shape[1]
X_train.shape, y_train.shape,
X_test.shape, y_test.shape,
dat.shape,
# y_test[0:10], y_train[0:10]

vals, counts = np.unique(y_train, return_counts=True)
list(zip(vals, counts))
plt.stem(vals, counts); plt.show()

from sklearn.linear_model import LinearRegression
model_skl = LinearRegression()
res = model_skl.fit(X_train, y_train)
model_skl.coef_, model_skl.intercept_

import pandas as pd

y_hat_train = model_skl.predict(X_train)
n = len(y_train)

sigma_hat_2 = (n-1)/(n-2) * np.var(y_train - y_hat_train.flatten(), ddof=1)
sigma_hat_2, np.sqrt(sigma_hat_2)

y_hat = model_skl.predict(X_test)
RMSE_sklearn = np.sqrt(np.mean((y_hat - y_test)**2))
MAE_Sklearn = np.mean(np.abs(y_hat - y_test))
RMSE_sklearn, MAE_Sklearn

NLL_sklearn =  0.5*np.log(2 * np.pi * sigma_hat_2) + 0.5*np.mean((y_test - y_hat.flatten())**2)/sigma_hat_2
df1 = pd.DataFrame(
          {'RMSE' : RMSE_sklearn, 'MAE' : MAE_Sklearn, 'NLL (mean)' : NLL_sklearn}, index=['Linear Regression (sklearn)']
)
df1

## Linear Regression (tensorflow)
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import Model, Sequential
from tensorflow.keras.optimizers import Adam

model_lr = Sequential()
model_lr.add(Input(shape=(d,)))
model_lr.add(Dense(1, activation='linear'))
model_lr.compile(loss='mse', optimizer=Adam(learning_rate=0.01))
model_lr.summary()

history = model_lr.fit(x=X_train, y=y_train, validation_data=(X_test, y_test), epochs=5000, verbose=0)
show_train_history(history)

y_train_hat = model_lr.predict(X_train)
sigma_hat_2 = (n-1)/(n-2) * np.var(y_train - y_train_hat.flatten(), ddof=1)

y_hat = model_lr.predict(X_test)
RMSE_lr = np.sqrt(np.mean((y_hat - y_test)**2))
MAE_lr = np.mean(np.abs(y_hat - y_test))
NLL_lr =  0.5*np.log(2 * np.pi * sigma_hat_2) + 0.5*np.mean((y_test - y_hat.flatten())**2)/sigma_hat_2
df2 = pd.DataFrame(
        {'RMSE' : RMSE_lr, 'MAE' : MAE_lr, 'NLL (mean)' : NLL_lr}, index=['Linear Regression (keras)']
)
df2

model_lr.get_weights()[0][:,0], model_skl.coef_
model_lr.get_weights()[1][0], model_skl.intercept_

## Poisson Regression

inputs = Input(shape=(d,))
rate = Dense(1, activation=tf.exp)(inputs)
p_y = tfp.layers.DistributionLambda(tfd.Poisson)(rate)

model_p = Model(inputs=inputs, outputs=p_y)
model_p.summary()

def nll(y_true, y_pred):
    return -y_pred.log_prob(y_true)

model_p.compile(loss=nll, optimizer=Adam(learning_rate=0.01))

history = model_p.fit(x=X_train, y=y_train, validation_data=(X_test, y_test), epochs=2000, verbose=0)
show_train_history(history)

y_hat_test = model_p.predict(X_test)

NLL = model_p.evaluate(X_test, y_test)
rmse=np.sqrt(np.mean((y_test - y_hat_test)**2))
mae=np.mean(np.abs(y_test - y_hat_test))

df3 = pd.DataFrame(
         { 'RMSE' : rmse, 'MAE' : mae, 'NLL (mean)' : NLL}, index=['Poisson Regression (TFP)']
)
df3

pd.concat([df1,df2,df3])

## Zero Inflated Poisson Regression

def zero_inf(out):
    rate = tf.squeeze(tf.math.exp(out[..., 0:1]), axis=-1)
    logits = tf.math.sigmoid(out[..., 1:2])
    probs = tf.concat([1 - logits, logits], axis=-1)
    return tfd.Mixture(
        cat=tfd.Categorical(probs=probs),
        components=[tfd.Deterministic(loc=tf.zeros_like(rate)),
                    tfd.Poisson(rate=rate)])


out = np.random.randn(100, 2)
zero_inf(out).sample(5).numpy()
tf.exp(1.0)
tf.math.sigmoid(-10.0)

t = np.ones((2,2), dtype=np.float32)
t[0,0] = 1
t[0,1] = 10#almost always take pois
t[1,0] = 1
t[1,1] = -10# almost always take zero
zero_inf(t).mean().numpy()

inputs = Input(shape=(d,))
out = Dense(2)(inputs)
p_y_zi = tfp.layers.DistributionLambda(zero_inf)(out)
model_zi = Model(inputs=inputs, outputs=p_y_zi)
model_zi.summary()

optimizer = Adam(learning_rate=0.05)
steps = 10
losses = np.zeros(steps)
for e in range(steps):
    with tf.GradientTape() as tape:
        y_hat = model_zi(X_train)
        loss = -tf.reduce_mean(y_hat.log_prob(y_train))
    grads = tape.gradient(loss, model_zi.trainable_weights)
    optimizer.apply_gradients(zip(grads, model_zi.trainable_weights))
    losses[e] = loss.numpy()
    print(e, loss.numpy())

model_zi.get_weights()

plt.plot(losses); plt.show()

def nll(y_true, y_pred):
    return -y_pred.log_prob(tf.reshape(y_true, (-1,)))

model_zi.compile(loss=nll, optimizer=Adam(learning_rate=0.01))
history = model_zi.fit(x=X_train, y=y_train, validation_data=(X_test, y_test), epochs=2000, verbose=1)

show_train_history(history)

y_hat_test = model_zi.predict(X_test).flatten()
mse=np.sqrt(np.mean((y_test - y_hat_test)**2))
mae=np.mean(np.abs(y_test - y_hat_test))

NLL = model_zi.evaluate(X_test, y_test) #returns the NLL


df4 = pd.DataFrame(
         { 'RMSE' : mse, 'MAE' : mae, 'NLL (mean)' : NLL}, index=['ZIP (TFP)']
)
pd.concat([df1,df2,df3,df4])

# Linear Regression (sklearn)   8.588127  4.705091    3.617528
# Linear Regression (keras)    11.156907  6.135615    3.617494
# Poisson Regression (TFP)     12.195639  5.232000    2.701954
# ZIP (TFP)                     8.028699  3.620000    2.202046

probs = model_zi(X_test).log_prob(np.arange(0,20,1).reshape(20,1)).numpy()

plt.stem(np.arange(0,20,1), tf.exp(probs[:,30])); plt.show()

probs.shape
plt.plot(probs)
plt.show()

