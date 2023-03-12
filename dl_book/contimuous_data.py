
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Input
from utils import show_train_history

np.set_printoptions(precision=5, suppress=True)

tfd = tfp.distributions
tfb = tfp.bijectors

np.random.seed(42)
tf.random.set_seed(42)

d = tfd.Normal(loc=3., scale=1.5)
x = d.sample(2)
px = d.prob(x)
x, px

d = tfd.Normal(loc=0., scale=1.)
d.sample(3).numpy()
d.prob([0,1,2,3,4,5]).numpy()
d.log_prob([0,1,2,3,4,5]).numpy()
d.cdf([0,1,2,3,4,5]).numpy()
d.mean().numpy()
d.stddev().numpy() # standard deviation



def simulated_data():
    x1 = np.arange(1, 12, 0.1)[::-1]
    x2 = np.repeat(1, 30)
    x3 = np.arange(1, 15, 0.1)
    x4 = np.repeat(15, 50)
    x5 = x3[::-1]
    x6 = np.repeat(1, 20)
    x = np.concatenate([x1, x2, x3, x4, x5, x6])

    np.random.seed(4710)
    noise = np.random.normal(0, x, len(x))
    np.random.seed(99)
    first_part=len(x1)
    x11 = np.random.uniform(-1, 1, first_part)
    np.random.seed(97)
    x12 = np.random.uniform(1, 6, len(noise)-first_part)
    x = np.concatenate([x11, x12])
    x = np.sort(x)
    y = 2.7*x + noise
    x = x.reshape((len(x), 1))
    y = y.reshape((len(y), 1))
    return x,y


def train_test_val_split(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)


    order_idx_train=np.squeeze(x_train.argsort(axis=0))
    x_train=x_train[order_idx_train]
    y_train=y_train[order_idx_train]

    order_idx_val=np.squeeze(x_val.argsort(axis=0))
    x_val=x_val[order_idx_val]
    y_val=y_val[order_idx_val]

    order_idx_test=np.squeeze(x_test.argsort(axis=0))
    x_test=x_test[order_idx_test]
    y_test=y_test[order_idx_test]
    return x_train,x_test,y_train,y_test,x_val,y_val

x, y = simulated_data()
# plt.scatter(x, y); plt.show()
x_train, x_test, y_train, y_test, x_val, y_val = train_test_val_split(x, y)
# [len(s) for s in [x, x_train, x_test, x_val]]

# plt.scatter(x_train, y_train, label='train')
# plt.scatter(x_val, y_val, label='val')
# plt.scatter(x_test, y_test, label='test')
# plt.legend()
# plt.show()

## Model with constant variance

def NLL(y, distr):
    return -distr.log_prob(y)

def normal_dist(mu, sigma=5.):
    return tfd.Normal(loc=mu, scale=sigma)


imput = Input(shape=(1,))
params = Dense(1)(imput)

dist = tfp.layers.DistributionLambda(normal_dist)(params)
model_sd_1 = tf.keras.Model(inputs=imput, outputs=dist)
model_sd_1.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss=NLL)
model_sd_1.summary()

history = model_sd_1.fit(x_train, y_train, epochs=1000, validation_data=(x_val, y_val), verbose=1)

show_train_history(history)

model_sd_1.evaluate(x_train, y_train)
model_sd_1.evaluate(x_test, y_test)
model_sd_1.evaluate(x_val, y_val)


def show_prediction(model, x_train,y_train, x_val, y_val):
    x_pred = np.arange(-1,6,0.1)
    preds = model.predict(x_pred)
    sigma = 5

    plt.figure(figsize=(14,5))
    plt.subplot(1,2,1)

    plt.scatter(x_train,y_train,color="steelblue") #observerd
    plt.plot(x_pred,preds,color="black",linewidth=2)
    plt.plot(x_pred,preds+2*sigma,color="red",linestyle="--",linewidth=2)
    plt.plot(x_pred,preds-2*sigma,color="red",linestyle="--",linewidth=2)
    plt.xlabel("x",size=16)
    plt.ylabel("y",size=16)
    plt.title("train data")
    plt.xlim([-1.5,6.5])
    plt.ylim([-30,55])

    plt.subplot(1,2,2)
    plt.scatter(x_val,y_val,color="steelblue") #observerd
    plt.plot(x_pred,preds,color="black",linewidth=2)
    plt.plot(x_pred,preds+2*sigma,color="red",linestyle="--",linewidth=2)
    plt.plot(x_pred,preds-2*sigma,color="red",linestyle="--",linewidth=2)
    plt.xlabel("x",size=16)
    plt.ylabel("y",size=16)
    plt.title("validation data")
    plt.xlim([-1.5,6.5])
    plt.ylim([-30,55])
    plt.show()


x_pred, preds, sigma = show_prediction(model_sd_1, x_train,y_train, x_val, y_val)

####################################################################################
import tensorboard

writer = tf.summary.create_file_writer("logs/continuous")


def NLL(y, distr):
    return -distr.log_prob(y)

def my_dist(params):
    mu = params[:,0:1]
    sigma = 1e-3 + 0.5 * tf.math.softplus(params[:,1:2])
    return tfd.Normal(loc=mu, scale=sigma)

inputs = Input(shape=(1,))
out1 = Dense(1)(inputs)
hidden1 = Dense(30, activation='relu')(inputs)
hidden1 = Dense(20, activation='relu')(hidden1)
hidden2 = Dense(20, activation='relu')(hidden1)
out2 = Dense(1)(hidden2)
params = tf.keras.layers.Concatenate()([out1, out2])
dist = tfp.layers.DistributionLambda(my_dist)(params)

model_sd_2 = tf.keras.Model(inputs=inputs, outputs=dist)
model_sd_2.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss=NLL)
model_sd_2.summary()

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs/continuous", histogram_freq=1)
history = model_sd_2.fit(x_train, y_train, epochs=300, validation_data=(x_val, y_val), verbose=1, callbacks=[tensorboard_callback]  )

show_train_history(history)

model_sd_2.evaluate(x_train, y_train), model_sd_2.evaluate(x_val, y_val), model_sd_2.evaluate(x_test, y_test),

x_pred = np.arange(-1,6,0.1)
preds = model_sd_2.predict(x_pred)

