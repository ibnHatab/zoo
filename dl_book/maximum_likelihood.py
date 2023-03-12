
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import binom

ndollars = np.array(np.linspace(0, 10, 11), dtype=np.int32)
pdolar_sign = binom.pmf(k=ndollars, n=10, p=1/6)
plt.stem(ndollars, pdolar_sign, label='r');
pdolar_sign = binom.pmf(k=ndollars, n=10, p=2/6)
plt.stem(ndollars, pdolar_sign, label='g');
plt.show()

ndollars = np.linspace(0, 6, 7)
pdollar = binom.pmf(k=4, n=10, p=ndollars/6)
plt.stem(ndollars, pdollar, label='r'); plt.show()

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, MaxPooling2D, Conv2D, Dropout, Activation

def create_sin_data(n_samples=1000, noise=0.1, n_periods=1, n_points=100):
    x = np.linspace(0, n_periods * 2 * np.pi, n_points)
    y = np.sin(x)
    x = x + np.random.normal(0, noise, x.shape)
    y = y + np.random.normal(0, noise, y.shape)
    return x, y

x, y = create_sin_data(noise=0.4, n_periods=2, n_points=1000)


model = Sequential()
model.add(Dense(1, activation='linear', input_shape=(1,)))
model.compile(optimizer='adam', loss='mse')
model.summary()

history = model.fit(x, y, batch_size=16, epochs=1000, verbose=0)

model.evaluate(x, y, verbose=1)


model = Sequential()
model.add(Dense(20, activation='relu', input_shape=(1,)))
model.add(Dense(50, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(optimizer='adam', loss='mse')
model.summary()

history = model.fit(x, y, batch_size=16, epochs=1000, verbose=1)

y_hat = model.predict(x)
plt.scatter(x, y_hat[:,0], c='r'); plt.scatter(x, y, c='b'); plt.show()

