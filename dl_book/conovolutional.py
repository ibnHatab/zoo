
from datetime import datetime
import tensorflow as tf
# tf.__version__

import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras
import tensorflow.keras as keras

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, MaxPooling2D, Conv2D, Dropout, Activation
from tensorflow.keras.utils import to_categorical

import tensorflow_probability as tfp

def generate_image_with_bars(size, bar_nr, vertical=True):
    img = np.zeros((size, size, 1), dtype=np.uint8)
    for i in range(0, bar_nr):
        x, y = np.random.randint(0, size, 2)
        l = int(np.random.randint(y, size, 1))

        if vertical:
            img[y:l, x, 0] = 255
        else:
            img[x, y:l, 0] = 255
    return img

# size, bar_nr = 50, 50
# img = generate_image_with_bars(size, bar_nr, vertical=False)
# plt.imshow(img.reshape(size, size), cmap='gray'); plt.show()

pixel_size = 50
num_images_trein = 1000
num_images_val = 1000

bernuoly = tfp.distributions.Bernoulli(probs=0.5)
Y_train = bernuoly.sample(num_images_trein)
Y_val = bernuoly.sample(num_images_val)
X_train = np.array([generate_image_with_bars(pixel_size, 10, vertical=Y_train[[i]]) for i in range(num_images_trein)])
X_val = np.array([generate_image_with_bars(pixel_size, 10, vertical=Y_val[i]) for i in range(num_images_val)])

X_train = X_train / 255
X_val = X_val / 255
# plt.imshow(X_val[0].reshape(pixel_size, pixel_size), cmap='gray'); plt.show()

model = Sequential()
model.add(Conv2D(1, (5, 5), padding='same', input_shape=(pixel_size, pixel_size, 1)))
model.add(Activation('linear'))
model.add(MaxPooling2D(pool_size=(pixel_size, pixel_size)))
model.add(Flatten())
model.add(Dense(2))
model.add(Activation('softmax'))
model.summary()

import tensorboard

log_dir = "logs/"  + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, to_categorical(Y_train), batch_size=64, epochs=50,
                    validation_data=(X_val, to_categorical(Y_val)), verbose=1, shuffle=True,
                    callbacks=[tensorboard_callback])

plt.figure(figsize=(12,4))
plt.subplot(1,2,(1))
plt.plot(history.history['accuracy'],linestyle='-.')
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='lower right')
plt.subplot(1,2,(2))
plt.plot(history.history['loss'],linestyle='-.')
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper right')
plt.show()

conv_filter = model.layers[0].get_weights()[0]
conv_filter.shape
conv_filter = conv_filter.reshape(5, 5)
plt.imshow(conv_filter, cmap='gray'); plt.show()