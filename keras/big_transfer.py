
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
import tensorflow_datasets as tfds

# %cd keras
import sys, os
sys.path.append(os.path.abspath('../src'))

from utils import show_images

SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)

train_ds, validation_ds, test_ds = tfds.load(
    "tf_flowers",
    split=["train[:80%]", "train[80%:90%]", "train[90%:]"],
    as_supervised=True,
)

images, labels = zip(*train_ds.take(9))
labels = [str(label.numpy()) for label in labels]
images[0].shape, labels
# show_images(images, 3, 3, titles=labels)

RESIZE_TO = 384
CROP_TO = 224
BATCH_SIZE = 64
STEPS_PER_EPOCH = 10
AUTO = tf.data.AUTOTUNE
SCHEDULE_LENGTH = (500)
SCHEDULE_BOUNDARIES = [200, 300, 400]
NUM_CLASSES = 5

SCHEDULE_LENGTH = SCHEDULE_LENGTH / 512 * BATCH_SIZE

@tf.function
def preprocess_train(image, label):
    image = tf.image.resize(image, (RESIZE_TO, RESIZE_TO))
    image = tf.image.random_crop(image, (CROP_TO, CROP_TO, 3))
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_crop(image, (CROP_TO, CROP_TO, 3))
    image = image / 255.0
    return image, label

@tf.function
def preprocess_test(image, label):
    image = tf.image.resize(image, (CROP_TO, CROP_TO))
    image = image / 255.0
    return image, label

DATASET_NUM_TRAIN_EXAMPLES = train_ds.cardinality().numpy()

repeate_count = int(SCHEDULE_LENGTH * BATCH_SIZE / DATASET_NUM_TRAIN_EXAMPLES * STEPS_PER_EPOCH)
repeate_count += 1 + 500 // BATCH_SIZE

pipeline_train = (
    train_ds.shuffle(10000)
    .repeat(repeate_count)
    .map(preprocess_train, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

pipeline_validation = (
    validation_ds.map(preprocess_test, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

image_batch, label_batch = next(iter(pipeline_train))
image_batch.shape, label_batch.shape
label_batch = [str(label.numpy()) for label in label_batch]

# plt.figure(figsize=(10, 10))
# for n in range(25):
#     ax = plt.subplot(5, 5, n + 1)
#     plt.imshow(image_batch[n])
#     plt.title(label_batch[n].numpy())
#     plt.axis("off")
# plt.show()

bit_model_url = "https://tfhub.dev/google/bit/m-r50x1/1"
bit_model = hub.KerasLayer(bit_model_url, trainable=False)

class BitModel(keras.Model):
    def __init__(self, bit_model, num_classes):
        super().__init__()
        self.bit_model = bit_model
        self.classifier = keras.layers.Dense(num_classes, kernel_initializer="zeros")

    def call(self, inputs):
        bit_embedding = self.bit_model(inputs)
        return self.classifier(bit_embedding)

model = BitModel(bit_model, num_classes=NUM_CLASSES)

learning_rate = 0.003 * BATCH_SIZE / 512
lr_schedule = keras.optimizers.schedules.PiecewiseConstantDecay(
    boundaries=SCHEDULE_BOUNDARIES, values=[learning_rate, 0.1 * learning_rate, 0.01 * learning_rate, 0.001 * learning_rate]
)
optimizer = keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9, nesterov=True)
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss_fn, metrics=["accuracy"])
train_callbacks = [
    keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True),
]

history = model.fit(
    pipeline_train,
    batch_size=BATCH_SIZE,
    epochs=int(SCHEDULE_LENGTH // STEPS_PER_EPOCH) + 1,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_data=pipeline_validation,
    callbacks=train_callbacks,
)

model.evaluate(pipeline_validation)[1] * 100

predictions = model.predict(image_batch)
predictions_labels = [predictions[i].argmax() for i in range(len(predictions))]
predictions_labels = [str(label) for label in predictions_labels]
','.join(label_batch)
','.join(predictions_labels)
np.where(np.array(label_batch) != np.array(predictions_labels))

