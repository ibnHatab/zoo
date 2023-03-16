import os
import sys

from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# %cd keras

sys.path.append(os.path.abspath('../src'))

from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import decode_predictions

model = EfficientNetB0(include_top=True, weights='imagenet', input_shape=(224, 224, 3))
model.summary()

keras.utils.plot_model(model, show_shapes=True)

image_size = (224, 224)
data_path = '../data/PetImages'
img = keras.preprocessing.image.load_img(data_path+'/Dog/1.jpg', target_size=image_size)
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
argmax = tf.argmax(predictions[0])
decode_predictions(predictions, top=1)[0][0]
plt.imshow(img_array[0] / 255.0); plt.show()


import tensorflow_datasets as tfds

bstch_size = 32

dataset_name = 'stanford_dogs'
dataset_name = "stanford_dogs"
(ds_train, ds_test), ds_info = tfds.load(
    dataset_name, split=["train", "test"], with_info=True, as_supervised=True
)
NUM_CLASSES = ds_info.features['label'].num_classes

ds_train = ds_train.map(lambda image, label: (tf.image.resize(image, image_size), label))
ds_test = ds_test.map(lambda image, label: (tf.image.resize(image, image_size), label))

labels_to_names = ds_info.features['label'].int2str
images, labels = zip(*ds_train.take(9))
labels = [labels_to_names(label).split('-')[1] for label in labels]

from utils import show_images
# show_images(images, 3, 3, titles=labels)

from tensorflow.keras.models import Sequential
from tensorflow.keras import layers

img_augmentation = Sequential([
    layers.RandomRotation(0.2),
    layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
    layers.RandomFlip(),
    layers.RandomContrast(factor=0.1),])

# for image, label in ds_train.take(1):
#     for i in range(9):
#         ax = plt.subplot(3, 3, i + 1)
#         aug_img = img_augmentation(tf.expand_dims(image, axis=0))
#         plt.imshow(aug_img[0].numpy().astype("uint8"))
#         plt.title(labels_to_names(label))
#         plt.axis("off")
# plt.show()

def input_preprocess(image, label):
    label = tf.one_hot(label, NUM_CLASSES)
    return image, label

ds_train = ds_train.map(input_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_train = ds_train.batch(bstch_size, drop_remainder=True)
ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

ds_test = ds_test.map(input_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_test = ds_test.batch(bstch_size, drop_remainder=True)

inputs = layers.Input(shape=(224, 224, 3))
x = img_augmentation(inputs)
outputs = EfficientNetB0(include_top=True, weights=None, classes=NUM_CLASSES)(x)

model = keras.Model(inputs, outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

epochs = 10
hist = model.fit(ds_train, epochs=epochs, validation_data=ds_test, verbose=2)

def plot_hist(hist):
    plt.plot(hist.history["accuracy"])
    plt.plot(hist.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()

plot_hist(hist)

## transfer learning

def build_model(num_classes):
    inputs = layers.Input(shape=(224, 224, 3))
    x = img_augmentation(inputs)
    model = EfficientNetB0(include_top=False, weights='imagenet', input_tensor=x)

    model.trainable = False

    x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = layers.BatchNormalization()(x)

    tot_dropout = 0.2
    x = layers.Dropout(tot_dropout, name='top_dropout')(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="pred")(x)

    model = keras.Model(inputs, outputs, name="EfficientNet")
    optimizer = keras.optimizers.Adam(learning_rate=1e-2)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

model_pre = build_model(NUM_CLASSES)
epochs = 10
hist = model_pre.fit(ds_train, epochs=epochs, validation_data=ds_test, verbose=2)
plot_hist(hist)

def unfreeze_model(model):
    for layer in model.layers[-20:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True

    optimizer = keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

unfreeze_model(model_pre)
epochs = 3
hist = model_pre.fit(ds_train, epochs=epochs, validation_data=ds_test, verbose=2)
plot_hist(hist)



