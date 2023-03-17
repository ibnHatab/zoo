
# %cd keras
import sys, os
sys.path.append(os.path.abspath('../src'))

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tqdm import tqdm
from matplotlib import pyplot as plt

plt.style.use('ggplot')

POSITIVE_CLASS = 1
BAG_COUNT = 1000
VAL_BAG_COUNT = 300
BAG_SIZE = 3
PLOT_SIZE = 3
ENSEMBLE_AVG_COUNT = 1

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# input_data, input_labes, positive_class, bag_count, instance_count = x_train, y_train, POSITIVE_CLASS, BAG_COUNT, BAG_SIZE

def create_bag(input_data, input_labes, positive_class, bag_count, instance_count):
    bags = []
    bag_labels = []
    input_data = np.divide(input_data, 255.)
    count = 0
    for _ in range(bag_count):
        index = np.random.choice(len(input_data), instance_count, replace=False)
        instance_data = input_data[index]
        instance_labels = input_labes[index]
        bag_label = 0
        if positive_class in instance_labels:
            bag_label = 1
            count += 1
        bags.append(instance_data)
        bag_labels.append(bag_label)
    return (list(np.swapaxes(bags, 0, 1)), np.array(bag_labels))

train_data, train_labels = create_bag(x_train, y_train, POSITIVE_CLASS, BAG_COUNT, BAG_SIZE)
val_data, val_labels = create_bag(x_test, y_test, POSITIVE_CLASS, VAL_BAG_COUNT, BAG_SIZE)

def plot(data, labels, bag_class, predictions=None, attention_weights=None):

    """"Utility for plotting bags and attention weights.

    Args:
      data: Input data that contains the bags of instances.
      labels: The associated bag labels of the input data.
      bag_class: String name of the desired bag class.
        The options are: "positive" or "negative".
      predictions: Class labels model predictions.
      If you don't specify anything, ground truth labels will be used.
      attention_weights: Attention weights for each instance within the input data.
      If you don't specify anything, the values won't be displayed.
    """

    labels = np.array(labels).reshape(-1)

    if bag_class == "positive":
        if predictions is not None:
            labels = np.where(predictions.argmax(1) == 1)[0]
            bags = np.array(data)[:, labels[0:PLOT_SIZE]]

        else:
            labels = np.where(labels == 1)[0]
            bags = np.array(data)[:, labels[0:PLOT_SIZE]]

    elif bag_class == "negative":
        if predictions is not None:
            labels = np.where(predictions.argmax(1) == 0)[0]
            bags = np.array(data)[:, labels[0:PLOT_SIZE]]
        else:
            labels = np.where(labels == 0)[0]
            bags = np.array(data)[:, labels[0:PLOT_SIZE]]

    else:
        print(f"There is no class {bag_class}")
        return

    print(f"The bag class label is {bag_class}")
    for i in range(PLOT_SIZE):
        figure = plt.figure(figsize=(8, 8))
        print(f"Bag number: {labels[i]}")
        for j in range(BAG_SIZE):
            image = bags[j][i]
            figure.add_subplot(1, BAG_SIZE, j + 1)
            plt.grid(False)
            if attention_weights is not None:
                plt.title(np.around(attention_weights[labels[i]][j], 2))
            plt.imshow(image)
        plt.show()

plot(val_data, val_labels, "positive")
plot(val_data, val_labels, "negative")


class MILAttentionLayer(layers.Layer):
    def __init__(self, weight_param_dim, kernel_initializer='glorot_uniform',
                         kernel_regularizer=None, use_gate=False, **kwargs):
        super().__init__(**kwargs)
        self.weight_param_dim = weight_param_dim
        self.use_gate = use_gate
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        # self = MILAttentionLayer(1)
        self.v_init = self.kernel_initializer
        self.w_init = self.kernel_initializer
        self.u_init = self.kernel_initializer
        self.v_regularizer = self.kernel_regularizer
        self.w_regularizer = self.kernel_regularizer
        self.u_regularizer = self.kernel_regularizer

    def build(self, input_shape=([1,1],1)):
        inpuit_dim = input_shape[0][1]
        self.v_weight_param = self.add_weight(
            shape=(inpuit_dim, self.weight_param_dim),
            initializer=self.v_init,
            name='v_weight_param',
            regularizer=self.v_regularizer,
            trainable=True)
        self.w_weight_param = self.add_weight(
            shape=(inpuit_dim, self.weight_param_dim),
            initializer=self.w_init,
            name='w_weight_param',
            regularizer=self.w_regularizer,
            trainable=True)

        if self.use_gate:
            self.u_weight_param = self.add_weight(
                shape=(inpuit_dim, self.weight_param_dim),
                initializer=self.u_init,
                name='u_weight_param',
                regularizer=self.u_regularizer,
                trainable=True)
        else:
            self.u_weight_param = None

        self.input_built = True

    def call(self, inputs):
        instances = [self.compute_attention_score(instance) for instance in inputs]
        alpha = tf.math.softmax(instances, axis=0)
        return [alpha[i] for i in range(len(alpha))]

    def compute_attention_score(self, instance):
        original_instance = instance
        instance = tf.math.tanh(tf.tensordot(instance, self.v_weight_param, axes=1))
        if self.use_gate:
            gate = tf.math.sigmoid(tf.tensordot(original_instance, self.u_weight_param, axes=1))
            instance = gate * instance
        return tf.tensordot(instance, self.w_weight_param, axes=1)

def create_model(instance_shape):
    inputs, embeddngs = [], []
    shared_dense_layeer_1 = layers.Dense(128, activation='relu')
    shared_dense_layeer_2 = layers.Dense(64, activation='relu')

    for _ in range(BAG_SIZE):
        input = layers.Input(shape=instance_shape)
        flatten = layers.Flatten()(input)
        x = shared_dense_layeer_1(flatten)
        x = shared_dense_layeer_2(x)
        inputs.append(input)
        embeddngs.append(x)

    alpha = MILAttentionLayer(
        weight_param_dim=256,
        kernel_regularizer=keras.regularizers.l2(1e-2),
        use_gate=True,
        name='alpha'
        )(embeddngs)
    multiply_layers = [layers.Multiply()([alpha[i], embeddngs[i]]) for i in range(len(alpha))]
    concat = layers.concatenate(multiply_layers, axis=1)
    output = layers.Dense(2, activation='softmax')(concat)
    return keras.Model(inputs=inputs, outputs=output)

def copute_class_weights(labels):
    labels = np.array(labels).reshape(-1)
    positive_count = len(np.where(labels == 1)[0])
    negative_count = len(np.where(labels == 0)[0])
    total_count = positive_count + negative_count
    return {
        0: (1 / negative_count) * (total_count) / 2.0,
        1: (1 / positive_count) * (total_count) / 2.0
    }


def train(train_data, train_labels, test_data, test_labels, model):
    file_path = "./mil_model.h5"
    model_checkpoint = keras.callbacks.ModelCheckpoint(
        file_path,
        monitor='val_loss',
        verbose=1,
        mode='min',
        save_best_only=True,
        save_weights_only=True)
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, verbose=1, mode='min')
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy'])


instance_shape = train_data[0][0].shape
modela = [create_model(instance_shape)  for _ in range(ENSEMBLE_AVG_COUNT)]