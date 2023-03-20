# %cd keras

import sys, os
from matplotlib import pyplot as plt
sys.path.append(os.path.abspath('../src'))

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds
import tensorflow_probability as tfp

tfd = tfp.distributions
tfb = tfp.bijectors

train_size = 600
hidden_units = [8, 8]
learning_rate = 0.001
dataset_size = 4898
batch_size = 256

def get_train_and_test_splits(train_size, batch_size=1):
    # We prefetch with a buffer the same size as the dataset because th dataset
    # is very small and fits into memory.
    dataset = (
        tfds.load(name="wine_quality", as_supervised=True, split="train")
        .map(lambda x, y: (x, tf.cast(y, tf.float32)))
        .prefetch(buffer_size=dataset_size)
        .cache()
    )
    # We shuffle with a buffer the same size as the dataset.
    train_dataset = (
        dataset.take(train_size).shuffle(buffer_size=train_size).batch(batch_size)
    )
    test_dataset = dataset.skip(train_size).batch(batch_size)

    return train_dataset, test_dataset

def run_experiment(model, loss, train_dataset, test_dataset):

    model.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=learning_rate),
        loss=loss,
        metrics=[keras.metrics.RootMeanSquaredError()],
    )

    print("Start training the model...")
    model.fit(train_dataset, epochs=num_epochs, validation_data=test_dataset)
    print("Model training finished.")
    _, rmse = model.evaluate(train_dataset, verbose=0)
    print(f"Train RMSE: {round(rmse, 3)}")

    print("Evaluating model performance...")
    _, rmse = model.evaluate(test_dataset, verbose=0)
    print(f"Test RMSE: {round(rmse, 3)}")


FEATURE_NAMES = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]

def create_model_inputs():
    inputs = {}
    for feature_name in FEATURE_NAMES:
        inputs[feature_name] = layers.Input(
            name=feature_name, shape=(1,), dtype=tf.float32
        )
    return inputs

def create_baseline_model():
    inputs = create_model_inputs()
    input_values = [value for _, value in sorted(inputs.items())]
    features = keras.layers.concatenate(input_values)
    features = layers.BatchNormalization()(features)

    # Create hidden layers with deterministic weights using the Dense layer.
    for units in hidden_units:
        features = layers.Dense(units, activation="sigmoid")(features)
    # The output is deterministic: a single point estimate.
    outputs = layers.Dense(units=1)(features)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

train_size = int(dataset_size * 0.7)
train_dataset, test_dataset = get_train_and_test_splits(train_size, batch_size)

num_epochs = 100
mse_loss_fn = keras.losses.MeanSquaredError()
baseline_model = create_baseline_model()
baseline_model.summary()

keras.utils.plot_model(baseline_model, show_shapes=True)
run_experiment(baseline_model, mse_loss_fn, train_dataset, test_dataset)

samples = 10
examples, targets = list(test_dataset.unbatch().shuffle(batch_size*10).batch(samples))[0]

predicted = baseline_model(samples).numpy()
for idx in range(samples):
    print(f"Predicted: {predicted[idx][0]:.2f}, target: {targets[idx].numpy():.2f}")

# xy = sorted(zip(targets.numpy(), predicted[:,0]))
# plt.plot(xy); plt.show()

def prior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    prior_model = keras.Sequential(
        [
            tfp.layers.DistributionLambda(
                lambda t: tfd.Independent(
                    tfd.Normal(loc=t, scale=1), reinterpreted_batch_ndims=1
                )
            )
        ]
    )
    return keras.Sequential(
        [
            layers.Dense(n, kernel_initializer="zeros", bias_initializer="zeros"),
            prior_model,
        ]
    )

def posterior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    posterior_model = keras.Sequential(
        [
            tfp.layers.VariableLayer(
                tfp.layers.MultivariateNormalTriL.params_size(n), dtype=dtype
            ),
            tfp.layers.MultivariateNormalTriL(n),
        ]
    )
    return posterior_model

def create_bnn(train_size):
    inputs = create_model_inputs()
    feateures = keras.layers.concatenate(list(inputs.values()))
    feateures = layers.BatchNormalization()(feateures)

    for units in hidden_units:
        feateures = tfp.layers.DenseVariational(
            units=units,
            make_prior_fn=prior,
            make_posterior_fn=posterior,
            kl_weight=1 / train_size,
            activation="sigmoid",
        )(feateures)

    outputs = layers.Dense(units=1)(feateures)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

num_epochs = 500
train_sample_size = int(train_size * 0.3)
small_train_dataset = train_dataset.unbatch().take(train_sample_size).batch(batch_size)
bnn_model_small = create_bnn(train_sample_size)
run_experiment(bnn_model_small, mse_loss_fn, small_train_dataset, test_dataset)

def compute_predictions(model, iterations=100):
    predicted = []
    for _ in range(iterations):
        predicted.append(model(examples).numpy())
    predicted = np.concatenate(predicted, axis=1)

    prediction_mean = np.mean(predicted, axis=1).tolist()
    prediction_min = np.min(predicted, axis=1).tolist()
    prediction_max = np.max(predicted, axis=1).tolist()
    prediction_range = (np.max(predicted, axis=1) - np.min(predicted, axis=1)).tolist()

    for idx in range(samples):
        print(
            f"Predictions mean: {round(prediction_mean[idx], 2)}, "
            f"min: {round(prediction_min[idx], 2)}, "
            f"max: {round(prediction_max[idx], 2)}, "
            f"range: {round(prediction_range[idx], 2)} - "
            f"Actual: {targets[idx]}"
        )


# compute_predictions(baseline_model)
compute_predictions(bnn_model_small)

