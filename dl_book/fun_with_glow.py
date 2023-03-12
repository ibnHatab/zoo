import os
from urllib.request import urlretrieve

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from utils import get_if_not_there

tfd = tfp.distributions
tfb = tfp.bijectors

from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam


# If blocksparse doesn't install, use unoptimized model (and set optimized=False in model.py)
get_if_not_there('https://storage.googleapis.com/glow-demo/large3', 'graph_unoptimized.pb')
# Get manipulation vectors
# !curl https://storage.googleapis.com/glow-demo/z_manipulate.npy > z_manipulate.npy
# tf.io.gfile.mkdir('test')
# !curl https://raw.githubusercontent.com/openai/glow/master/demo/test/img.png  > test/img.png