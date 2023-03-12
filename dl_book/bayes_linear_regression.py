import tensorflow as tf
import tensorflow_probability as tfp

import numpy as np
import matplotlib.pyplot as plt

from urllib.request import urlretrieve, urlopen
from PIL import Image
import seaborn as sns

# Creation of the artificial data
sigma = 3.
xmin = -5.; xmax = 5.; ymin = -40.; ymax = 40.
nbins_c = 40
save = True
num = 4
x = np.linspace(-2, 2, num).reshape(-1, 1)
y = 2 * x[:,0] -1 + 1.*np.random.normal(sigma, num)

# The MaxLike solution
# The CPD 𝑝(𝑦|𝑥,𝑎,𝑏)
# for abitrary values of 𝑎,𝑏
# using a binning approch
# The Likelihood 𝑝(𝑎,𝑏)
# for values 𝑎,𝑏
# using a binning approch
# The resulting CPD 𝑝(𝑦|𝑥)
# combining the CPD 𝑝(𝑦|𝑥,𝑎,𝑏) and the likelihood 𝑝(𝑎,𝑏)
# An analytical solution