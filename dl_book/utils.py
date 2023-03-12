
import os
from urllib.request import urlretrieve
from matplotlib import pyplot as plt


def show_train_history(history):
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='val')
    plt.legend(['train', 'val'])
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()

def get_if_not_there(base_url, filename):
    if not os.path.exists(filename):
        print('Downloading dear data...')
        url = '{}/{}'.format(base_url, filename)
        urlretrieve(url, filename)
    return os.path.abspath(filename)
