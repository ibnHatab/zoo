
import os
import pandas as pd
import torch

os.makedirs(os.path.join('data'), exist_ok=True)
data_file = os.path.join('data', 'house_tiny.csv')
# https://www.kaggle.com/datasets/mengting098/house-tiny/download and place in data

data = pd.read_csv(data_file)

data.head()

inputs, targets = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = pd.get_dummies(inputs, dummy_na=True)
inputs = inputs.fillna(inputs.mean())

X, y = torch.tensor(inputs.values), torch.tensor(targets.values)

inputs["NumRooms"].value_counts()
