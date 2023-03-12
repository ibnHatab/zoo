import os
import time
import PIL
from matplotlib import pyplot as plt
import torch
import torchvision
from torchvision import transforms

# %cd src
from utils import DataModule, HyperParameters, Module, Trainer, add_to_class, show_images

torch.set_printoptions(2)

def box_corner_to_center(boxes):
    """Convert from (upper-left, lower-right) to (center, width, height)."""
    return torch.cat(((boxes[:, 2:] + boxes[:, :2]) / 2,  # x, y
                      boxes[:, 2:] - boxes[:, :2]), 1)  # w, h

def box_center_to_corner(boxes):
    """Convert from (center, width, height) to (upper-left, lower-right)."""
    return torch.cat((boxes[:, :2] - boxes[:, 2:] / 2,  # x1, y1
                      boxes[:, :2] + boxes[:, 2:] / 2), 1)  # x2, y2


bbox_to_rect = lambda bbox, color='red': plt.Rectangle(
    xy=(bbox[0], bbox[1]), width=bbox[2] - bbox[0], height=bbox[3] - bbox[1],
    fill=False, edgecolor=color, linewidth=2)

dog_bbox, cat_bbox = torch.tensor([[60, 45, 378, 516], [400, 112, 655, 493]])
boxes = torch.stack((dog_bbox, cat_bbox))
box_center_to_corner(box_corner_to_center(boxes)) == boxes

img = PIL.Image.open('../d2l/img/catdog.jpg')
fig = plt.imshow(img)
fig.axes.add_patch(bbox_to_rect(dog_bbox, 'blue'))
fig.axes.add_patch(bbox_to_rect(cat_bbox, 'red'))
plt.show(block=False)

