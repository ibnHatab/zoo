
import argparse
import os
import sys
import time
from matplotlib import pyplot as plt
import numpy as np
import torch
import torchinfo
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

try: sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
except: pass
# %cd visualizing_feature_maps
sys.path.append('../src')

from utils import accuracy, load_checkpoint, save_checkpoint, ensure_dir, graph_forward, show_images


from tensorboardX import SummaryWriter
# seeds
torch.backends.cudnn.deterministic = True
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)


if __name__ == '__main__':
    in_image = 'dog.jpg'
    if len(sys.argv) > 1:
        in_image = sys.argv[1]

    use_cuda = torch.cuda.is_available()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=0., std=1.)
        ])

    image_origin = Image.open(in_image)
    # plt.imshow(image); plt.show()

    net = torchvision.models.resnet18(pretrained=True)

    image = transform(image_origin)
    image = image.unsqueeze(0)
    net = torchvision.models.resnet18(pretrained=True)

    if use_cuda:
        net = net.cuda()
        image = image.cuda()

    y = net(image)
    # graph_forward(net, image, depth=4)
    torchinfo.summary(net, input_size=(1, 3, 224, 224))

    model_weights = []
    conv_layers = []
    model_children = list(net.children())
    counter = 0
    for i in range(len(model_children)):
        if type(model_children[i]) == torch.nn.modules.conv.Conv2d:
            model_weights.append(model_children[i].weight)
            conv_layers.append(model_children[i])
            counter += 1
        elif type(model_children[i]) == torch.nn.modules.container.Sequential:
            for j in range(len(model_children[i])):
                for child in model_children[i][j].children():
                    if type(child) == torch.nn.modules.conv.Conv2d:
                        model_weights.append(child.weight)
                        conv_layers.append(child)
                        counter += 1


    print('Number of conv layers: ', counter)

    outputs = []
    names = []
    input = image
    for layer in conv_layers:
        input = layer(input)
        outputs.append(input)
        names.append(str(layer))

    processed = []
    feature_map = outputs[0]
    for feature_map in outputs:
        feature_map = feature_map.squeeze(0)
        grey_scale = torch.sum(feature_map, 0)
        grey_scale = grey_scale / feature_map.shape[0]
        processed.append(grey_scale.detach().cpu().numpy())

    processed = [image_origin] + processed
    short_names = [name.split(',')[0:4] for name in names]
    show_images(processed, 6, 3, titles=['in']+short_names)
    plt.show()
    # for mw in model_weights:
    #     print(mw.shape)

    # m = model_weights[0].mean(0).mean(0)
    # m = model_weights[7][10].mean(0)
    # plt.imshow(m.detach().cpu().numpy()); plt.show()