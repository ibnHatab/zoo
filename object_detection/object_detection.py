
import sys
from matplotlib import pyplot as plt
import torch
import torchvision
from torch.utils.data import DataLoader
from PIL import Image
import pycocotools.mask as mask_util


# %cd object_detection
sys.path.append('../src')
from utils import save_checkpoint

from data import get_transform, PennFundalDataset
from model import get_instance_segmentation_model

import vision_utils
from engine import train_one_epoch, evaluate

DATADIR = './data/PennFudanPed/'

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    num_classes = 2
    dataset = PennFundalDataset(DATADIR, get_transform(train=True))
    dataset_test = PennFundalDataset(DATADIR, get_transform(train=False))

    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    data_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=vision_utils.collate_fn)
    data_loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=4, collate_fn=vision_utils.collate_fn)

    model = get_instance_segmentation_model(num_classes)

    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    num_epochs = 10

    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        lr_scheduler.step()
        evaluate(model, data_loader_test, device=device)

        save_checkpoint({
                'epoch': epoch,
                'net': model.state_dict(),
                'optim': optimizer.state_dict(),
            }, 'ckpt_{}.pth'.format(epoch))



if __name__ == '__main__':
    main()