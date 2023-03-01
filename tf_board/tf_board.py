
import argparse
import os
import sys
import time
import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from tqdm import tqdm

try: sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
except: pass
# %cd tf_board
sys.path.append('../src')


from tensorboardX import SummaryWriter
from model import Net

from utils import RunningMean, accuracy, load_checkpoint, matplot_imshow, save_checkpoint

torch.backends.cudnn.deterministic = True
# seeds
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)


def parse_args(descrtiption = 'PyTorch Training'):
    parser = argparse.ArgumentParser(description=descrtiption)
    parser.add_argument('--data', default='./data', type=str, help='path to dataset')
    parser.add_argument('-lr', '--learning-rate', default=5e-5, type=float, help='learning rate')
    parser.add_argument('--resume', action='store_true', help='path to latest checkpoint (default: none)')
    parser.add_argument('--path_to_checkpoint', default='', type=str, help='path to checkpoint')
    parser.add_argument('--epochs', default=10, type=int, help='number of total epochs to run')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--num_workers', default=4, type=int, help='number of workers')
    parser.add_argument('--no-autoindent', action='store_true', help='ipython dummy')
    return parser.parse_args()

if __name__ == '__main__':
    opt = parse_args()

    transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5, ))
    ])

    trainset = torchvision.datasets.FashionMNIST('./data', download=True, train=True, transform=transformer)
    testset = torchvision.datasets.FashionMNIST('./data', download=True, train=False, transform=transformer)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

    classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')

    # img = testloader.dataset[0][0]
    # matplot_imshow(img, one_channel=True)

    net = Net()
    writer = SummaryWriter()

    dataiter = iter(trainloader)
    images, labels = next(dataiter)
    writer.add_graph(net, images)

    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        net = net.cuda()
        loss = loss.cuda()

    start_n_iter = 0
    start_eposch = 0
    if opt.resume:
        ckpt = load_checkpoint(opt.path_to_checkpoint)
        net.load_state_dict(ckpt['net'])
        start_eposch = ckpt['epoch']
        start_n_iter = ckpt['n_iter']
        optimizer.load_state_dict(ckpt['optim'])


    running_loss, running_train_accuracy = RunningMean(), RunningMean()
    n_iter = start_n_iter
    for epoch in range(start_eposch, opt.epochs):
        n_iter += 1 # for tensorboard

        pbar = tqdm(enumerate(trainloader), total=len(trainloader))
        start_time = time.time()

        for i, data in pbar:
            img, label = data
            if use_cuda:
                img = img.cuda()
                label = label.cuda()

            preparation_time = time.time() - start_time

            # foreward and backward pass
            net.train()
            optimizer.zero_grad()
            out = net(img)
            l = loss(out, label)

            l.sum().backward()
            optimizer.step()

            # calculate l, accuracy
            running_loss.update(l.sum().item()/label.shape[0])
            running_train_accuracy.update(accuracy(out, label)/label.numel())

            # update tfboard
            if i % 100 == 99:
                pass
            idx = epoch * len(trainloader) + i
            writer.add_scalar('train_loss', running_loss(), idx)
            writer.add_scalar('train_accuracy', running_train_accuracy(), idx)

            process_time = time.time() - start_time - preparation_time
            compute_efficiency = (process_time / (process_time + preparation_time)) * 100
            pbar.set_description('train epoch: {}/{}, l: {:.4f}, acc: {:.2f}, compute: {:.2f}%'.
                                 format(epoch, opt.epochs,  running_loss(), running_train_accuracy(), compute_efficiency))
            start_time = time.time()

        if epoch % 1 == 0:
            net.eval()
            running_test_accuracy = RunningMean()
            pbar = tqdm(enumerate(testloader), total=len(testloader))
            with torch.no_grad():
                for i, data in pbar:
                    img, label = data

                    if use_cuda:
                        img = img.cuda()
                        label = label.cuda()

                    out = net(img)
                    running_test_accuracy.update(accuracy(out, label)/label.numel())

                    pbar.set_description('test epoch:  {}/{}, test acc: {:.2f}'.
                                         format(epoch, opt.epochs, running_test_accuracy()))

                    if i % 100 == 99:
                        pass
                        idx = epoch * len(trainloader) + i
                        writer.add_scalar('test_accuracy', running_test_accuracy(), idx)


            save_checkpoint({
                'epoch': epoch,
                'net': net.state_dict(),
                'optim': optimizer.state_dict(),
                'n_iter': n_iter,
            }, 'ckpt_{}.pth'.format(epoch))






