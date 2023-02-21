

import argparse
import os
import time
from matplotlib import pyplot as plt
import numpy as np
import torch
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import accuracy, load_checkpoint, save_checkpoint, ensure_dir

from model import ResNet18

from tensorboardX import SummaryWriter
# seeds

torch.backends.cudnn.deterministic = True
# np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)

def parse_args(descrtiption = 'PyTorch Training'):
    parser = argparse.ArgumentParser(description=descrtiption)
    parser.add_argument('--data', default='./data', type=str, help='path to dataset')
    parser.add_argument('-lr', '--learning-rate', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--resume', action='store_true', help='path to latest checkpoint (default: none)')
    parser.add_argument('--path_to_checkpoint', default='', type=str, help='path to checkpoint')
    parser.add_argument('--epochs', default=10, type=int, help='number of total epochs to run')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--num_workers', default=4, type=int, help='number of workers')

    parser.add_argument('--no-autoindent', action='store_true', help='ipython dummy')
    return parser.parse_args()

if __name__ == '__main__':
    opt = parse_args()

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = datasets.CIFAR10(root=opt.data, train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(root=opt.data, train=False, download=True, transform=test_transform)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    classes = np.array(classes)

    train_dataset_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    test_dataset_loader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)

    net = ResNet18(10, 3)

    critrrion_CE = torch.nn.CrossEntropyLoss()
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        net = net.cuda()
        critrrion_CE = critrrion_CE.cuda()

    optim = torch.optim.Adam(net.parameters(), lr=opt.learning_rate)

    start_n_iter = 0
    start_eposch = 0
    if opt.resume:
        ckpt = load_checkpoint(opt.path_to_checkpoint)
        net.load_state_dict(ckpt['net'])
        start_eposch = ckpt['epoch']
        start_n_iter = ckpt['n_iter']
        optim.load_state_dict(ckpt['optim'])

    writer = SummaryWriter()
    X = torch.rand(1, 3, 32, 32)
    if use_cuda:
        X = X.cuda()
    net(X)
    writer.add_graph(net, X)
    running_loss = 0.
    running_train_accuracy = 0.

    n_iter = start_n_iter
    for epoch in range(start_eposch, opt.epochs):

        net.train()

        pbar = tqdm(enumerate(train_dataset_loader), total=len(train_dataset_loader))
        start_time = time.time()

        for i, data in pbar:
            img, label = data

            # plt.imshow(img[3].cpu().permute(1, 2, 0)); plt.show()

            if use_cuda:
                img = img.cuda()
                label = label.cuda()

            preparation_time = time.time() - start_time

            # foreward and backward pass
            out = net(img)
            loss = critrrion_CE(out, label)
            optim.zero_grad()
            loss.backward()
            optim.step()

            # calculate loss, accuracy
            train_accuracy = accuracy(out, label)
            running_loss += loss.item()
            running_train_accuracy += train_accuracy

            # update tfboard
            if i % 100 == 99:
                running_loss /= 100.
                running_train_accuracy /= 100.

                idx = epoch * len(train_dataset_loader) + i
                writer.add_scalar('train_loss', running_loss, idx)
                writer.add_scalar('train_accuracy', running_train_accuracy, idx)
                running_loss = 0.
                running_train_accuracy = 0.

            process_time = time.time() - start_time - preparation_time
            compute_efficiency = (process_time / (process_time + preparation_time)) * 100
            pbar.set_description('epoch: {}/{}, loss: {:.4f}, acc: {:.2f}%, compute: {:.2f}%'.
                                 format(epoch, opt.epochs,  loss.item(), train_accuracy, compute_efficiency))
            start_time = time.time()

        if epoch % 1 == 0:
            net.eval()
            test_accuracy = 0.
            pbar = tqdm(enumerate(test_dataset_loader), total=len(test_dataset_loader))
            with torch.no_grad():
                for i, data in pbar:
                    img, label = data

                    if use_cuda:
                        img = img.cuda()
                        label = label.cuda()

                    out = net(img)

                    test_accuracy += accuracy(out, label)

                    pbar.set_description('test acc: {:.2f}%'.format(test_accuracy))

                    if i % 100 == 99:
                        test_accuracy /= 100.
                        idx = epoch * len(train_dataset_loader) + i
                        writer.add_scalar('test_accuracy', test_accuracy, idx)
                        test_accuracy = 0.

                image_grid = torchvision.utils.make_grid(img, nrow=2, normalize=True, scale_each=True)
                writer.add_image('test_images', image_grid)

            save_checkpoint({
                'epoch': epoch,
                'net': net.state_dict(),
                'optim': optim.state_dict(),
                'n_iter': n_iter,
            }, 'ckpt_{}.pth'.format(epoch))

if False:
    # Helper function
    def select_n_random(data, labels, n=100):
        '''
        Selects n random datapoints and their corresponding labels from a dataset
        '''
        assert len(data) == len(labels)

        perm = torch.randperm(len(data))
        return torch.tensor(data)[perm][:n], torch.tensor(labels)[perm][:n]

    data, labels = train_dataset.data, train_dataset.targets
    # select random images and their target indices
    images, labels = select_n_random(train_dataset.data, train_dataset.targets)

    # get the class labels for each image
    class_labels = [classes[lab] for lab in labels]

    # log embeddings
    # extracting the features from my model (output of the fully connected layer)
    features = images.view(-1, 32 * 32)
    writer.add_embedding(features,
                        metadata=class_labels,
                        label_img=images.unsqueeze(1))

