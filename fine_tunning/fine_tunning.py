import argparse
import os
import sys
import time
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm

try: sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
except: pass
# %cd fine_tunning
sys.path.append('../src')

from utils import RunningMean, accuracy, load_checkpoint, save_checkpoint, ensure_dir, download_extract, show_images


from tensorboardX import SummaryWriter

# seeds
torch.backends.cudnn.deterministic = True
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
    parser.add_argument('--param_group', action='store_false', help='param_group for optimizer')

    parser.add_argument('--no-autoindent', action='store_true', help='ipython dummy')
    return parser.parse_args()

if __name__ == '__main__':
    opt = parse_args()

    DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'
    dl_data = (DATA_URL + 'hotdog.zip', 'fba480ffa8aa7e0febbb511d181409f899b9baa5')
    data_dir = download_extract(dl_data, opt.data)

    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_augs = transforms.Compose([
        torchvision.transforms.RandomResizedCrop(224),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        normalize])
    test_augs = transforms.Compose([
        torchvision.transforms.Resize([256, 256]),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        normalize])

    train_imgs = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_augs)
    test_imgs = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=test_augs)

    # hotdogs = [train_imgs[i][0].permute(1, 2, 0) for i in range(8)]
    # not_hotdogs = [train_imgs[-i - 1][0].permute(1, 2, 0) for i in range(8)]
    # show_images(hotdogs + not_hotdogs, 2, 8, scale=1.4)

    train_dataset_loader = DataLoader(train_imgs, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    test_dataset_loader = DataLoader(test_imgs, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)

    net = torchvision.models.resnet18(pretrained=True)
    net.fc = torch.nn.Linear(net.fc.in_features, 2)
    nn.init.xavier_uniform_(net.fc.weight)

    loss = nn.CrossEntropyLoss(reduction='none')
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        net = net.cuda()
        loss = loss.cuda()

    if opt.param_group:
        params_1x = [param for name, param in net.named_parameters()
                            if name not in ["fc.weight", "fc.bias"]]
        optimizer = torch.optim.SGD([{'params': params_1x},
                                   {'params': net.fc.parameters(),
                                    'lr': opt.learning_rate * 10}],
                                    lr=opt.learning_rate, weight_decay=0.001)
    else:
        optimizer = torch.optim.SGD(net.parameters(), lr=opt.learning_rate, weight_decay=0.001)

    start_n_iter = 0
    start_eposch = 0
    if opt.resume:
        ckpt = load_checkpoint(opt.path_to_checkpoint)
        net.load_state_dict(ckpt['net'])
        start_eposch = ckpt['epoch']
        start_n_iter = ckpt['n_iter']
        optimizer.load_state_dict(ckpt['optim'])

    writer = SummaryWriter()

    running_loss, running_train_accuracy = RunningMean(), RunningMean()
    n_iter = start_n_iter
    for epoch in range(start_eposch, opt.epochs):
        n_iter += 1 # for tensorboard

        pbar = tqdm(enumerate(train_dataset_loader), total=len(train_dataset_loader))
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
            idx = epoch * len(train_dataset_loader) + i
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
            pbar = tqdm(enumerate(test_dataset_loader), total=len(test_dataset_loader))
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
                    idx = epoch * len(train_dataset_loader) + i
                    writer.add_scalar('test_accuracy', running_test_accuracy(), idx)


            save_checkpoint({
                'epoch': epoch,
                'net': net.state_dict(),
                'optim': optimizer.state_dict(),
                'n_iter': n_iter,
            }, 'ckpt_{}.pth'.format(epoch))





