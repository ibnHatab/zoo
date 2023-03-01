import argparse
import os
import sys
from typing import Tuple, Optional
from matplotlib import pyplot as plt

import torch
import numpy as np

try: sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
except: pass
# %cd nerf_from_nothing
sys.path.append('../src')

from geometry import crop_center, get_rays, inverse_rendering, sample_stratified
from model import EarlyStopping, NeRF, PositionalEncoder, init_model, nerf_forward
from utils import RunningMean, download


from tensorboardX import SummaryWriter
np.set_printoptions(suppress=True, precision=4)

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

    parser.add_argument('--no-autoindent', action='store_true', help='ipython dummy')
    return parser.parse_args()

if __name__ == '__main__':
    opt = parse_args()

    DATA_URL = 'http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/tiny_nerf_data.npz'
    fname = download(DATA_URL, opt.data)
    data = np.load(fname)
    images = data['images']
    poses = data['poses']
    focal = data['focal']

    # plt.imshow(testimg); plt.show()
    # inverse_rendering(poses, kos=True)

    height, width, _ = images[0].shape
    near, far = 2., 6.

    n_training = 100
    testimg_idx = 101

    images = torch.from_numpy(data['images'][:n_training])
    poses = torch.from_numpy(data['poses'][:n_training])
    focal = torch.from_numpy(data['focal'])
    testimg = torch.from_numpy(data['images'][testimg_idx])
    testpose = torch.from_numpy(data['poses'][testimg_idx])


    use_cuda = torch.cuda.is_available()
    if use_cuda:
        images = images.cuda()
        poses = poses.cuda()
        focal = focal.cuda()
        testimg = testimg.cuda()
        testpose = testpose.cuda()


    d_input = 3
    n_freqs = 10
    log_space = True
    use_viewdirs = True
    n_freq_viewdirs = 4

    n_samples = 64
    perturb = True
    inverse_depth = False

    d_filter = 128
    n_layers = 2
    skip = []
    use_fine_model = True
    d_filter_fine = 128
    n_layers_fine = 6

    n_samples_hierarchical = 64
    perturb_hierarchical = False

    lr = 5e-4

    n_iters = 10000
    batch_size = 2**14
    one_image_per_step = True
    chunk_size = 2**14
    center_crop = True
    center_crop_iters = 50
    diplay_rate = 25

    warmup_iters = 100
    warmup_min_fitness = 10.
    n_restarts = 10

    kwargs_sample_stratified = dict(n_samples=n_samples, perturb=perturb, inverse_depth=inverse_depth)
    kwargs_sample_hierarchical = dict(perturb=perturb_hierarchical)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model, model_fine, encode, encode_viewdirs, optimiser, warmup_stopper = init_model(
        d_input, n_layers, d_filter, skip,
        n_freqs, log_space, use_viewdirs, n_freq_viewdirs,
        use_fine_model, n_layers_fine, d_filter_fine,
        lr, device)

def train():
    if not one_image_per_step:
        height, width = images.shape[1:3]
        all_rays = torch.stack([torch.stack(get_rays(height, width, focal, p), 0)
                            for p in poses[:n_training]], 0)
        rays_rgb = torch.cat([all_rays, images[:, None]], 1)
        rays_rgb = torch.permute(rays_rgb, [0, 2, 3, 1, 4])
        rays_rgb = rays_rgb.reshape([-1, 3, 3])
        rays_rgb = rays_rgb.type(torch.float32)
        rays_rgb = rays_rgb[torch.randperm(rays_rgb.shape[0])]

        i_batch = 0

    train_psnrs = []
    val_psnrs = []
    iternums = []
    for i in range(n_iters):
        model.train()

        if one_image_per_step:
            target_img_idx = np.random.randint(images.shape[0])
            target_img = images[target_img_idx].to(device)
            if center_crop and i < center_crop_iters:
                target_img = crop_center(target_img)
            height, width = target_img.shape[:2]
            target_pose = poses[target_img_idx].to(device)
            rays_o, rays_d = get_rays(height, width, focal, target_pose)
            rays_o, rays_d = rays_o.reshape([-1,3]).to(device), rays_d.reshape([-1,3]).to(device)
        else:
            batch = rays_rgb[i_batch:i_batch+batch_size]
            batch = torch.transpose(batch, 0, 1)
            rays_o, rays_d, target_img = batch
            height, width = target_img.shape[:2]
            i_batch += batch_size
            if i_batch >= rays_rgb.shape[0]:
                rays_rgb = rays_rgb[torch.randperm(rays_rgb.shape[0])]
                i_batch = 0
        target_img = target_img.reshape((-1,3)).to(device)

        output = nerf_forward(rays_o, rays_d, near, far, encoding_fn=encode,
                            coarse_model=model, kwargs_sample_stratified=kwargs_sample_stratified,
                            n_samples_hierarchical=n_samples_hierarchical,
                            kwargs_sample_hierarchical=kwargs_sample_hierarchical,
                            fine_model=model_fine, viewdirs_encoding_fn=encode_viewdirs,
                            chunk_size=chunk_size)


