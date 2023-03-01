
from typing import Tuple, Optional
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn

def inverse_rendering(poses, kos=False):
    dirs_x = np.stack([np.sum([1,0,0] * pose[:3,:3], axis=-1) for pose in poses])
    dirs_y = np.stack([np.sum([0,1,0] * pose[:3,:3], axis=-1) for pose in poses])
    dirs_z = np.stack([np.sum([0,0,1] * pose[:3,:3], axis=-1) for pose in poses])
    dirs_z_inv = np.stack([np.sum([0,0,-1] * pose[:3,:3], axis=-1) for pose in poses])
    origins = np.stack([pose[:3,-1] for pose in poses])

    ax = plt.figure(figsize=(10,10)).add_subplot(111, projection='3d')
    if kos:
        ax.quiver(origins[:,0], origins[:,1], origins[:,2], dirs_x[:,0], dirs_x[:,1], dirs_x[:,2], length=0.5, color='r')
        ax.quiver(origins[:,0], origins[:,1], origins[:,2], dirs_y[:,0], dirs_y[:,1], dirs_y[:,2], length=0.5, color='g')
        ax.quiver(origins[:,0], origins[:,1], origins[:,2], dirs_z[:,0], dirs_z[:,1], dirs_z[:,2], length=0.5, color='b')
    ax.quiver(origins[:,0], origins[:,1], origins[:,2], dirs_z_inv[:,0], dirs_z_inv[:,1], dirs_z_inv[:,2], length=0.5, color='k')
    plt.show(block=True)

def get_rays(height, width, focal, pose):
    r"""
    Find origin and direction of rays through every pixel and camera origin.
    """
    # Apply pinhole camera model to gather directions at each pixel
    i, j = torch.meshgrid(
        torch.arange(width, dtype=torch.float32).to(pose),
        torch.arange(height, dtype=torch.float32).to(pose),
        indexing='ij')
    i, j = i.transpose(-1, -2), j.transpose(-1, -2)
    directions = torch.stack([(i - width * .5) / focal,
                                -(j - height * .5) / focal,
                                -torch.ones_like(i)
                            ], dim=-1)

    # Apply camera pose to directions
    rays_d = torch.sum(directions[..., None, :] * pose[:3, :3], dim=-1)

    # Origin is same for all directions (the optical center)
    rays_o = pose[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d

def sample_stratified(rays_o, rays_d, near, far, n_samples, perturb=True, inverse_depth= False):
    r"""
    Sample along ray from regularly-spaced bins.
    """
    # Grab samples for space integration along ray
    t_vals = torch.linspace(0., 1., n_samples, device=rays_o.device)
    if not inverse_depth:
        # Sample linearly between `near` and `far`
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        # Sample linearly in inverse depth (disparity)
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

    # Draw uniform samples from bins along ray
    if perturb:
        mids = .5 * (z_vals[1:] + z_vals[:-1])
        upper = torch.concat([mids, z_vals[-1:]], dim=-1)
        lower = torch.concat([z_vals[:1], mids], dim=-1)
        t_rand = torch.rand([n_samples], device=z_vals.device)
        z_vals = lower + (upper - lower) * t_rand
    z_vals = z_vals.expand(list(rays_o.shape[:-1]) + [n_samples])

    # Apply scale from `rays_d` and offset from `rays_o` to samples
    # pts: (width, height, n_samples, 3)
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
    return pts, z_vals

def cumprod_exclusive(tensor):
    """ X = torch.tensor([[[1,2,3],[4,5,6],[7,8,9]]])
    cumprod_exclusive(X) """
    cumprod = torch.cumprod(tensor, dim=-1)
    cumprod = torch.roll(cumprod, shifts=1, dims=-1)
    cumprod[..., 0] = 1.
    return cumprod

def raw2outputs(raw, z_vals, rays_d, raw_noice_std, white_bkgd):
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, 1e10 * torch.ones_like(dists[..., :1])], dim=-1)
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    noise = 0
    if raw_noice_std > 0:
        noise = torch.randn_like(raw[..., :3].shape) * raw_noice_std

    alpha = 1. - torch.exp(-nn.functional.relu(raw[..., 3] + noise) * dists)
    weights = alpha * cumprod_exclusive(1. - alpha + 1e-10)
    rgb = torch.sigmoid(raw[..., :3])
    rgb_map = torch.sum(weights[..., :, None] * rgb, dim=-2)
    depth_map = torch.sum(weights * z_vals, dim=-1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map),
                            depth_map / torch.sum(weights, dim=-1))
    acc_map = torch.sum(weights, dim=-1)
    if white_bkgd:
        rgb_map = rgb_map + (1. - acc_map[..., None])

    return rgb_map, disp_map, acc_map, weights

def sample_pdf(bins, weights, n_samples, perturb=False):
    pdf = (weights + 1e-5) / torch.sum(weights + 1e-5, dim=-1, keepdim=True)
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = torch.concat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)

    if not perturb:
        u = torch.linspace(0., 1., n_samples, device=cdf.device)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples], device=cdf.device)

    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)

    below = torch.clamp(inds - 1, min=0)
    above = torch.clamp(inds, max=cdf.shape[-1]-1)
    inds_g = torch.stack([below, above], dim=-1)

    matched_shape = list(inds_g.shape[:-1]) + [cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(-2).expand(matched_shape), dim=-1, index=inds_g)
    bins_g = torch.gather(bins.unsqueeze(-2).expand(matched_shape), dim=-1, index=inds_g)

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples

def sample_hierarchical(rays_o, rays_d, z_vals, weights, n_samples, perturb=False):
    z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
    new_z_sample = sample_pdf(z_vals_mid, weights[..., 1:-1], n_samples, perturb)
    new_z_sample = new_z_sample.detach()

    z_vals_combined, _ = torch.sort(torch.cat([z_vals, new_z_sample], dim=-1), dim=-1)
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals_combined[..., :, None]
    return pts, z_vals_combined, new_z_sample

def plot_samples(z_vals, z_hierarch, ax):
    y_vals = 1 + torch.zeros_like(z_vals)

    if ax is None:
        ax = plt.subplots()
    ax.plot(z_vals, y_vals, 'b-o')
    if z_hierarch is not None:
        y_hierarchy = torch.zeros_like(z_hierarch)
        ax.plot(z_hierarch, y_hierarchy, 'r-o')
    ax.set_ylim([-1, 2])
    ax.grid(True)
    return ax

def crop_center(img, frac=0.5):
    h, w = img.shape[:2]
    h_crop = int(frac / 2 * h)
    w_crop = int(frac / 2 * w)
    return img[h_crop:-h_crop, w_crop:-w_crop]

