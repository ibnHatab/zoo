
import torch
import torch.nn as nn

from geometry import raw2outputs, sample_hierarchical, sample_stratified

class PositionalEncoder(nn.Module):
    def __init__(self, d_input, n_freqs, log_space=False):
        super().__init__()
        self.d_input = d_input
        self.n_freqs = n_freqs
        self.log_space = log_space
        self.d_output = d_input + (2 * n_freqs + 1)
        self.embed_fns = [lambda x: x]

        if self.log_space:
            freq_band = 2.**torch.linspace(0., n_freqs - 1, n_freqs)
        else:
            freq_band = torch.linspace(2.**0., 2.**(n_freqs - 1), n_freqs)

        for freq in freq_band:
            self.embed_fns.append(lambda x, freq=freq: torch.sin(freq * x))
            self.embed_fns.append(lambda x, freq=freq: torch.cos(freq * x))

    def forward(self, x):
        return torch.concat([fn(x) for fn in self.embed_fns], dim=-1)

class NeRF(nn.Module):
    def __init__(self, d_input=3, n_layers=8, d_filter=256, skip=(4,), d_viewdirs=None):
        super().__init__()
        self.d_input = d_input
        self.skip = skip
        self.act = nn.functional.relu
        self.d_viewdirs = d_viewdirs

        # Create model layers
        self.layers = nn.ModuleList(
        [nn.Linear(self.d_input, d_filter)] +
        [nn.Linear(d_filter + self.d_input, d_filter) if i in skip \
        else nn.Linear(d_filter, d_filter) for i in range(n_layers - 1)]
        )

        # Bottleneck layers
        if self.d_viewdirs is not None:
            # If using viewdirs, split alpha and RGB
            self.alpha_out = nn.Linear(d_filter, 1)
            self.rgb_filters = nn.Linear(d_filter, d_filter)
            self.branch = nn.Linear(d_filter + self.d_viewdirs, d_filter // 2)
            self.output = nn.Linear(d_filter // 2, 3)
        else:
            # If no viewdirs, use simpler output
            self.output = nn.Linear(d_filter, 4)

    def forward(self, x, viewdirs=None):
        x_input = x
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x))
            if i in self.skip:
                x = torch.cat([x, x_input], dim=-1)
        if self.d_viewdirs is not None:
            alpha = self.alpha_out(x)

            x = self.rgb_filters(x)
            x = torch.concat([x, viewdirs], dim=-1)
            x = self.act(self.branch(x))
            x = self.output(x)

            x = torch.concat([x, alpha], dim=-1)
        else:
            x = self.output(x)
        return x

def get_chunks(inputs, chunk_size=2**15):
    return [inputs[i:i+chunk_size] for i in range(0, inputs.shape[0], chunk_size)]

def prepare_chunks(points, encoding_function, chunk_size=2**15):
    points = points.reshape(-1, 3)
    points = encoding_function(points)
    points = get_chunks(points, chunk_size)
    return points

def prepare_viewdir_chunks(points, rays_d, encoding_function, chunk_size=2**15):
   viewdirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
   viewdirs = viewdirs[:, None, ...].expand(points.shape).reshape((-1, 3))
   viewdirs = encoding_function(viewdirs)
   viewdirs = get_chunks(viewdirs, chunk_size)
   return viewdirs

def nerf_forward(rays_o, rays_d, near, far, encoding_fn, coarse_model, kwargs_sample_stratified,
                 n_samples_hierarchical, kwargs_sample_hierarchical,
                 fine_model=None, viewdirs_encoding_fn=None, chunk_size=2**15):

    # Set no kwargs if none are given.
    if kwargs_sample_stratified is None:
        kwargs_sample_stratified = {}
    if kwargs_sample_hierarchical is None:
        kwargs_sample_hierarchical = {}

    # Sample query points along each ray.
    query_points, z_vals = sample_stratified(
        rays_o, rays_d, near, far, **kwargs_sample_stratified)

    # Prepare batches.
    batches = prepare_chunks(query_points, encoding_fn, chunk_size=chunk_size)
    if viewdirs_encoding_fn is not None:
        batches_viewdirs = prepare_viewdir_chunks(query_points, rays_d,
                                                viewdirs_encoding_fn,
                                                chunk_size=chunk_size)
    else:
        batches_viewdirs = [None] * len(batches)

    # Coarse model pass.
    # Split the encoded points into "chunks", run the model on all chunks, and
    # concatenate the results (to avoid out-of-memory issues).
    predictions = []
    for batch, batch_viewdirs in zip(batches, batches_viewdirs):
        predictions.append(coarse_model(batch, viewdirs=batch_viewdirs))
    raw = torch.cat(predictions, dim=0)
    raw = raw.reshape(list(query_points.shape[:2]) + [raw.shape[-1]])

    # Perform differentiable volume rendering to re-synthesize the RGB image.
    rgb_map, depth_map, acc_map, weights = raw2outputs(raw, z_vals, rays_d)
    # rgb_map, depth_map, acc_map, weights = render_volume_density(raw, rays_o, z_vals)
    outputs = {
        'z_vals_stratified': z_vals
    }

    # Fine model pass.
    if n_samples_hierarchical > 0:
        # Save previous outputs to return.
        rgb_map_0, depth_map_0, acc_map_0 = rgb_map, depth_map, acc_map

        # Apply hierarchical sampling for fine query points.
        query_points, z_vals_combined, z_hierarch = sample_hierarchical(
        rays_o, rays_d, z_vals, weights, n_samples_hierarchical,
        **kwargs_sample_hierarchical)

        # Prepare inputs as before.
        batches = prepare_chunks(query_points, encoding_fn, chunksize=chunk_size)
        if viewdirs_encoding_fn is not None:
            batches_viewdirs = prepare_viewdir_chunks(query_points, rays_d,
                                                    viewdirs_encoding_fn,
                                                    chunksize=chunk_size)
        else:
            batches_viewdirs = [None] * len(batches)

        # Forward pass new samples through fine model.
        fine_model = fine_model if fine_model is not None else coarse_model
        predictions = []
        for batch, batch_viewdirs in zip(batches, batches_viewdirs):
            predictions.append(fine_model(batch, viewdirs=batch_viewdirs))
        raw = torch.cat(predictions, dim=0)
        raw = raw.reshape(list(query_points.shape[:2]) + [raw.shape[-1]])

        # Perform differentiable volume rendering to re-synthesize the RGB image.
        rgb_map, depth_map, acc_map, weights = raw2outputs(raw, z_vals_combined, rays_d)

        # Store outputs.
        outputs['z_vals_hierarchical'] = z_hierarch
        outputs['rgb_map_0'] = rgb_map_0
        outputs['depth_map_0'] = depth_map_0
        outputs['acc_map_0'] = acc_map_0

    # Store outputs.
    outputs['rgb_map'] = rgb_map
    outputs['depth_map'] = depth_map
    outputs['acc_map'] = acc_map
    outputs['weights'] = weights
    return outputs


#     if kwargs_sample_stratified is None:
#         kwargs_sample_stratified = {}
#     if kwargs_sample_hierarchical is None:
#         kwargs_sample_hierarchical = {}

#     query_points, z_vals = sample_stratified(rays_o, rays_d, near, far, **kwargs_sample_stratified)

#     batches = prepare_chunks(query_points, encoding_fn, chunk_size)
#     if viewdirs_encoding_fn is not None:
#         viewdirs_batches = prepare_viewdir_chunks(query_points, rays_d, viewdirs_encoding_fn, chunk_size)
#     else:
#         viewdirs_batches = [None] * len(batches)

#     predictions = []
#     for batch, viewdirs_batch in zip(batches, viewdirs_batches):
#         predictions.append(coarse_model(batch, viewdirs_batch))

#     raw = torch.cat(predictions, dim=0)
#     raw = raw.reshape(list(query_points.shape[:2]) + [raw.shape[-1]])

#     rgb_map, disp_map, acc_map, weights = raw2outputs(raw, z_vals, rays_d)
#     output = {
#         'z_vals_stratified': z_vals,
#     }

#     if n_samples_hierarchical > 0:
#         rgb_map_0, depth_map_0, acc_map_0 = rgb_map, depth_map, acc_map
#         query_points, z_val_combined, z_hierarchy = sample_hierarchical(
#             rays_o, rays_d, z_vals, weights, n_samples_hierarchical, **kwargs_sample_hierarchical)

#         batches = prepare_chunks(query_points, encoding_fn, chunk_size)
#         if viewdirs_encoding_fn is not None:
#             viewdirs_batches = prepare_viewdir_chunks(query_points, rays_d, viewdirs_encoding_fn, chunk_size)
#         else:
#             viewdirs_batches = [None] * len(batches)

#         fine_model = fine_model or coarse_model

#         predictions = []
#         for batch, viewdirs_batch in zip(batches, viewdirs_batches):
#             predictions.append(fine_model(batch, viewdirs=viewdirs_batch))
#         raw = torch.cat(predictions, dim=0)
#         raw = raw.reshape(list(query_points.shape[:2]) + [raw.shape[-1]])

#         rgb_map, depth_map, acc_map, weights = raw2outputs(raw, z_val_combined, rays_d)

#         outputs = output.update({
#             'z_vals_hierarchical': z_hierarchy,
#             'rgb_map_0': rgb_map_0,
#             'depth_map_0': depth_map_0,
#             'acc_map_0': acc_map_0,
#         })

#     output.update({
#         'rgb_map': rgb_map,
#         'depth_map': depth_map,
#         'acc_map': acc_map,
#         'weights': weights,
#     })
#     return output

class EarlyStopping:
    def __init__(self, patience=30, margin=1e-4):
        self.patience = patience
        self.margin = margin
        self.best_fitness = 0.
        self.best_iter = 0.

    def __call__(self, iter, fitness):
        if (fitness - self.best_fitness) > self.margin:
            self.best_fitness = fitness
            self.best_iter = iter
        delta = iter - self.best_iter
        stop = delta > self.patience
        return stop


def init_model(d_input, n_layers, d_filter, skip,
               n_freqs, log_space, use_viewdirs, n_freq_viewdirs,
               use_fine_model, n_layers_fine, d_filter_fine,
               lr, device):
    encoder = PositionalEncoder(d_input, n_freqs, log_space)
    encode = lambda x: encoder(x)

    if use_viewdirs:
        encoder_viewdirs = PositionalEncoder(d_input, n_freq_viewdirs, log_space)
        encode_viewdirs = lambda x: encoder_viewdirs(x)
        d_viewdirs = encoder_viewdirs.d_output
    else:
        encode_viewdirs = None
        d_viewdirs = None

    model = NeRF(encoder.d_output, n_layers=n_layers, d_filter=d_filter, skip=skip, d_viewdirs=d_viewdirs)
    mode = model.to(device)
    model_params = list(model.parameters())

    if use_fine_model:
        model_fine = NeRF(encoder.d_output, n_layers=n_layers_fine, d_filter=d_filter_fine, skip=skip, d_viewdirs=d_viewdirs)
        model_fine = model_fine.to(device)
        model_params += list(model_fine.parameters())
    else:
        model_fine = None

    optimiser = torch.optim.Adam(model_params, lr=lr)

    warmup_stopper = EarlyStopping(patience=50)

    return model, model_fine, encode, encode_viewdirs, optimiser, warmup_stopper
