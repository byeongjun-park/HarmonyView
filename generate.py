import argparse
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from skimage.io import imsave

from ldm.models.diffusion.sync_dreamer import SyncMultiviewDiffusion, SyncDDIMSampler
from ldm.util import instantiate_from_config, prepare_inputs


def load_model(cfg,ckpt,strict=True):
    config = OmegaConf.load(cfg)
    model = instantiate_from_config(config.model)
    print(f'loading model from {ckpt} ...')
    ckpt = torch.load(ckpt,map_location='cpu')
    model.load_state_dict(ckpt['state_dict'],strict=strict)
    model = model.cuda().eval()
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg',type=str, default='configs/syncdreamer.yaml')
    parser.add_argument('--ckpt',type=str, default='ckpt/syncdreamer-pretrain.ckpt')
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--elevation', type=float, required=True)

    parser.add_argument('--sample_num', type=int, default=2)
    parser.add_argument('--crop_size', type=int, default=-1)
    parser.add_argument('--cfg_scales', type=float, nargs='+', default=[1.0, 1.0])
    parser.add_argument('--decomposed_sampling', action='store_true')
    parser.add_argument('--seed', type=int, default=6033)

    parser.add_argument('--sampler', type=str, default='ddim')
    parser.add_argument('--sample_steps', type=int, default=50)
    flags = parser.parse_args()

    torch.random.manual_seed(flags.seed)
    np.random.seed(flags.seed)

    model = load_model(flags.cfg, flags.ckpt, strict=False)
    assert isinstance(model, SyncMultiviewDiffusion)
    Path(f'{flags.output}').mkdir(exist_ok=True, parents=True)

    # prepare data
    data = prepare_inputs(flags.input, flags.elevation, flags.crop_size)
    for k, v in data.items():
        data[k] = v.unsqueeze(0).cuda()
        data[k] = torch.repeat_interleave(data[k], flags.sample_num, dim=0)

    if flags.sampler=='ddim':
        sampler = SyncDDIMSampler(model, flags.sample_steps)
    else:
        raise NotImplementedError

    if flags.decomposed_sampling:
        mode = 'DG'
        print("Decomposed Guidance (DG) Sampling")
        print(f"unconditional scales {flags.cfg_scales}")
    else:
        mode = 'CFG'
        print("Classifier-Free Guidance (CFG) Sampling")
        print(f"unconditional scale {flags.cfg_scales[0]:.1f}")

    if flags.decomposed_sampling:
        x_sample = model.sample(sampler, data, flags.cfg_scales)
    else:
        x_sample = model.sample(sampler, data, flags.cfg_scales[0])

    B, N, _, H, W = x_sample.shape
    x_sample = (torch.clamp(x_sample, max=1.0, min=-1.0) + 1) * 0.5
    x_sample = x_sample.permute(0, 1, 3, 4, 2).cpu().numpy() * 255
    x_sample = x_sample.astype(np.uint8)

    save_fig = []
    for bi in range(B):
        tmp_img = np.concatenate([x_sample[bi, ni] for ni in range(N)], 1)
        output_fn = Path(flags.output) / f'{bi}.png'
        imsave(output_fn, tmp_img)
        save_fig.append(tmp_img)

    save_fig.append(np.concatenate([x_sample[0,ni] for ni in range(N)], 1))
    if flags.decomposed_sampling:
        output_fn = Path(flags.output)/ f'{mode}_{flags.cfg_scales[0]}_{flags.cfg_scales[1]}.png'
    else:
        output_fn = Path(flags.output)/ f'{mode}_{flags.cfg_scales[0]}.png'

    imsave(output_fn, np.concatenate([save_fig[bi] for bi in range(flags.sample_num)], 0))

if __name__=="__main__":
    main()

