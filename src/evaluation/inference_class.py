#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  SPDX-License-Identifier: CC-BY-NC-4.0

import os
import torch

from utils.file_utils import read_json
from utils.logging_utils import get_logger
from utils.mini_batch_multi_split_utils import MiniBatchUtils
from models.gan_model import Generator

_log = get_logger(__name__)


class Inference():
    def __init__(self, model_dir, ckpt=None):
        _log.info('Init inference class...')
        self.model_dir = model_dir
        self.ckpt = ckpt
        self.model, self.batch_utils, self.config, self.ckpt_iter = self.retrieve_model(model_dir, ckpt)
        self.noise = None
        self.noise = self.reset_noise()
        self.mean_w_latent = None
        self.mean_w_latents = None

    def calc_mean_w_latents(self):
        _log.info('Calc mean_w_latents...')
        mean_latent_w_list = []
        for i in range(100):
            latent_z = torch.randn(1000, self.config.model_config['latent_size'], device='cuda')
            if isinstance(self.model, torch.nn.DataParallel):
                latent_w = self.model.module.style(latent_z).cpu()
            else:
                latent_w = self.model.style(latent_z).cpu()
            mean_latent_w_list.append(latent_w.mean(dim=0).unsqueeze(0))
        self.mean_w_latent = torch.cat(mean_latent_w_list, dim=0).mean(0)
        self.mean_w_latents = {}
        for place_in_latent_key in self.batch_utils.place_in_latent_dict.keys():
            self.mean_w_latents[place_in_latent_key] = self.mean_w_latent[self.batch_utils.place_in_latent_dict[place_in_latent_key][0]: self.batch_utils.place_in_latent_dict[place_in_latent_key][1]]

    def reset_noise(self):
        if isinstance(self.model, torch.nn.DataParallel):
            self.noise = self.model.module.make_noise(device='cuda')
        else:
            self.noise = self.model.make_noise(device='cuda')

    @staticmethod
    def expend_noise(noise, batch_size):
        noise = [torch.cat([noise[n].clone() for _ in range(batch_size)], dim=0) for n in range(len(noise))]
        return noise

    def calc_truncation(self, latent_w, truncation=0.7):
        if truncation >= 1:
            return latent_w
        if self.mean_w_latents is None:
            self.calc_mean_w_latents()
        for key in self.batch_utils.place_in_latent_dict.keys():
            place_in_latent = self.batch_utils.place_in_latent_dict[key]
            latent_w[:, place_in_latent[0]: place_in_latent[1]] = \
                truncation * (latent_w[:, place_in_latent[0]: place_in_latent[1]] - torch.cat(
                    [self.mean_w_latents[key].clone().unsqueeze(0) for _ in range(latent_w.shape[0])], dim=0
                ).cuda()) + torch.cat(
                    [self.mean_w_latents[key].clone().unsqueeze(0) for _ in range(latent_w.shape[0])], dim=0
                ).cuda()
        return latent_w

    def check_valid_group(self, group):
        if group not in self.batch_utils.sub_group_names:
            raise ValueError(
                'group: %s not in valid group names for this model\n'
                'Valid group names are:\n'
                '%s' % str(self.batch_utils.sub_group_names)
            )

    @staticmethod
    def retrieve_model(model_dir, ckpt):
        config_path = os.path.join(model_dir, 'args.json')

        _log.info('Retrieve config from %s' % config_path)
        checkpoints_path = os.path.join(model_dir, 'checkpoint')
        ckpt_list = list(os.listdir(checkpoints_path))
        ckpt_list.sort()
        if ckpt is None:
            ckpt_path = ckpt_list[-1]
            ckpt_iter = ckpt_path.split('.')[0]
        else:
            ckpt_path = None
            ckpt_iter = None
            for ckpt_path in ckpt_list:
                if int(ckpt_path.split('.')[0]) == ckpt and str(ckpt) in ckpt_path:
                    ckpt_iter = ckpt_path.split('.')[0]
                    break
        _log.info('Loading %s ckpt' % ckpt_path)
        config = read_json(config_path, return_obj=True)
        ckpt = torch.load(os.path.join(checkpoints_path, ckpt_path))

        batch_utils = None
        if not config.model_config['vanilla']:
            _log.info('Init Batch Utils...')
            batch_utils = MiniBatchUtils(
                config.training_config['mini_batch'],
                config.training_config['sub_groups_dict'],
                total_batch=config.training_config['batch']
            )
            batch_utils.print()

        _log.info('Init Model...')
        model = Generator(
            config.model_config['size'],
            config.model_config['latent_size'],
            config.model_config['n_mlp'],
            channel_multiplier=config.model_config['channel_multiplier'],
            out_channels=config.model_config['img_channels'],
            split_fc=config.model_config['split_fc'],
            fc_config=None if config.model_config['vanilla'] else batch_utils.get_fc_config(),
            conv_transpose=config.model_config['conv_transpose'],
            noise_mode=config.model_config['g_noise_mode']
        ).cuda()
        _log.info('Loading Model: %s, ckpt iter %s' % (model_dir, ckpt_iter))
        model.load_state_dict(ckpt['g_ema'])
        model = torch.nn.DataParallel(model)
        model.eval()

        return model, batch_utils, config, ckpt_iter




