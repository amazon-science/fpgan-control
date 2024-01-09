#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  SPDX-License-Identifier: CC-BY-NC-4.0

import sys
import os
from pathlib import Path
from tqdm import tqdm
import argparse
import torch
from multiprocessing import Pool
import torchvision.transforms as transforms


_PWD = Path(__file__).absolute().parent
sys.path.append(os.path.join(Path(__file__).absolute(), 'src'))
from evaluation.inference_class import Inference


def duplicate_noise(noise, n):
    noise = [noise[i].expand(n, -1, -1, -1).clone() for i in range(len(noise))]
    return noise


def save_image(image_dict):
    image_dict['image'].save(image_dict['save_path'])


@torch.no_grad()
def make_pid(inference_model, number_images_per_ids):
    id_latent = torch.randn(1, 256, device='cuda').repeat(number_images_per_ids, 1)
    id_noise = duplicate_noise(inference_model.model.module.make_noise(1), number_images_per_ids)
    pose_latent = torch.randn(number_images_per_ids, 256, device='cuda')

    latent = torch.cat([id_latent, pose_latent], dim=1)
    images, _ = inference_model.model([latent], noise=id_noise)
    images = images.mul(0.5).add(0.5).clamp(min=0., max=1.)

    image_list = torch.split(images, 1)

    return image_list


def generate_synth_ids(model_dir, save_path, number_of_ids, number_images_per_ids):
    to_pil = transforms.ToPILImage()
    inference_model = Inference(model_dir)
    instance_num = 0
    res_flag = False
    for id_num in tqdm(range(number_of_ids)):
        id_name = 'ID_%07d' % id_num
        id_save_dir = os.path.join(save_path, id_name)
        os.makedirs(id_save_dir, exist_ok=True)
        id_image_list = make_pid(inference_model, number_images_per_ids)
        image_dict_list = []
        for im_num, image in enumerate(id_image_list):
            image = image.cpu()
            image = to_pil(image[0])
            save_name = '%s_im%03d_instance%07d.png' % (id_name, im_num, instance_num)
            im_save_path = os.path.join(id_save_dir, save_name)
            images_dict = {'save_path': im_save_path, 'image': image}
            image_dict_list.append(images_dict)
            instance_num += 1
        if res_flag:
            res.wait()
        else:
            pool = Pool(processes=20)
        res = pool.map_async(save_image, image_dict_list, 1)
        res_flag = True
    if res_flag:
        res.wait()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default=os.path.join(_PWD, 'models/id06fre20_fingers384_id_noise_same_id_idl005_posel000_large_pose_20230606-082209'))
    parser.add_argument('--save_path', type=str, default='where to save the images')
    parser.add_argument('--number_of_ids', type=int, default=100)
    parser.add_argument('--number_images_per_id', type=int, default=11)

    args = parser.parse_args()

    generate_synth_ids(
            args.model_dir,
            args.save_path,
            args.number_of_ids,
            args.number_images_per_id
        )