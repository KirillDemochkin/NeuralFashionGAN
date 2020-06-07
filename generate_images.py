import argparse
import torch
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import random
import os
from os.path import join
from torch.utils.data import Dataset
# from datasets.deepfashion2 import DeepFashion2Dataset
from datasets.style_dataset import StyleDataset
from datasets.custom_dataset import CustomDataset

from models.gaugan_generators import GauGANUnetStylizationGenerator
from models.encoders import StyleEncoder, MappingNetwork

from albumentations import (
    Compose,
    Resize
)

from albumentations.pytorch import ToTensorV2


parser = argparse.ArgumentParser()
parser.add_argument('--test_dataset_name', help='dataset short name to work with: deepfashion2 or custom', type=str, default='deepfashion2')
parser.add_argument('--resize_height', help='images will be resized to this height', type=int, default=128)
parser.add_argument('--resize_width', help='images will be resized to this width', type=int, default=128)
parser.add_argument('--style_dataset_dir', help='path to style images', type=str, default='styles')
parser.add_argument('--dataset_dir', help='path to dataset', type=str, default='custom')
parser.add_argument('--mask_channels', default=13, type=float)
parser.add_argument('--encoder_latent_dim', default=256, type=float)
parser.add_argument('--unet_ch', default=4, type=float)
parser.add_argument('--max_num', help='maximum number of examples to process', default=10, type=int)
parser.add_argument('--test_image_dir', help='path to where to save generated images', type=str, default='results')
parser.add_argument('--weights_generator', help='path to generator model', type=str, default='trained_models/NetGDeepFashionStyleGANImproved.best.pth')
parser.add_argument('--weights_style_encoder', help='path to style encoder', type=str, default='trained_models/NetSDeepFashionStyleGANImproved.best.pth')
parser.add_argument('--weights_style_mapping', help='path to mapping network', type=str, default='trained_models/NetMDeepFashionStyleGANImproved.best.pth')

args = parser.parse_args()

import json
from pycocotools import mask as maskUtils

def ann_to_mask(segm, h, w):
    rles = maskUtils.frPyObjects(segm, h, w)
    rle = maskUtils.merge(rles)
    m = maskUtils.decode(rle)
    return m

# DeepFashion2Dataset modified for convinient generation of images
class DeepFashion2Dataset(Dataset):
    def __init__(self, root, num_classes=13, transform=None, return_masked_image=False, noise = False):
        super().__init__()

        self.image_folder = join(root, 'image')
        self.annos_folder = join(root, 'annos')
        self.length = len(os.listdir(self.image_folder))
        self.num_classes = num_classes
        self.transform = transform
        self.return_masked_image = return_masked_image
        self.noise = noise

    def get_item(self, idx, average_color=None):
        name = f'{idx + 1:06d}'

        image_path = join(self.image_folder, name + '.jpg')
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.

        width, height = image.shape[1], image.shape[0]

        ann_path = join(self.annos_folder, name + '.json')
        with open(ann_path) as f:
            ann = json.load(f)

        full_mask = np.zeros((height, width, self.num_classes),
                             dtype=np.float32)
        for key, item in ann.items():
            if key in ['source', 'pair_id']:
                continue
            clid = item['category_id']
            mask = ann_to_mask(item['segmentation'], height, width)
            full_mask[mask > 0, clid - 1] = 1

        if self.transform is not None:
            augmented = self.transform(image=image, mask=full_mask)
            image = augmented['image']
            full_mask = augmented['mask'].permute(2, 0, 1)

        image *= 2
        image -= 1

        if self.return_masked_image:
            masked_image = image.clone()

            if average_color is None:
                average_color = torch.mean(masked_image.view(3, -1), dim=-1)
            m = full_mask.sum(dim=0) > 0
            masked_image[0, m] = average_color[0]
            masked_image[1, m] = average_color[1]
            masked_image[2, m] = average_color[2]
            # masked_image[:, m] = 1.0
            if self.noise:
                noise = torch.zeros_like(masked_image).uniform_(-0.1, 0.1)
                noise[:, full_mask.sum(dim=0) <= 0] = 0.
                masked_image += noise
                masked_image = torch.clamp(masked_image, -1, 1)
            loss_mask = torch.ones_like(masked_image)
            loss_mask[:, full_mask.sum(dim=0) > 0] = 0.
            return image, full_mask, masked_image, loss_mask
        else:
            return image, full_mask

    def __len__(self):
        return self.length


def image_to_plot(image):
    image_p = (np.transpose(image, [1, 2, 0]) + 1 / 2)
    return image_p

def generate_images():

    # transformation for all images is the same
    transform = Compose([
        Resize(args.resize_height, args.resize_width),
        ToTensorV2(),
        ])

    # create StyleDataset that will randomly give style image
    styles_dataset = StyleDataset(args.style_dataset_dir, transform=transform)

    # create dataset
    if args.test_dataset_name == 'deepfashion2':
        dataset = DeepFashion2Dataset(
            args.dataset_dir, transform=transform, return_masked_image=True, noise=True)
    else:
        dataset = CustomDataset(
            args.dataset_dir, transform=transform, return_masked_image=True, noise=True)

    print('...Length of dataset:', len(dataset))

    # set device - cpu
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    print('...Loading models')

    # load generator
    netG = GauGANUnetStylizationGenerator(args.mask_channels, args.encoder_latent_dim, 2, args.unet_ch, device)
    state_dict_generator = torch.load(args.weights_generator, map_location=device)
    netG.load_state_dict(state_dict_generator)
    netG.to(device)

    # load style encoder
    netS = StyleEncoder(args.encoder_latent_dim, args.unet_ch, 2)
    state_dict_style_encoder = torch.load(args.weights_style_encoder, map_location=device)
    netS.load_state_dict(state_dict_style_encoder)
    netS.to(device)

    # load mapping network
    netM = MappingNetwork(args.encoder_latent_dim)
    state_dict_mapping= torch.load(args.weights_mapping, map_location=device)
    netM.load_state_dict(state_dict_mapping)
    netM.to(device)

    print('...Started generating images')

    # start generating clothes
    for i in range(1, min(len(dataset), args.max_num)):

        # take style image
        style_image = styles_dataset.get_image()
        style_image_net = style_image.unsqueeze(0).to(device)

        average_color = torch.mean(style_image.view(3, -1), dim=-1)
        # print(average_color)

        # take dataset image
        image, mask, masked_image, _ = dataset.get_item(1, average_color)

        # make it like-batch
        image_net = image.unsqueeze(0).to(device)
        mask_net = mask.unsqueeze(0).to(device)
        masked_image = masked_image.unsqueeze(0).to(device)

        # generate new image
        _, test_skips = netS(masked_image)
        # test_embed, _ = netS(image_net, False)
        test_embed, _ = netS(style_image_net, False)
        style_code, _, _ = netM(test_embed)
        test_generated = netG(style_code, mask_net.float(), test_skips).detach().cpu()

        # save result
        generated = test_generated.squeeze(0).detach().cpu().numpy()

        norm_image = cv2.normalize(image_to_plot(generated), None,
                           alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F).astype(np.uint8)

        im = Image.fromarray(norm_image)
        im.save(args.test_image_dir + '/' str(i) + '.jpg')

if __name__ == '__main__':
    generate_images()
