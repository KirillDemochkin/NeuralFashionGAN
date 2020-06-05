import argparse
import torch
from datasets.deepfashion2 import DeepFashion2Dataset
from datasets.style_dataset import StyleDataset
from datasets.custom_dataset import CustomDataset
from utils import visualization as vutils

from models.gaugan_generators import GauGANUnetStylizationGenerator
from models.encoders import StyleEncoder

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
parser.add_argument('--weights_generator', help='path to generator model', type=str, default='trained_models/NetGDeepFashionStyleGAN.best.pth')
parser.add_argument('--weights_style_encoder', help='path to style encoder', type=str, default='trained_models/NetSDeepFashionStyleGAN.best.pth')

args = parser.parse_args()

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
            args.dataset_dir, transform=transform, return_masked_image=True)
    else:
        dataset = CustomDataset(
            args.dataset_dir, transform=transform, return_masked_image=True)

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

    print('...Started generating images')

    # start generating clothes
    for i in range(1, min(len(dataset), args.max_num)):
        # take one dataset example
        image, mask, masked_image, _ = dataset[i]

        # make it like-batch
        image_ = image.unsqueeze(0).to(device)
        mask_ = mask.unsqueeze(0).to(device)
        masked_image_ = masked_image.unsqueeze(0).to(device)

        # take random style image
        style_image = styles_dataset.get_image().to(device)
        style_image_ = style_image.unsqueeze(0).to(device)

        _, test_skips = netS(masked_image_)
        test_embed, _ = netS(style_image_, False)
        test_generated = netG(test_embed, mask_, test_skips).detach().cpu()

        # save result
        a = vutils.save_image(test_generated, args.test_image_dir + str(i) + '.jpg',
            normalize=True, save=True)


if __name__ == '__main__':
    generate_images()
