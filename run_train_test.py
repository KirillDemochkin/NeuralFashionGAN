import argparse
import pickle
import time

import numpy as np
import os
import sys
from datetime import datetime
import torch
import torch.nn
import torch.utils.data as data_utils
import torch.optim as optim
from tensorboardX import SummaryWriter
#import matplotlib.pyplot as plt
from datasets.gaugan_datasets import CocoDataset
# from datasets.deepfashion2 import DeepFashion2Dataset
from models.gaugan_generators import GauGANGenerator
from models.discriminator import MultiscaleDiscriminator
from models.encoders import BasicEncoder, Vgg19Full
from utils.weights_init import weights_init
from utils import losses
from utils import visualization as vutils

parser = argparse.ArgumentParser()
parser.add_argument('--data_root', help='path to data', type=str, default= '/home/kdemochkin/NeuralFashionGAN/data')
parser.add_argument('--root_path', help='path', type=str, default='/home/kdemochkin/NeuralFashionGAN')
parser.add_argument('--basenetG', help='pretrained generator model')
parser.add_argument('--basenetD', help='pretrained discriminator model')
parser.add_argument('--basenetE', help='pretrained encoder model')
parser.add_argument('--jaccard_threshold', default=0.5,
                    type=float, help='Min Jaccard index for matching')
parser.add_argument('-b', '--batch_size', default=32,
                    type=int, help='Batch size for training')
parser.add_argument('--num_workers', default=3,
                    type=int, help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True,
                    type=bool, help='Use cuda to train model')
parser.add_argument('--lr', '--learning-rate',
                    default=0.0002, type=float, help='initial learning rate')
parser.add_argument('-epoch', '--max_epoch', default=200,
                    type=int, help='max epoch for training')
parser.add_argument('--save_folder', default='img/',
                    help='Location to save checkpoint models')
parser.add_argument('--save_frequency', default=1)
parser.add_argument('--test_frequency', default=10)
parser.add_argument('--mode', type=str, choices=['train', 'test'], default='train')
parser.add_argument('--weight_decay', default=5e-4,
                    type=float, help='Weight decay for SGD')
parser.add_argument('--momentum', default=0.999, type=float, help='momentum')
parser.add_argument('--betas', default=0.5,
                    type=float)
parser.add_argument('--fm_lambda', default=10, type=float)
parser.add_argument('--kl_lambda', default=0.05, type=float)
parser.add_argument('--encoder_latent_dim', default=256, type=float)
parser.add_argument('--mask_channels', default=182, type=float)
parser.add_argument('--load', default=False, help='resume net for retraining')
args = parser.parse_args()

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)
sys.stdout.flush()
test_save_dir = args.save_folder
if not os.path.exists(test_save_dir):
    os.makedirs(test_save_dir)


def setup_experiment(title, logdir="./tb"):
    experiment_name = "{}@{}".format(title, datetime.now().strftime("%d.%m.%Y-%H:%M:%S")).replace(":","_")
    writer = SummaryWriter(log_dir=os.path.join(logdir, experiment_name))
    best_model_path = f"{title}.best.pth"
    return writer, experiment_name, best_model_path


##LOAD DATE
# train_images_dir = 'train/image/'
# train_annos_dir = 'train/annos/'
# validation_images_dir = 'validation/image/'
# validation_annos_dir = 'validation/annos/'
# test_images_dir = 'test/image/'
# train_coco_annos = 'cocoInstances_train.json'
# validation_coco_annos = 'cocoInstances_validation.json'
coco_images_dir = str(args.data_root) + '/coco-stuff/val_images2017/'
coco_masks_dir = str(args.data_root) + '/coco-stuff/val_labels2017/'
coco_images_files = os.listdir(coco_images_dir)
coco_images_files.sort()
coco_train_images_files = coco_images_files
#coco_val_images_files = coco_images_files[4000:4500]
#coco_test_images_files = coco_images_files[4500:5000]
coco_labels_num = 182
# train_dataset = DeepFashion2Dataset(train_images_dir,  train_coco_annos)
# val_dataset = DeepFashion2Dataset(validation_images_dir, validation_coco_annos)
coco_train_dataset = CocoDataset(coco_images_dir, coco_masks_dir, coco_train_images_files, coco_labels_num)
#coco_val_dataset = CocoDataset(coco_images_dir, coco_masks_dir, coco_val_images_files, coco_labels_num)
#coco_test_dataset = CocoDataset(coco_images_dir, coco_masks_dir, coco_test_images_files, coco_labels_num)

print('Loading Dataset...')
sys.stdout.flush()
train_loader = data_utils.DataLoader(coco_train_dataset, batch_size=args.batch_size, shuffle=True)
#val_loader = data_utils.DataLoader(coco_val_dataset, batch_size=args.batch_size, shuffle=True)
#test_loader = data_utils.DataLoader(coco_test_dataset, batch_size=args.batch_size, shuffle=True)

test_batch = next(iter(train_loader))
fixed_test_images = test_batch[0].to(device)
fixed_test_masks = test_batch[1].to(device)
_ = vutils.save_image(fixed_test_images.cpu().data[:16], '!test.png', normalize=True)

##MODE
netD = MultiscaleDiscriminator(args.mask_channels + 3).to(device)
netD.apply(weights_init)

netG = GauGANGenerator(args.mask_channels, args.encoder_latent_dim, 1).to(device)
netG.apply(weights_init)

netE = BasicEncoder(args.encoder_latent_dim, reduce_size=4).to(device)
netE.apply(weights_init)

vgg = Vgg19Full()
vgg.eval()


writer, experiment_name, best_model_path = setup_experiment(netG.__class__.__name__, logdir=os.path.join(args.root_path, "tb"))
print(f"Experiment name: {experiment_name}")
sys.stdout.flush()

if args.load:
    # load network
    resume_netG_path = args.basenetG
    resume_netD_path = args.basenetD
    resume_encoder_path = args.basenetE
    print('Loading resume network', resume_netG_path, resume_netD_path, resume_encoder_path)
    sys.stdout.flush()
    netG.load(resume_netG_path)
    netD.load(resume_netD_path)
    netE.load(resume_encoder_path)

optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(args.betas, args.momentum),
                        weight_decay=args.weight_decay)
optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.betas, args.momentum),
                        weight_decay=args.weight_decay)
optimizerE = optim.Adam(netE.parameters(), lr=args.lr, betas=(args.betas, args.momentum),
                        weight_decay=args.weight_decay)


def train():
    # Inspired by https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
    # Lists to keep track of progress
    num_epochs = args.max_epoch
    #img_list = []
    iters = 0
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Starting Training Loop...")
    sys.stdout.flush()
    for epoch in range(num_epochs):
        start = time.time()
        G_losses = []
        D_losses = []
        for i, data in enumerate(train_loader, 0):
            global_i = len(train_loader) * epoch + i
            ############################
            # D network
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            real_image, mask = data[0].to(device), data[1].to(device)
            print(real_image.shape, mask.shape)
            real_preds, real_feats = netD(real_image, mask)
            ## Train with all-fake batch
            # noise = torch.randn(b_size, nz, 1, 1, device=device)
            latent_code, mu, sigma = netE(real_image)
            fake = netG(latent_code, mask)
            print(fake.shape)
            fake_preds, fake_feats = netD(fake.detach(), mask)

            errD = 0.0
            for fp, rp in zip(fake_preds, real_preds):
                errD += losses.hinge_loss_discriminator(fp, rp)
            errD.backward()
            optimizerD.step()

            # dump train metrics to tensorboard
            if writer is not None:
                writer.add_scalar(f"loss_D", errD.item(), global_i)
            ############################
            # G network
            ###########################
            netG.zero_grad()
            netE.zero_grad()
            dkl = args.kl_lambda * losses.KL_divergence(mu, sigma)
            dkl.backward(retain_graph=True)
            fake_vgg_feats = vgg(fake)
            real_vgg_feats = vgg(real_image)
            errG_p = 0.0
            for ff, rf in zip(fake_vgg_feats, real_vgg_feats):
                errG_p += losses.perceptual_loss(ff, rf.detach(), args.fm_lambda)
            errG_p.backward(retain_graph=True)
            fake_preds, fake_feats = netD(fake, mask) ##view -1
            errG_hinge = 0.0
            for fp in fake_preds:
                errG_hinge += losses.hinge_loss_generator(fp)
            errG_hinge.backward(retain_graph=True)
            errG_fm = 0.0
            for ff, rf in zip(fake_feats, real_feats):
                errG_fm += losses.perceptual_loss(ff, rf.detach(), args.fm_lambda)
            errG_fm.backward()
            errG = errG_hinge.item() + errG_fm.item()

            optimizerG.step()
            optimizerE.step()
            if writer is not None:
                writer.add_scalar(f"loss_G", errG, global_i)
            # Output training stats
            if i % 100 == 99:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\t'
                      % (epoch, num_epochs, i, len(train_loader), errD.item(), errG))
                sys.stdout.flush()

            G_losses.append(errG)
            D_losses.append(errD)

            # Check how the generator is doing by saving G's output on fixed_noise
        end = time.time()
        hours, rem = divmod(end - start, 3600)
        minutes, seconds = divmod(rem, 60)
        if epoch % args.save_frequency == 0:
            with torch.no_grad():
                netG.eval()
                netE.eval()
                test_generated = netG(netE(fixed_test_images)[0], fixed_test_masks).detach().cpu()
                netG.train()
                netE.train()
            #img_list.append(fake.data.numpy())

            print("Epoch %d - Elapsed time: {:0>2}:{:0>2}:{:05.2f}".format(epoch, int(hours), int(minutes), seconds))
            sys.stdout.flush()
            tim = vutils.save_image(test_generated.data[:16], '%s/%d.png' % (test_save_dir, epoch), normalize=True)
            writer.add_image('generated', tim, epoch, dataformats='HWC')
            torch.save(netG.state_dict(), os.path.join(args.root_path, 'NetG' + best_model_path ))
            torch.save(netD.state_dict(), os.path.join(args.root_path, 'NetD' + best_model_path))
            torch.save(netE.state_dict(), os.path.join(args.root_path, 'NetE' + best_model_path))
            #plt.imsave(os.path.join(
                #'./{}/'.format(test_save_dir) + 'img{}.png'.format(datetime.now().strftime("%d.%m.%Y-%H:%M:%S"))),
                       #((img_list[-1][0] + 1) / 2.0).transpose([1, 2, 0]), cmap='gray', interpolation="none")
            if writer is not None:
                writer.add_scalar(f"loss_G_epoch", np.sum(G_losses) / len(train_loader), epoch)
                writer.add_scalar(f"loss_D_epoch", np.sum(D_losses) / len(train_loader), epoch)
            iters += 1


def test_net():
    pass


if __name__ == '__main__':
    if args.mode == 'train':
        train()
    elif args.mode == 'test':
        test_net()
