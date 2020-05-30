import argparse
import pickle
import time

import numpy as np
import os
import datetime
import torch
import torch.nn
import torch.utils.data as data_utils
import torch.optim as optim
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
import matplotlib.pyplot as plt

from datasets.deepfashion2 import DeepFashion2Dataset
from models.gaugan_generators import GauGANGenerator
from models.discriminator import GauGANDiscriminator
from utils.weights_init import weights_init


parser = argparse.ArgumentParser()
parser.add_argument('--basenetG', help='pretrained base model')
parser.add_argument('--basenetD', help='pretrained base model')
parser.add_argument('--jaccard_threshold', default=0.5,
                    type=float, help='Min Jaccard index for matching')
parser.add_argument('-b', '--batch_size', default=8,
                    type=int, help='Batch size for training')
parser.add_argument('--num_workers', default=0,
                    type=int, help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True,
                    type=bool, help='Use cuda to train model')
parser.add_argument('--lr', '--learning-rate',
                    default=1e-3, type=float, help='initial learning rate')
parser.add_argument('-epoch', '--max_epoch', default=30,
                    type=int, help='max epoch for training')
parser.add_argument('--save_folder', default='img/',
                    help='Location to save checkpoint models')
parser.add_argument('--save_frequency', default=10)
parser.add_argument('--test_frequency', default=10)
parser.add_argument('--mode', type=str, choices=['train', 'test'], default='train')
parser.add_argument('--weight_decay', default=5e-4,
                    type=float, help='Weight decay for SGD')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--betas', default=0.0,
                    type=float)
parser.add_argument('--load', default=False, help='resume net for retraining')
args = parser.parse_args()


test_save_dir = args.save_folder
if not os.path.exists(test_save_dir):
    os.makedirs(test_save_dir)
def setup_experiment(title, logdir="./tb"):
    experiment_name = "{}@{}".format(title, datetime.now().strftime("%d.%m.%Y-%H:%M:%S"))
    writer = SummaryWriter(log_dir=os.path.join(logdir, experiment_name))
    best_model_path = f"{title}.best.pth"
    return writer, experiment_name, best_model_path



##LOAD DATE
train_images_dir = 'train/image/'
train_annos_dir = 'train/annos/'
validation_images_dir = 'validation/image/'
validation_annos_dir = 'validation/annos/'
test_images_dir = 'test/image/'
train_coco_annos = 'cocoInstances_train.json'
validation_coco_annos = 'cocoInstances_validation.json'

train_dataset = DeepFashion2Dataset(train_images_dir,  train_coco_annos)
val_dataset = DeepFashion2Dataset(validation_images_dir, validation_coco_annos)

print('Loading Dataset...')
train_loader = data_utils.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = data_utils.DataLoader(val_dataset , batch_size=args.batch_size, shuffle=True)

##MODEL
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

netD = GauGANDiscriminator.to(device)
netD.apply(weights_init)

netG = GauGANGenerator.to(device)
netG.apply(weights_init)

writer, experiment_name, best_model_path = setup_experiment(netG.__class__.__name__, logdir="./tb")
print(f"Experiment name: {experiment_name}")

if args.load:
    # load network
    resume_netG_path = args.basenetG
    resume_netD_path = args.basenetD
    print('Loading resume network', resume_netG_path, resume_netD_path)
    netG.load(resume_netG_path)
    netD.load(resume_netD_path)


optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(args.betas, args.momentum), weight_decay=args.weight_decay)
optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.betas, args.momentum), weight_decay=args.weight_decay)
criterion = ()


def train():
    # Inspired by https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
    # Lists to keep track of progress
    num_epochs = args.max_epoch
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Starting Training Loop...")
    for epoch in range(num_epochs):
        for i, data in enumerate(train_loader, 0):
            global_i = len(train_loader) * epoch + i
            ############################
            # D network
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            real_cpu = data.to(device)
            output = netD(real_cpu).view(-1)
            errD_real = criterion(output, torch.zeros_like(output))
            errD_real.backward()


            ## Train with all-fake batch
            noise = torch.randn(data.shape[0], 256, device=device)
            # noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake = netG(noise)

            output = netD(fake.detach()).view(-1)
            errD_fake = criterion(output, torch.ones_like(output))
            errD_fake.backward()
            errD = errD_real + errD_fake
            optimizerD.step()

            # dump train metrics to tensorboard
            if writer is not None:
                writer.add_scalar(f"loss_D", errD.item(), global_i)
            ############################
            # G network
            ###########################
            netG.zero_grad()
            output = netD(fake).view(-1)
            errG = criterion(output, torch.zeros_like(output))
            errG.backward()
            optimizerG.step()
            if writer is not None:
                writer.add_scalar(f"loss_G", errG.item(), global_i)
            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\t'
                      % (epoch, num_epochs, i, len(train_loader), errD.item(), errG.item()))

            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if epoch % args.save_frequency == 0:
                with torch.no_grad():
                    fixed_noise = torch.randn(data.shape[0], 256, device=device)
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(fake.data.cpu().numpy())
                plt.imsave(os.path.join('./{}/'.format(test_save_dir ) + 'img{}.png'.format(datetime.now().strftime("%d.%m.%Y-%H:%M:%S"))),
                                 ((img_list[-1][0] + 1) / 2.0).transpose([1, 2, 0]), cmap='gray', interpolation="none")
            if writer is not None:
                writer.add_scalar(f"loss_G_epoch", np.sum(G_losses) / len(train_loader), epoch)
                writer.add_scalar(f"loss_D_epoch", np.sum(D_losses) / len(train_loader), epoch)
            iters += 1


if __name__ == '__main__':
    if args.mode == 'train':
        train()
    elif args.mode == 'test':
        test_net()
