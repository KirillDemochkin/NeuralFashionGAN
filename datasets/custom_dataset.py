import numpy as np
import cv2
import os
from os.path import join
import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, root, num_classes=13, transform=None, return_masked_image=False):
        super().__init__()

        self.image_folder = join(root, 'image')
        self.annos_folder = join(root, 'annos')
        self.length = len(os.listdir(self.image_folder))
        self.num_classes = num_classes
        self.transform = transform
        self.return_masked_image = return_masked_image

    def __getitem__(self, idx):
        # name = f'{idx + 1:06d}'
        name = str(idx + 1).zfill(6)

        image_path = join(self.image_folder, name + '.jpg')
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 127.5 - 1.0

        mask_path = join(self.annos_folder, name + '.png')
        mask_raw = cv2.cvtColor(cv2.imread(mask_path), cv2.COLOR_BGR2RGB)[:, :, 0]

        if name == '000001':
            mask = self._remake_mask(mask_raw, 1) # short sleeve top

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            image *= 2
            image -= 1
            full_mask = augmented['mask'].permute(2, 0, 1)

        if self.return_masked_image:
            masked_image = image.clone()
            masked_image[:, full_mask.sum(dim=0) > 0] = 1.
            loss_mask = torch.ones_like(masked_image)
            loss_mask[:, full_mask.sum(dim=0) > 0] = 0.
            return image, full_mask, masked_image, loss_mask
        else:
            return image, full_mask

    def _remake_mask(self, mask_raw, label):
        width, height = mask_raw.shape[1], mask_raw.shape[0]
        new_mask = np.zeros((self.num_classes, mask_raw.shape[0], mask_raw.shape[1]))
        new_mask[label - 1][mask_raw > 0] = 1
        return np.transpose(new_mask, [1, 2, 0])

    def __len__(self):
        return self.length