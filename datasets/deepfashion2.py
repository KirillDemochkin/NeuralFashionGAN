import os
import json
import numpy as np
import cv2
from os.path import join
import torch
from torch.utils.data import Dataset
from pycocotools import mask as maskUtils


def ann_to_mask(segm, h, w):
    rles = maskUtils.frPyObjects(segm, h, w)
    rle = maskUtils.merge(rles)
    m = maskUtils.decode(rle)
    return m


class DeepFashion2Dataset(Dataset):
    def __init__(self, root, num_classes=13, transform=None, return_masked_image=False):
        super().__init__()
        
        self.image_folder = join(root, 'image')
        self.annos_folder = join(root, 'annos')
        self.length = len(os.listdir(self.image_folder))
        self.num_classes = num_classes
        self.transform = transform
        self.return_masked_image = return_masked_image
        
    def __getitem__(self, idx):
        name = f'{idx+1:06d}'
        
        image_path = join(self.image_folder, name+'.jpg')
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 127.5 - 1.0

        width, height = image.shape[1], image.shape[0]
        
        ann_path = join(self.annos_folder, name+'.json')
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
    
    def __len__(self):
        return self.length

class DeepFashion2DatasetOC(Dataset):
    def __init__(self, root, num_classes=13, transform=None, expansion=40):
        super().__init__()
        
        self.image_folder = join(root, 'image')
        self.annos_folder = join(root, 'annos')
        self.length = len(os.listdir(self.image_folder))
        self.num_classes = num_classes
        self.transform = transform
        self.dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (expansion, expansion))
        
    def __getitem__(self, idx):
        name = f'{idx+1:06d}'
        
        image_path = join(self.image_folder, name+'.jpg')
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 127.5 - 1.0

        width, height = image.shape[1], image.shape[0]
        
        ann_path = join(self.annos_folder, name+'.json')
        with open(ann_path) as f:
            ann = json.load(f)
        
        full_mask = np.zeros((height, width, self.num_classes+1),
                             dtype=np.float32)
        for key, item in ann.items():
            if key in ['source', 'pair_id']:
                continue
            clid = item['category_id']
            mask = ann_to_mask(item['segmentation'], height, width)
            full_mask[mask > 0, clid - 1] = 1
            exp_mask = cv2.dilate(mask, self.dilation_kernel)
            full_mask[exp_mask > 0, self.num_classes] = 1
        
        if self.transform is not None:
            augmented = self.transform(image=image, mask=full_mask)
            image = augmented['image']
            image *= 2
            image -= 1
            full_mask = augmented['mask'].permute(2, 0, 1)
        
        return image, full_mask[:self.num_classes], full_mask[self.num_classes:]
    
    def __len__(self):
        return self.length