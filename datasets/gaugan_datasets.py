import numpy as np
import cv2
from PIL import Image
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset


class CocoDataset(Dataset):
    def __init__(self, images_dir, masks_dir, images_files, labels_num):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.labels_num = labels_num

        self.new_h = 256
        self.new_w = 256

        self.examples = []

        for file_name in images_files:
            image_id = file_name.split('.jpg')[0]
            image_path = self.images_dir + file_name
            mask_path = self.masks_dir + image_id + ".png"

            example = {}
            example["image_id"] = image_id
            example["image_path"] = image_path
            example["mask_path"] = mask_path
            self.examples.append(example)

        self.num_examples = len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]


        image_path = example["image_path"]
        image = Image.open(image_path)
        mask_path = example["mask_path"]
        mask = Image.open(mask_path)

        image = TF.center_crop(image, (self.new_w, self.new_h))
        mask = TF.center_crop(mask, (self.new_w, self.new_h))  # shape = (256, 256, 3)

        # image = cv2.resize(image, (self.new_w, self.new_h),
        #                    interpolation=cv2.INTER_NEAREST) # shape = (256, 256, 3)
        # mask = cv2.resize(mask, (self.new_w, self.new_h),
        #                   interpolation=cv2.INTER_NEAREST) # shape = (256, 256, 3)

        # normalize
        image = np.asarray(image)
        mask = np.asarray(mask)
        image = image / 127.5 - 1

        # convert numpy -> torch
        try:
            image = np.transpose(image, (2, 0, 1))  # shape = (3, 256, 256)
            image = image.astype(np.float32)
        except Exception as ex:
            print(ex)
            print(image.shape)
            return None

        # mask_image = mask[:, :, 0]  # shape = (256, 256)
        mask_image = mask
        mask = np.stack([mask_image == i for i in range(1, self.labels_num + 1)],
                        axis=2)  # shape = (256, 256, labels_num)
        mask = np.transpose(mask, (2, 0, 1)).astype(np.float32)  # shape = (labels_num, 256, 256)

        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)

        return (image, mask, mask_image)  # image, mask as input to NN, mask to visualize

    def __len__(self):
        return self.num_examples

    class ADEDataset(Dataset):
        def __init__(self, images_dir, masks_dir, images_files, labels_num):
            self.images_dir = images_dir
            self.masks_dir = masks_dir
            self.labels_num = labels_num

            self.new_h = 256
            self.new_w = 256

            self.examples = []

            for file_name in images_files:
                image_id = file_name.split('ADE_train_')[1].split('.jpg')[0]
                image_path = self.images_dir + file_name
                mask_path = self.masks_dir + 'ADE_train_' + image_id + ".png"

                example = {}
                example["image_id"] = image_id
                example["image_path"] = image_path
                example["mask_path"] = mask_path
                self.examples.append(example)

            self.num_examples = len(self.examples)

        def __getitem__(self, index):
            example = self.examples[index]

            image_path = example["image_path"]
            image = cv2.imread(image_path)

            mask_path = example["mask_path"]
            mask = cv2.imread(mask_path)

            image = cv2.resize(image, (self.new_w, self.new_h),
                               interpolation=cv2.INTER_NEAREST)  # shape = (256, 256, 3)
            mask = cv2.resize(mask, (self.new_w, self.new_h),
                              interpolation=cv2.INTER_NEAREST)  # shape = (256, 256, 3)

            # # normalize
            # image = image / 255.0
            #
            # # convert numpy -> torch
            # image = np.transpose(image, (2, 0, 1))  # shape = (3, 256, 256)
            # image = image.astype(np.float32)
            image = np.float32(image).transpose((2, 0, 1)) / 127.5 - 1.0

            mask_image = mask[:, :, 0]  # shape = (256, 256)
            mask = np.stack([mask_image == i for i in range(1, self.labels_num + 1)],
                            axis=2)  # shape = (256, 256, labels_num)
            mask = np.transpose(mask, (2, 0, 1)).astype(np.int)  # shape = (labels_num, 256, 256)

            image = torch.from_numpy(image)
            mask = torch.from_numpy(mask)

            return (image, mask, mask_image)  # image, mask as input to NN, mask to visualize

        def __len__(self):
            return self.num_examples
