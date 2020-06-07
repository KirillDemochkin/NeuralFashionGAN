class CustomDataset(Dataset):
    def __init__(self, root, num_classes=13, transform=None, return_masked_image=False, noise=False):
        super().__init__()

        self.image_folder = join(root, 'image')
        self.annos_folder = join(root, 'annos')
        self.length = len(os.listdir(self.image_folder))
        self.num_classes = num_classes
        self.transform = transform
        self.return_masked_image = return_masked_image
        self.noise = noise

    def get_item(self, idx, average_color=None):
        # name = f'{idx + 1:06d}'
        name = str(idx + 1).zfill(6)

        image_path = join(self.image_folder, name + '.jpg')
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        print(np.max(image), np.min(image))
        image = image.astype(np.float32) / 255.
        print(np.max(image), np.min(image))

        if name == '000001': # Masha's photo
            mask_path = join(self.annos_folder, name + '.png')
            mask_raw = cv2.cvtColor(cv2.imread(mask_path), cv2.COLOR_BGR2RGB)[:, :, 0]
            full_mask = self._remake_mask(mask_raw, 1) # short sleeve top
        if name == '000002': # Kirill's photo
            mask_1_path = join(self.annos_folder, name + '_1.PNG')
            mask_2_path = join(self.annos_folder, name + '_2.JPG')
            mask_raw_1 = cv2.cvtColor(cv2.imread(mask_1_path), cv2.COLOR_BGR2RGB)[:, :, 0]
            mask_raw_2 = cv2.cvtColor(cv2.imread(mask_2_path), cv2.COLOR_BGR2RGB)[:, :, 0]
            full_mask_1 = self._remake_mask(mask_raw_1, 2) # long sleeve top
            full_mask_2 = self._remake_mask(mask_raw_2, 8) # trousers
            full_mask_1[full_mask_2 > 0] = full_mask_2[np.where(full_mask_2 > 0)]
            full_mask = full_mask_1

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

    def _remake_mask(self, mask_raw, label):
        width, height = mask_raw.shape[1], mask_raw.shape[0]
        new_mask = np.zeros((self.num_classes, mask_raw.shape[0], mask_raw.shape[1]))
        new_mask[label - 1][mask_raw > 0] = 1
        return np.transpose(new_mask, [1, 2, 0])

    def __len__(self):
        return self.length
