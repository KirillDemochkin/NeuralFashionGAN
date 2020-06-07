class StyleDataset(Dataset):
    def __init__(self, styles_dir, transform=None):
        super().__init__()

        self.styles_dir = styles_dir
        self.styles = os.listdir(styles_dir)
        self.transform = transform

    def get_image(self):

        # randomly select style
        style = random.choice(self.styles)
        print(style)

        # randomly select image of this style
        style_images_path = join(self.styles_dir, style)
        image_file = random.choice(os.listdir(style_images_path))
        image_path = join(style_images_path, image_file)
        # image_path = 'styles/000457.jpg'

        # read image
        image = Image.open(image_path)
        image = np.array(image)
        image = image.astype(np.float32) / 255.

        width, height = image.shape[1], image.shape[0]

        if self.transform is not None:
            augmented = self.transform(image=image)
            image = augmented['image']
        image *= 2
        image -= 1

        return image
