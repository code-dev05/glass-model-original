from glass import GLASS
from anomalib.data import MVTecAD
from torchvision import transforms
from perlin import perlin_mask
import numpy as np
import torch
from torch import nn
import glob
from PIL import Image

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class Tranform:
    def __init__(self,
                resize=288,
                imagesize=288,
                rotate_degrees=0,
                translate=0,
                brightness_factor=0,
                contrast_factor=0,
                saturation_factor=0,
                gray_p=0,
                h_flip_p=0,
                v_flip_p=0,
                distribution=0,
                mean=0.5,
                std=0.1,
                fg=0,
                rand_aug=1,
                downsampling=8,
                scale=0,):
        self.resize = resize
        self.imagesize = imagesize
        self.rotate_degrees = rotate_degrees
        self.translate = translate
        self.brightness_factor = brightness_factor
        self.contrast_factor = contrast_factor
        self.saturation_factor = saturation_factor
        self.gray_p = gray_p
        self.h_flip_p = h_flip_p
        self.v_flip_p = v_flip_p
        self.distribution = distribution
        self.mean = mean
        self.std = std
        self.fg = fg
        self.rand_aug = rand_aug 
        self.downsampling = downsampling
        self.scale = scale

        transform_img = [
            transforms.Resize(resize),
            transforms.ColorJitter(brightness_factor, contrast_factor, saturation_factor),
            transforms.RandomHorizontalFlip(h_flip_p),
            transforms.RandomVerticalFlip(v_flip_p),
            transforms.RandomGrayscale(gray_p),
            transforms.RandomAffine(rotate_degrees,
                                    translate=(translate, translate),
                                    scale=(1.0 - scale, 1.0 + scale),
                                    interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
        self.transform_img = transforms.Compose(transform_img)

        transform_mask = [
            transforms.Resize(resize),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
        ]
        self.transform_mask = transforms.Compose(transform_mask)

        self.list_aug = [
            transforms.ColorJitter(contrast=(0.8, 1.2)),
            transforms.ColorJitter(brightness=(0.8, 1.2)),
            transforms.ColorJitter(saturation=(0.8, 1.2), hue=(-0.2, 0.2)),
            transforms.RandomHorizontalFlip(p=1),
            transforms.RandomVerticalFlip(p=1),
            transforms.RandomGrayscale(p=1),
            transforms.RandomAutocontrast(p=1),
            transforms.RandomEqualize(p=1),
            transforms.RandomAffine(degrees=(-45, 45)),
        ]

    def transform(self, img):
        if isinstance(img, torch.Tensor):
            # Convert tensor to NumPy then to PIL
            img = img.detach().cpu()
            if img.ndim == 3:
                img = img.permute(1, 2, 0)  # C, H, W -> H, W, C
            img = img.numpy()
            img = (img * 255).astype(np.uint8)  # assume in [0,1], scale to [0,255]
            img = Image.fromarray(img)

        aug_idx = np.random.choice(np.arange(len(self.list_aug)), 3, replace=False)
        transform_aug = [
            transforms.Resize(self.resize),
            self.list_aug[aug_idx[0]],
            self.list_aug[aug_idx[1]],
            self.list_aug[aug_idx[2]],
            transforms.CenterCrop(self.imagesize),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]

        img = self.transform_img(img)
        mask_fg = mask_s = aug_image = torch.tensor([1])
        aug = Image.open(np.random.choice(anomaly_source_paths)).convert("RGB")
        if self.rand_aug:
            transform_aug = transforms.Compose(transform_aug)
            aug = transform_aug(aug)
        mask_all = perlin_mask(img.shape, self.imagesize // self.downsampling, 0, 6, mask_fg, 1)
        mask_s = torch.from_numpy(mask_all[0])
        mask_l = torch.from_numpy(mask_all[1])
        beta = np.random.normal(loc=self.mean, scale=self.std)
        beta = np.clip(beta, .2, .8)
        aug_image = img * (1 - mask_l) + (1 - beta) * aug * mask_l + beta * img * mask_l
        
        return {
            "image": img,
            "aug": aug_image,
            "mask_s": mask_s,
        }

# have to put seed for numpy
model = GLASS()
model.load((3, 224, 224))
t = Tranform()
datamodule = MVTecAD(root="../datasets/MVTecAD", category="bottle", num_workers=4, train_batch_size=8, seed=42)
datamodule.setup()
anomaly_source_path = "../datasets/dtd/images"
anomaly_source_paths = sorted(1 * glob.glob(anomaly_source_path + "/*/*.jpg"))
train_data = []
for batch in datamodule.train_dataloader():
    tranformed = []
    b, _, height, width = batch.image.shape
    for i in range(b):
        out = t.transform(batch.image[i])
        tranformed.append(out)
    image = {
        "image": torch.stack([x["image"] for x in tranformed]),
        "aug": torch.stack([x["aug"] for x in tranformed]),
        "mask_s": torch.stack([x["mask_s"] for x in tranformed]),
    }
    train_data.append(image)
model.trainer(train_data)
