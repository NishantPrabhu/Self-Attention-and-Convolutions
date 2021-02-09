"""
Augmentation pipelines and functions to generate them.
Authors: Mukund Varma T, Nishant Prabhu
"""

# Dependencies
from PIL import ImageFilter
from torchvision import transforms
import numpy as np
import random
import torch


class GaussianBlur:
    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, img):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
        return img


class Cutout:
    def __init__(self, n_cuts=0, max_len=1):
        self.n_cuts = n_cuts
        self.max_len = max_len

    def __call__(self, img):
        h, w = img.shape[1:3]
        cut_len = random.randint(1, self.max_len)
        mask = np.ones((h, w), np.float32)

        for _ in range(self.n_cuts):
            x, y = random.randint(0, w), random.randint(0, h)
            x1 = np.clip(x - cut_len // 2, 0, w)
            x2 = np.clip(x + cut_len // 2, 0, w)
            y1 = np.clip(y - cut_len // 2, 0, h)
            y2 = np.clip(y + cut_len // 2, 0, h)
            mask[y1:y2, x1:x2] = 0

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        return img * mask

# Transformation helper
TRANSFORM_HELPER = {
    "gaussian_blur": GaussianBlur,
    "color_jitter": transforms.ColorJitter,
    "random_gray": transforms.RandomGrayscale,
    "random_crop": transforms.RandomCrop,
    "random_resized_crop": transforms.RandomResizedCrop,
    "center_crop": transforms.CenterCrop,
    "resize": transforms.Resize,
    "random_flip": transforms.RandomHorizontalFlip,
    "to_tensor": transforms.ToTensor,
    "normalize": transforms.Normalize,
    "cutout": Cutout,
}


def get_transform(config, db_norm):
    """
    Generates a torchvision.transforms.Compose pipeline
    based on given configurations.
    """
    if config["normalize"] is None:
        config["normalize"] = db_norm
    transform = []

    # Obtain transforms from config in sequence
    for key, value in config.items():
        if value is not None:
            p = value.pop("apply_prob", None)
            tr = TRANSFORM_HELPER[key](**value)
            if p is not None:
                tr = transforms.RandomApply([tr], p=p)
        else:
            tr = TRANSFORM_HELPER[key]()
        transform.append(tr)
    return transforms.Compose(transform)