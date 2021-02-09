"""
Image transforms, datasets and dataloaders
Authors: Mukund Varma T, Nishant Prabhu
"""
import random
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets

cifar10 = {
    "data": datasets.CIFAR10,
    "classes": 10,
    "norm": {"mean": [0.4914, 0.4822, 0.4465], "std": [0.2470, 0.2435, 0.2616]},
}

DATASET_HELPER = {"cifar10": cifar10}

def get_dataset(name, dataroot, split, transform):
    """
    Generates a dataset object with required images and/or labels
    transformed as specified.
    """
    base_class = DATASET_HELPER[name]["data"]

    # Image dataset class
    class ImageDataset(base_class):
        def __init__(self, root, transform, train, download=True):
            super().__init__(root=root, train=train, download=download)
            self.transform = transform

        def __getitem__(self, i):
            # Load image and target
            img, target = self.data[i], self.targets[i]
            img = Image.fromarray(img)

            # Perform transformations
            data = {}
            data["img"] = self.transform(img)
            data["target"] = target
            return data

    # Return dataset object
    return ImageDataset(root=dataroot, train=split == "train", transform=transform)

def get_dataloader(dataset, batch_size, num_workers=1, shuffle=False):
    """ Returns a DataLoader with specified configuration """
    
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)