
""" 
Dataloaders and stuff
"""

from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import ImageFilter
from torchvision import transforms
import random


DATASET_HELPER = {
	'cifar10': datasets.CIFAR10,
	'cifar100': datasets.CIFAR100
}


class GaussianBlur:
    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, img):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
        return img


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
    "normalize": transforms.Normalize
}


def get_transform(config):
    """
    Generates a torchvision.transforms.Compose pipeline
    based on given configurations.
    """
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


def get_dataloader(config):

	name = config.get('name', None)
	root = config.get('root', './')
	train_transform = config.get('train_transform', None)
	val_transform = config.get('val_transform', None)
	assert (train_transform is not None) and (val_transform is not None), 'Some transforms were not found'
	assert name in DATASET_HELPER.keys(), f'name should be one of {list(DATASET_HELPER.keys())}'

	train_transform = get_transform(train_transform)
	val_transform = get_transform(val_transform)

	# Obtain datasets
	train_dset = DATASET_HELPER[name](root=root, train=True, transform=train_transform, download=True)
	val_dset = DATASET_HELPER[name](root=root, train=False, transform=val_transform, download=True)

	# Loaders
	train_loader = DataLoader(train_dset, batch_size=config['batch_size'], num_workers=4, shuffle=True)
	val_loader = DataLoader(val_dset, batch_size=config['batch_size'], num_workers=4, shuffle=False)
	return train_loader, val_loader