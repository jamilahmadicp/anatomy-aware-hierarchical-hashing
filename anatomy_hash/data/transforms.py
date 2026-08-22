from __future__ import annotations
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transform(image_size=224, train=False, mean=None, std=None):
    """Build a pickle-safe torchvision transform pipeline.

    Important for Windows: DataLoader workers use the ``spawn`` start method,
    so locally defined lambdas cannot be pickled.  ``Grayscale(3)`` performs
    the required grayscale-to-3-channel conversion without a lambda and works
    with ``num_workers > 0`` on Windows.
    """
    mean = mean or IMAGENET_MEAN
    std = std or IMAGENET_STD
    ops = [
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((image_size, image_size)),
    ]
    if train:
        ops += [
            transforms.RandomAffine(degrees=7, translate=(0.03, 0.03), scale=(0.95, 1.05)),
            transforms.ColorJitter(brightness=0.08, contrast=0.08),
        ]
    ops += [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
    return transforms.Compose(ops)
