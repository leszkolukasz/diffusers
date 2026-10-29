import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed
from PIL import Image
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision import datasets, transforms

from src.config import DatasetConfig
from src.distributed import get_rank, get_world_size, is_distributed

N_SAMPLES = 10000


def unnormalize(img: torch.Tensor) -> torch.Tensor:
    img = (img + 1.0) / 2.0

    return torch.clamp(img, 0.0, 1.0)


def get_dataloader(
    batch_size: int,
    dataset_config: DatasetConfig,
    shuffle=True,
    num_workers=1,
):
    is_toy_dataset = dataset_config.dataset_class in [AYSToyDataset, PatternDataset]

    if is_toy_dataset:
        transform = None
    else:
        transform_steps = [
            transforms.Resize(dataset_config.img_size),
            transforms.CenterCrop(dataset_config.img_size),
        ]
        if dataset_config.horizontal_flip:
            transform_steps.append(transforms.RandomHorizontalFlip(p=0.5))
        transform_steps.extend(
            [
                transforms.ToTensor(),
                transforms.Lambda(
                    lambda x: (
                        x.repeat(dataset_config.channels, 1, 1)
                        if x.shape[0] == 1 and dataset_config.channels > 1
                        else x
                    )
                ),
                transforms.Normalize(mean=(0.5,), std=(0.5,)),
            ]
        )
        transform = transforms.Compose(transform_steps)

    dataset_kwargs: dict[str, Any] = (
        deepcopy(dataset_config.kwargs) if dataset_config.kwargs is not None else {}
    )

    if is_toy_dataset:
        dataset_kwargs["n_channels"] = dataset_config.channels
        dataset_kwargs["img_size"] = dataset_config.img_size

    if dataset_config.split is not None:
        dataset_kwargs["split"] = dataset_config.split

    if is_distributed() and get_rank() != 0:
        torch.distributed.barrier()

    should_download = True
    if is_distributed():
        should_download = get_rank() == 0

    if dataset_config.path is None:
        dataset_kwargs["root"] = "./data"
        dataset_kwargs["transform"] = transform
        dataset_kwargs["download"] = should_download

        dataset = dataset_config.dataset_class(
            **dataset_kwargs,
        )
    else:
        dataset = datasets.ImageFolder(
            root=dataset_config.path,
            transform=transform,
        )

    if is_distributed() and get_rank() == 0:
        torch.distributed.barrier()

    sampler = None
    if is_distributed():
        sampler = DistributedSampler(
            dataset, num_replicas=get_world_size(), rank=get_rank(), shuffle=True
        )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        sampler=sampler,
    )

    return loader


class AYSToyDataset(Dataset):
    img_size: int
    n_channels: int
    num_samples: int
    c: float
    data: torch.Tensor

    def __init__(
        self,
        num_samples: int,
        img_size: int = 32,
        n_channels: int = 1,
        c: float = 0.5,
        seed: int = 42,
        **kwargs,
    ):
        super().__init__()
        self.img_size = img_size
        self.n_channels = n_channels
        self.num_samples = num_samples
        self.c = c

        gen = torch.Generator().manual_seed(seed)
        self.data = (
            torch.randn(num_samples, n_channels, img_size, img_size, generator=gen)
            * self.c
        )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        return self.data[index], torch.tensor(0)


def load_pattern_density(pattern_path: str | Path, invert: bool = True) -> np.ndarray:
    if not os.path.exists(pattern_path):
        raise FileNotFoundError(f"Pattern image not found: {pattern_path}")

    img = Image.open(pattern_path).convert("L")
    img_arr = np.array(img, dtype=np.float32)

    if invert:
        img_arr = 255.0 - img_arr

    img_arr[img_arr < 10.0] = 0.0

    total_density = img_arr.sum()
    if total_density <= 0:
        raise ValueError(
            f"Pattern image '{pattern_path}' has no dark pixels to sample from."
        )

    return img_arr / total_density


class PatternDataset(Dataset):
    pattern_path: str
    num_samples: int
    rng: np.random.Generator
    pdf: np.ndarray
    h: int
    w: int
    gaussian: bool
    gaussian_sigma: float

    def __init__(
        self,
        pattern_path: str,
        num_samples: int,
        invert: bool = True,
        seed: int = 42,
        gaussian: bool = False,
        gaussian_sigma: float = 0.5,
        **kwargs,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.pattern_path = str(pattern_path)
        self.gaussian = gaussian
        self.gaussian_sigma = float(gaussian_sigma)

        self.rng = np.random.default_rng(seed)

        density = load_pattern_density(self.pattern_path, invert=invert)
        self.h, self.w = density.shape
        self.pdf = density.flatten()

        self.data = self._generate_points()

    def _generate_points(self) -> torch.Tensor:
        indices = self.rng.choice(self.h * self.w, size=self.num_samples, p=self.pdf)

        y_discrete, x_discrete = np.unravel_index(indices, (self.h, self.w))

        if self.gaussian:
            # Each pixel is a Gaussian component centered at its pixel index.
            x_continuous = x_discrete.astype(np.float32) + self.rng.normal(
                0.0, self.gaussian_sigma, self.num_samples
            )
            y_continuous = y_discrete.astype(np.float32) + self.rng.normal(
                0.0, self.gaussian_sigma, self.num_samples
            )
        else:
            # Add Unif([-0.5, 0.5]) so that points are not centered on pixels.
            x_continuous = (
                x_discrete.astype(np.float32) + self.rng.random(self.num_samples) - 0.5
            )
            y_continuous = (
                y_discrete.astype(np.float32) + self.rng.random(self.num_samples) - 0.5
            )

        x_norm = (x_continuous / self.w) * 2.0 - 1.0
        y_norm = 1.0 - (y_continuous / self.h) * 2.0

        points = np.stack([x_norm, y_norm], axis=1)

        # View (2, 1, 1) is needed to work with convolutional layers
        return torch.tensor(points, dtype=torch.float32).view(self.num_samples, 2, 1, 1)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        return self.data[index], torch.tensor(0)
