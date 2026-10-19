import os
from copy import deepcopy
from typing import Any

import torch
import torch.distributed
from PIL import Image
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms
from torchvision.datasets import VisionDataset

from src.config import DatasetConfig
from src.distributed import get_rank, get_world_size, is_distributed


def unnormalize(img: torch.Tensor) -> torch.Tensor:
    img = (img + 1.0) / 2.0

    return torch.clamp(img, 0.0, 1.0)


def get_dataloader(
    batch_size: int,
    dataset_config: DatasetConfig,
    shuffle=True,
    num_workers=1,
):
    transform = transforms.Compose(
        [
            transforms.Resize(dataset_config.img_size),
            transforms.CenterCrop(dataset_config.img_size),
            transforms.RandomHorizontalFlip(p=0.5),
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

    dataset_kwargs: dict[str, Any] = (
        deepcopy(dataset_config.kwargs) if dataset_config.kwargs is not None else {}
    )

    # Auto-generated datasets use those kwargs
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
        dataset = dataset_config.dataset_class(
            root="./data",
            download=should_download,  # ty: ignore
            transform=transform,
            **dataset_kwargs,
        )
    else:
        # This is mostly hack for CelebA, as Dataset.CelebA is stored on
        # cloud where download fails. So we need to download it manually.
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


class AYSToyDataset(VisionDataset):
    path: str

    def __init__(
        self,
        root: str,
        img_size: int,
        n_channels: int = 1,
        c: float = 0.5,
        download: bool = True,
        num_samples: int = 1000,
        transform=None,
    ):
        super().__init__(root=root, transform=transform)
        self.img_size = img_size
        self.c = c
        self.n_channels = n_channels
        self.num_samples = num_samples
        self.path = os.path.join(root, "ays_toy")

        if download:
            os.makedirs(self.path, exist_ok=True)
            self.generate_data()

        self.image_paths = self.get_image_paths()

    def generate_data(self):
        for i in range(self.num_samples):
            img = torch.rand(self.n_channels, self.img_size, self.img_size) * self.c
            img_path = os.path.join(self.path, f"sample_{i:04d}.png")
            transforms.ToPILImage()(img).save(img_path)

    def get_image_paths(self):
        paths = []
        for img_file in sorted(os.listdir(self.path)):
            if img_file.endswith(".png"):
                paths.append(os.path.join(self.path, img_file))
        return paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]

        mode = "RGB" if self.n_channels == 3 else "L"
        img = Image.open(img_path).convert(mode)

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(0)  # Dummy label
