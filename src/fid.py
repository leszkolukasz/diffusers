import random
import shutil
import tempfile
from enum import Enum
from pathlib import Path
from typing import Any

import torch
from cleanfid import fid
from loguru import logger
from PIL import Image
from torchvision import datasets
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm

from src.common import get_device
from src.config import DatasetConfig, DatasetType
from src.config.presets import DATASET_CONFIGS
from src.dataset import AYSToyDataset, PatternDataset

VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


class FIDMetric(str, Enum):
    fid = "fid"
    clip_fid = "clip_fid"
    kid = "kid"
    all = "all"


class FIDEvaluator:
    def __init__(
        self,
        real_path: str | Path,
        gen_path: str | Path,
        *,
        image_size: int | None = None,
        device: torch.device | None = None,
        num_workers: int = 4,
        batch_size: int = 32,
        verbose: bool = True,
    ):
        self.real_path = Path(real_path)
        self.gen_path = Path(gen_path)
        self.resize_to = (image_size, image_size) if image_size is not None else None
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.verbose = verbose
        self.device = device or get_device()

    def evaluate(self, metric: FIDMetric | str = FIDMetric.fid) -> float:
        metric = FIDMetric(metric)
        if metric == FIDMetric.all:
            raise ValueError("Use evaluate_all() to compute every metric.")

        for path in (self.real_path, self.gen_path):
            if not path.is_dir():
                raise ValueError(f"Directory does not exist: {path}")

        real_files = self._get_image_files(self.real_path)
        gen_files = self._get_image_files(self.gen_path)

        if not real_files or not gen_files:
            raise ValueError("One of the directories contains no valid images.")

        logger.info(
            f"Using all images: {len(real_files)} real, {len(gen_files)} generated."
        )

        with (
            tempfile.TemporaryDirectory() as tmp_real,
            tempfile.TemporaryDirectory() as tmp_gen,
        ):
            real_dir = Path(tmp_real)
            gen_dir = Path(tmp_gen)
            self._prepare_dir(real_files, real_dir, self.resize_to)
            self._prepare_dir(gen_files, gen_dir, self.resize_to)

            return self._compute(metric, real_dir, gen_dir)

    def evaluate_all(self) -> dict[str, float]:
        return {
            metric.value: self.evaluate(metric)
            for metric in (
                FIDMetric.fid,
                FIDMetric.kid,
            )  # (FIDMetric.fid, FIDMetric.clip_fid, FIDMetric.kid)
        }

    def _get_image_files(self, path: Path) -> list[Path]:
        return sorted(
            file
            for file in path.rglob("*")
            if file.is_file() and file.suffix.lower() in VALID_EXTENSIONS
        )

    def _prepare_dir(
        self,
        files: list[Path],
        target_dir: Path,
        resize_to: tuple[int, int] | None,
    ) -> None:
        target_dir.mkdir(parents=True, exist_ok=True)
        for file in files:
            target = target_dir / file.name
            if resize_to is None:
                shutil.copy2(file, target)
                continue
            with Image.open(file) as image:
                image.convert("RGB").resize(resize_to, Image.Resampling.LANCZOS).save(
                    target
                )

    def _compute(self, metric: FIDMetric, real_dir: Path, gen_dir: Path) -> float:
        common = {
            "device": self.device,
            "num_workers": self.num_workers,
            "batch_size": self.batch_size,
            "verbose": self.verbose,
        }

        if metric == FIDMetric.fid:
            score = fid.compute_fid(str(real_dir), str(gen_dir), **common)
        elif metric == FIDMetric.clip_fid:
            score = fid.compute_fid(
                str(real_dir),
                str(gen_dir),
                model_name="clip_vit_b_32",
                **common,
            )
        elif metric == FIDMetric.kid:
            score = fid.compute_kid(str(real_dir), str(gen_dir), **common)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        return float(score)


def materialize_dataset_images(
    config: DatasetConfig,
    num_images: int,
    out_dir: str | Path,
) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if config.path is not None:
        if not Path(config.path).is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {config.path}")
        dataset: Any = datasets.ImageFolder(root=config.path, transform=None)
        count = min(num_images, len(dataset))

        indices = random.Random(42).sample(range(len(dataset)), count)
    else:
        kwargs = dict(config.kwargs or {})
        if config.split is not None:
            kwargs["split"] = config.split
        dataset_class: Any = config.dataset_class
        dataset = dataset_class(
            root="./data",
            download=True,
            transform=None,
            **kwargs,
        )
        count = min(num_images, len(dataset))
        indices = list(range(count))

    for out_index, index in enumerate(tqdm(indices, desc="Materializing real images")):
        image, _ = dataset[index]
        if not isinstance(image, Image.Image):
            image = to_pil_image(image)
        image.convert("RGB").save(out / f"{out_index:06d}.png")

    return out


def materialize_real_images(
    dataset: DatasetType,
    num_images: int,
    *,
    cache_dir: str | Path = "data/fid_real",
) -> Path:
    config = DATASET_CONFIGS[dataset]

    if config.dataset_class in (AYSToyDataset, PatternDataset):
        raise ValueError(f"Dataset '{dataset.value}' is not an image dataset")

    out_dir = Path(cache_dir) / dataset.value
    existing = (
        sorted(
            file
            for file in out_dir.iterdir()
            if file.is_file() and file.suffix.lower() in VALID_EXTENSIONS
        )
        if out_dir.is_dir()
        else []
    )

    if len(existing) >= num_images:
        logger.info(f"Reusing cached real images from {out_dir}.")
        return out_dir

    if out_dir.exists():
        shutil.rmtree(out_dir)

    logger.info(
        f"Materializing {num_images} real images from '{dataset.value}' into {out_dir}."
    )
    return materialize_dataset_images(config, num_images, out_dir)
