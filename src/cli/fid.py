from pathlib import Path

import typer
from loguru import logger

from src.common import get_device
from src.config import DatasetType
from src.config.presets import DATASET_CONFIGS
from src.fid import FIDEvaluator, FIDMetric, materialize_real_images

app = typer.Typer(help="Compute FID, CLIP-FID or KID")


@app.callback(invoke_without_command=True)
def fid(
    gen_path: str = typer.Option(
        ...,
        "--gen-path",
        help="Directory with generated images.",
    ),
    dataset: DatasetType | None = typer.Option(
        None,
        "--dataset",
        help="Dataset used to materialize real images.",
    ),
    real_path: str | None = typer.Option(
        None,
        "--real-path",
        help="Use this directory instead of materializing the dataset.",
    ),
    metric: FIDMetric = typer.Option(
        FIDMetric.fid,
        "--metric",
        help="Metric to compute.",
    ),
    num_real: int = typer.Option(
        1000,
        "--num-real",
        help="Number of real images to materialize from the dataset.",
    ),
    image_size: int | None = typer.Option(
        None,
        "--image-size",
        help="Resize both sides to this size. Defaults to the dataset image size.",
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        help="Seed used when balancing real and generated images.",
    ),
    num_workers: int = typer.Option(
        4,
        "--num-workers",
        help="Data loading workers passed to clean-fid.",
    ),
    batch_size: int = typer.Option(
        32,
        "--batch-size",
        help="Batch size passed to clean-fid.",
    ),
):
    if real_path is None and dataset is None:
        logger.error("Provide either --dataset or --real-path.")
        raise typer.Exit(code=1)

    if real_path is not None:
        real = Path(real_path)
    else:
        assert dataset is not None
        real = materialize_real_images(dataset, num_real)

    if image_size is None and dataset is not None:
        image_size = DATASET_CONFIGS[dataset].img_size

    logger.info(f"Real images: {real}")
    logger.info(f"Generated images: {gen_path}")
    logger.info(f"Metric: {metric.value}")

    if image_size is not None:
        logger.info(f"Resizing both sides to {image_size}x{image_size}")

    evaluator = FIDEvaluator(
        real_path=real,
        gen_path=gen_path,
        image_size=image_size,
        seed=seed,
        device=get_device(),
        num_workers=num_workers,
        batch_size=batch_size,
    )

    if metric == FIDMetric.all:
        for name, score in evaluator.evaluate_all().items():
            logger.success(f"{name}: {score:.4f}")
        return

    score = evaluator.evaluate(metric)
    logger.success(f"{metric.value}: {score:.4f}")
