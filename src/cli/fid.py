from pathlib import Path

import typer
from cleanfid import fid as cleanfid
from loguru import logger

from src.common import get_device, safe_filename_part
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
        FIDMetric.all,
        "--metric",
        help="Metric to compute.",
    ),
    precompute: bool = typer.Option(
        False,
        "--precompute",
        help="Precompute clean-fid reference statistics before evaluating.",
    ),
    stats_name: str | None = typer.Option(
        None,
        "--stats-name",
        help="Name for clean-fid custom statistics (default: dataset/path name).",
    ),
    precompute_num: int | None = typer.Option(
        None,
        "--precompute-num",
        help="Limit precomputed reference images (default: all).",
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

    if precompute:
        if metric == FIDMetric.all:
            metric = FIDMetric.fid
            logger.info("Precomputed statistics support FID; computing FID only.")
        if metric != FIDMetric.fid:
            logger.error("--precompute supports only --metric fid.")
            raise typer.Exit(code=1)
        if dataset is not None and real_path is None:
            logger.warning(
                "Precompute uses materialized images; pass --real-path for the full dataset."
            )
        if precompute_num is None:
            logger.warning(
                "Precomputing all reference images can use a lot of RAM; "
                "use --precompute-num to limit it."
            )

        name = stats_name or safe_filename_part(
            dataset.value if dataset is not None else real.name
        )
        if cleanfid.test_stats_exists(name, "clean"):
            logger.info(f"Using existing clean-fid stats: {name}")
        else:
            logger.info(f"Precomputing clean-fid stats: {name}")
            cleanfid.make_custom_stats(
                name,
                str(real),
                num=precompute_num,
                mode="clean",
                num_workers=num_workers,
                batch_size=batch_size,
                device=get_device(),
            )

        score = cleanfid.compute_fid(
            str(gen_path),
            dataset_name=name,
            dataset_split="custom",
            mode="clean",
            device=get_device(),
            num_workers=num_workers,
            batch_size=batch_size,
        )
        logger.success(f"fid: {score:.8g}")
        return

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
        device=get_device(),
        num_workers=num_workers,
        batch_size=batch_size,
    )

    if metric == FIDMetric.all:
        for name, score in evaluator.evaluate_all().items():
            logger.success(f"{name}: {score:.8g}")
        return

    score = evaluator.evaluate(metric)
    logger.success(f"{metric.value}: {score:.8g}")
