from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import typer
from loguru import logger
from matplotlib.colors import LogNorm
from PIL import Image

from src.config import DatasetType
from src.config.presets import DATASET_CONFIGS
from src.dataset import load_pattern_density


def load_generated_samples(
    path: str | Path,
) -> tuple[torch.Tensor, DatasetType | None]:
    if not Path(path).exists():
        raise FileNotFoundError(f"Samples file not found: {path}")

    state = torch.load(path, weights_only=False, map_location="cpu")
    samples = state.get("samples") if isinstance(state, dict) else state

    if not isinstance(samples, torch.Tensor):
        raise TypeError(f"File '{path}' does not contain a 'samples' tensor.")

    dataset: DatasetType | None = None
    dataset_value = state.get("dataset") if isinstance(state, dict) else None
    if dataset_value:
        try:
            dataset = DatasetType(dataset_value)
        except ValueError:
            logger.warning(f"Unknown dataset in samples file: {dataset_value}")

    return samples, dataset


def validate_pattern_samples(samples: torch.Tensor) -> np.ndarray:
    if samples.ndim != 4 or tuple(samples.shape[1:]) != (2, 1, 1):
        raise ValueError(
            f"Expected PatternDataset samples with shape (N, 2, 1, 1), got {tuple(samples.shape)}."
        )

    return samples.reshape(samples.shape[0], 2).numpy()


def compute_density_map(points: np.ndarray, bins: tuple[int, int]) -> np.ndarray:
    height, width = bins
    density, _, _ = np.histogram2d(
        points[:, 0],
        points[:, 1],
        bins=(width, height),
        range=[[-1.0, 1.0], [-1.0, 1.0]],
    )

    density = density.T
    total = density.sum()
    return density / total if total > 0 else density


def resample_density(density: np.ndarray, bins: int) -> np.ndarray:
    if density.shape == (bins, bins):
        return density

    image = Image.fromarray(density.astype(np.float32))
    resized = np.asarray(image.resize((bins, bins), Image.Resampling.BOX))
    total = resized.sum()
    return resized / total if total > 0 else resized


def make_norm(
    density: np.ndarray, log_scale: bool, shared_max: float | None
) -> LogNorm | None:
    if not log_scale:
        return None
    top = float(density.max()) if shared_max is None else shared_max
    top = max(top, 1e-12)
    return LogNorm(vmin=max(top * 1e-4, 1e-12), vmax=top)


def pattern_path_from_dataset(dataset: DatasetType | None) -> str:
    if dataset is not None:
        kwargs = DATASET_CONFIGS[dataset].kwargs or {}
        path = kwargs.get("pattern_path")
        if isinstance(path, str):
            return path
    return "assets/spiral.png"


def pattern(
    samples_path: str = typer.Argument(
        ..., help="Path to the .pth with generated PatternDataset samples."
    ),
    pattern_path: str | None = typer.Option(
        None,
        "--pattern-path",
        help="Pattern image. Defaults to the dataset config recorded in the samples file.",
    ),
    output: str = typer.Option(
        "pattern_density.png", "--output", "-o", help="Output plot path."
    ),
    log_scale: bool = typer.Option(
        True,
        "--log-scale/--linear-scale",
        help="Use a logarithmic color scale.",
    ),
    bins: int = typer.Option(
        128,
        "--bins",
        help="Density map resolution (bins per axis).",
    ),
    share_scale: bool = typer.Option(
        False,
        "--share-scale/--per-panel-scale",
        help="Use one color scale for both panels.",
    ),
):
    try:
        samples, dataset = load_generated_samples(samples_path)
        points = validate_pattern_samples(samples)
        resolved_pattern_path = pattern_path or pattern_path_from_dataset(dataset)
        true_density = load_pattern_density(resolved_pattern_path)
    except (FileNotFoundError, TypeError, ValueError) as exc:
        logger.error(str(exc))
        raise typer.Exit(code=1) from exc

    true_density = resample_density(true_density, bins)
    generated_density = compute_density_map(points, (bins, bins))

    # PatternDataset row 0 is the top of the image, while imshow(origin="lower")
    # puts row 0 at the bottom.
    true_density = np.flipud(true_density)

    shared_max = (
        max(float(true_density.max()), float(generated_density.max()))
        if share_scale
        else None
    )
    norms = (
        make_norm(true_density, log_scale, shared_max),
        make_norm(generated_density, log_scale, shared_max),
    )

    colormap = plt.get_cmap("magma").copy()
    colormap.set_bad("white")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=110)
    extent = [-1.0, 1.0, -1.0, 1.0]
    panels = (
        (true_density, "True pattern density", norms[0]),
        (
            generated_density,
            f"Generated density (N={len(points)})",
            norms[1],
        ),
    )

    for axis, (density, title, norm) in zip(axes, panels, strict=True):
        image = axis.imshow(
            density,
            origin="lower",
            extent=extent,
            cmap=colormap,
            norm=norm,
            interpolation="nearest",
        )
        axis.set_title(title)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_aspect("equal")
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)

    logger.success(f"Saved pattern density comparison to {output}")
