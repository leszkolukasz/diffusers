from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import typer
from loguru import logger
from matplotlib.colors import LogNorm

from src.dataset import load_pattern_density


def load_generated_samples(path: str | Path) -> torch.Tensor:
    if not Path(path).exists():
        raise FileNotFoundError(f"Samples file not found: {path}")

    state = torch.load(path, weights_only=False, map_location="cpu")
    samples = state.get("samples") if isinstance(state, dict) else state

    if not isinstance(samples, torch.Tensor):
        raise TypeError(f"File '{path}' does not contain a 'samples' tensor.")

    return samples


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


def pattern(
    samples_path: str = typer.Argument(
        ..., help="Path to the .pth with generated PatternDataset samples."
    ),
    pattern_path: str = typer.Option(
        "assets/spiral.png",
        "--pattern-path",
        help="Pattern image used for training.",
    ),
    output: str = typer.Option(
        "pattern_density.png", "--output", "-o", help="Output plot path."
    ),
    log_scale: bool = typer.Option(
        True,
        "--log-scale/--linear-scale",
        help="Use a logarithmic color scale.",
    ),
):
    try:
        samples = load_generated_samples(samples_path)
        points = validate_pattern_samples(samples)
        true_density = load_pattern_density(pattern_path)
    except (FileNotFoundError, ValueError) as exc:
        logger.error(str(exc))
        raise typer.Exit(code=1) from exc

    height, width = true_density.shape
    generated_density = compute_density_map(points, (height, width))

    # PatternDataset row 0 is the top of the image, while imshow(origin="lower")
    # puts row 0 at the bottom.
    true_density = np.flipud(true_density)

    vmax = max(float(true_density.max()), float(generated_density.max()))
    norm = LogNorm(vmin=max(vmax * 1e-4, 1e-12), vmax=vmax) if log_scale else None

    colormap = plt.get_cmap("magma").copy()
    colormap.set_bad("white")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=110)
    extent = [-1.0, 1.0, -1.0, 1.0]
    panels = (
        (true_density, "True pattern density"),
        (generated_density, f"Generated density (N={len(points)})"),
    )

    for axis, (density, title) in zip(axes, panels, strict=True):
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
