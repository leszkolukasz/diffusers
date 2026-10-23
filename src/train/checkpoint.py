import math
from pathlib import Path


def snapshot_steps(total_steps: int, percent: float) -> list[int]:
    if total_steps < 0:
        raise ValueError("total_steps must be non-negative")
    if not 0 <= percent <= 100:
        raise ValueError("percent must be in [0, 100]")

    if total_steps == 0 or percent == 0:
        return []

    steps: list[int] = []
    index = 1
    while True:
        threshold = math.ceil(total_steps * index * percent / 100)
        if threshold >= total_steps:
            break
        if not steps or threshold != steps[-1]:
            steps.append(threshold)
        index += 1

    return steps


def snapshot_stem(file_name: str, *, epoch: int, step: int) -> str:
    return f"{Path(file_name).stem}_epoch_{epoch:03d}_step_{step:08d}"


def resolve_snapshot_paths(path: str | Path) -> tuple[str, str, str]:
    snapshot_path = Path(path)
    stem = snapshot_path.stem
    for prefix in ("raw_", "trainer_"):
        if stem.startswith(prefix):
            stem = stem[len(prefix) :]
            break

    parent = snapshot_path.parent
    if parent == Path("."):
        parent = Path("models")

    return (
        str(parent / f"raw_{stem}.pth"),
        str(parent / f"trainer_{stem}.pth"),
        stem,
    )
