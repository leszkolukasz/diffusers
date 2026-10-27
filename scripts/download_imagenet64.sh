#!/usr/bin/env bash
set -euo pipefail

REPO_ID="${IMAGENET64_HF_REPO:-benjamin-paine/imagenet-1k-64x64}"
SPLIT="${1:-validation}"
OUTPUT="${2:-${IMAGENET64_PATH:-data/imagenet64}}"

if [[ "$SPLIT" != "validation" && "$SPLIT" != "train" && "$SPLIT" != "test" ]]; then
    echo "Usage: $0 [validation|train|test] [output-dir]" >&2
    exit 2
fi

if [[ -e "$OUTPUT" && -n "$(find "$OUTPUT" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "Output directory already exists and is not empty: $OUTPUT" >&2
    echo "Remove it first if you want to regenerate it." >&2
    exit 1
fi

mkdir -p "$OUTPUT"

echo "Repo:   $REPO_ID"
echo "Split:  $SPLIT"
echo "Output: $OUTPUT"

uv run --with pyarrow --with pillow --with huggingface-hub python - "$REPO_ID" "$SPLIT" "$OUTPUT" <<'PY'
from io import BytesIO
from pathlib import Path
import sys

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image
from tqdm import tqdm

repo_id, split, output = sys.argv[1], sys.argv[2], Path(sys.argv[3])

api = HfApi()
files = [
    name
    for name in api.list_repo_files(repo_id, repo_type="dataset")
    if name.startswith(f"data/{split}-") and name.endswith(".parquet")
]
if not files:
    raise SystemExit(f"No parquet files found for split '{split}' in {repo_id}")

print(f"Parquet files: {len(files)}")
paths = [
    hf_hub_download(
        repo_id=repo_id,
        filename=name,
        repo_type="dataset",
    )
    for name in files
]

total = sum(pq.ParquetFile(path).metadata.num_rows for path in paths)
written = 0

with tqdm(total=total, desc=f"Converting ImageNet-64 {split}") as progress:
    for path in paths:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=512, columns=["image", "label"]):
            images = batch.column("image").to_pylist()
            labels = batch.column("label").to_pylist()
            for image_record, label in zip(images, labels):
                class_dir = output / f"{int(label):04d}"
                class_dir.mkdir(parents=True, exist_ok=True)
                image = Image.open(BytesIO(image_record["bytes"])).convert("RGB")
                image.save(class_dir / f"{written:08d}.png")
                written += 1
            progress.update(len(images))

print(f"Saved {written} images to {output}")
PY

find "$OUTPUT" -type f | wc -l
du -sh "$OUTPUT"
