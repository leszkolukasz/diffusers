#!/usr/bin/env bash
#
# Download the official ILSVRC2012 ImageNet-1k split from HF, extract it to
# raw images, and convert it with EDM2's center-crop-dhariwal.
#
# The HF dataset is gated: accept its terms on
# https://huggingface.co/datasets/ILSVRC/imagenet-1k first.
#
# Usage:
#   scripts/download_imagenet64_hf.sh [validation|train]
#
# Default split: validation
#
# Optional environment variables:
#   HF_REPO          Dataset repo (default: ILSVRC/imagenet-1k)
#   HF_CACHE         Directory for downloaded parquet files
#                    (default: data/_imagenet_<split>_parquet)
#   RAW_DIR          Directory for extracted raw JPEGs
#                    (default: data/_imagenet_<split>_raw)
#   IMG64_VAL_DIR    Validation output (default: data/imagenet64_val)
#   IMG64_TRAIN_DIR  Train output (default: data/imagenet64_train)
#   PYTHON           Python executable (default: python)
#   KEEP_RAW         1 = keep parquet/raw after conversion (default: 1)

set -euo pipefail

SPLIT="${1:-${SPLIT:-validation}}"
if [[ "$SPLIT" != "validation" && "$SPLIT" != "train" ]]; then
    echo "Usage: $0 [validation|train]" >&2
    exit 2
fi

HF_REPO="${HF_REPO:-ILSVRC/imagenet-1k}"
HF_CACHE="${HF_CACHE:-data/_imagenet_${SPLIT}_parquet}"
RAW_DIR="${RAW_DIR:-data/_imagenet_${SPLIT}_raw}"
PYTHON="${PYTHON:-python}"
KEEP_RAW="${KEEP_RAW:-1}"

if [[ "$SPLIT" == "validation" ]]; then
    IMG64_OUT="${IMG64_VAL_DIR:-data/imagenet64_val}"
    EXPECTED=50000
else
    IMG64_OUT="${IMG64_TRAIN_DIR:-data/imagenet64_train}"
    EXPECTED=1281167
fi

if [[ -e "$RAW_DIR" && -n "$(find "$RAW_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "Raw directory already exists and is not empty: $RAW_DIR" >&2
    echo "Remove it or set RAW_DIR to a new path." >&2
    exit 2
fi
if [[ -d "$IMG64_OUT" && -n "$(find "$IMG64_OUT" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "Output directory already exists and is not empty: $IMG64_OUT" >&2
    echo "Remove it or set the output directory variable to a new path." >&2
    exit 2
fi

mkdir -p "$HF_CACHE" "$RAW_DIR"

echo "Dataset: $HF_REPO"
echo "Split:   $SPLIT"
echo "Parquet: $HF_CACHE"
echo "Raw:     $RAW_DIR"
echo "Output:  $IMG64_OUT"

uv run --with pyarrow --with pillow python - \
    "$HF_REPO" "$SPLIT" "$HF_CACHE" "$RAW_DIR" "$EXPECTED" <<'PY'
from io import BytesIO
from pathlib import Path
import sys

import pyarrow.parquet as pq
from huggingface_hub import snapshot_download
from PIL import Image

repo_id, split, cache_dir, raw_dir, expected = sys.argv[1:]
cache_dir = Path(cache_dir)
raw_dir = Path(raw_dir)
expected = int(expected)

print(f"Downloading {split} parquet files...")
snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    allow_patterns=[f"data/{split}-*.parquet"],
    local_dir=cache_dir,
)

files = sorted(cache_dir.glob(f"data/{split}-*.parquet"))
if not files:
    raise SystemExit(f"No {split} parquet files found under {cache_dir}")

total = sum(pq.ParquetFile(path).metadata.num_rows for path in files)
print(f"{split.capitalize()} parquet files: {len(files)}, images: {total}")
if total != expected:
    print(f"WARNING: expected {expected} images, parquet contains {total}")

written = 0
for path in files:
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=256, columns=["image", "label"]):
        images = batch.column("image").to_pylist()
        labels = batch.column("label").to_pylist()
        for image_record, label in zip(images, labels):
            data = image_record["bytes"]
            # Keep the original encoded bytes; use PIL only to choose extension.
            fmt = Image.open(BytesIO(data)).format
            ext = "jpg" if fmt == "JPEG" else fmt.lower()
            class_dir = raw_dir / f"{int(label):04d}"
            class_dir.mkdir(parents=True, exist_ok=True)
            (class_dir / f"{written:08d}.{ext}").write_bytes(data)
            written += 1
            if written % 1000 == 0:
                print(f"  extracted {written}/{total}", flush=True)

print(f"Extracted {written} images to {raw_dir}")
PY

echo "Converting with EDM2 center-crop-dhariwal..."
EDM2_REPO="${EDM2_REPO:-}" \
IMG64_DIR="$IMG64_OUT" \
PYTHON="$PYTHON" \
    scripts/prepare_edm2_img64.sh "$RAW_DIR" "$SPLIT"

if [[ "$KEEP_RAW" != "1" ]]; then
    rm -rf "$HF_CACHE" "$RAW_DIR"
fi

echo
echo "Done."
echo "$SPLIT ImageFolder: $IMG64_OUT"
