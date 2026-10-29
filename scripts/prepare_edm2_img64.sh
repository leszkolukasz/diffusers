#!/usr/bin/env bash
#
# Convert raw ImageNet ILSVRC2012 data to EDM2 ImageNet-64 preprocessing:
# 64x64, center-crop-dhariwal.
#
# Usage:
#   scripts/prepare_edm2_img64.sh SOURCE [validation|train]
#
# Examples:
#   scripts/prepare_edm2_img64.sh /path/to/ILSVRC/Data/CLS-LOC/val validation
#   scripts/prepare_edm2_img64.sh /path/to/ILSVRC/Data/CLS-LOC/train train
#
# Optional environment variables:
#   EDM2_REPO       Path to the NVlabs/EDM2 checkout (auto-detected otherwise)
#   IMG64_DIR       Output ImageFolder
#                   (validation: data/imagenet64_val,
#                    train:      data/imagenet64_train)
#   IMG64_ZIP       Output EDM2 zip. Empty for validation by default;
#                   for train defaults to $EDM2_REPO/datasets/img64.zip.
#   PYTHON          Python executable (default: python)
#

set -euo pipefail

SOURCE="${1:-${IMAGENET_ROOT:-}}"
SPLIT="${2:-${SPLIT:-validation}}"

if [[ -z "$SOURCE" ]]; then
    echo "Usage: $0 SOURCE [validation|train]" >&2
    exit 2
fi
if [[ ! -d "$SOURCE" ]]; then
    echo "Source directory does not exist: $SOURCE" >&2
    exit 2
fi
if [[ "$SPLIT" != "validation" && "$SPLIT" != "train" ]]; then
    echo "Split must be 'validation' or 'train', got: $SPLIT" >&2
    exit 2
fi

PYTHON="${PYTHON:-python}"

if [[ -z "${EDM2_REPO:-}" ]]; then
    for candidate in "$HOME/Desktop/edm2" "$HOME/edm2" "$PWD/edm2"; do
        if [[ -f "$candidate/dataset_tool.py" ]]; then
            EDM2_REPO="$candidate"
            break
        fi
    done
fi
if [[ -z "${EDM2_REPO:-}" || ! -f "$EDM2_REPO/dataset_tool.py" ]]; then
    echo "Could not find the EDM2 repository. Set EDM2_REPO." >&2
    exit 2
fi

EDM2_REPO="$(cd "$EDM2_REPO" && pwd)"

if [[ "$SPLIT" == "validation" ]]; then
    IMG64_DIR="${IMG64_DIR:-${IMG64_VAL_DIR:-data/imagenet64_val}}"
    IMG64_ZIP="${IMG64_ZIP:-}"
    EXPECTED=50000
else
    IMG64_DIR="${IMG64_DIR:-${IMG64_TRAIN_DIR:-data/imagenet64_train}}"
    IMG64_ZIP="${IMG64_ZIP:-$EDM2_REPO/datasets/img64.zip}"
    EXPECTED=1281167
fi

if [[ -n "$IMG64_ZIP" && -e "$IMG64_ZIP" ]]; then
    echo "Output zip already exists: $IMG64_ZIP" >&2
    echo "Remove it or set IMG64_ZIP to a new path." >&2
    exit 2
fi
if [[ -d "$IMG64_DIR" && -n "$(find "$IMG64_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "Output directory already exists and is not empty: $IMG64_DIR" >&2
    echo "Remove it or set IMG64_DIR to a new path." >&2
    exit 2
fi

mkdir -p "$(dirname "$IMG64_DIR")"
mkdir -p "$IMG64_DIR"
if [[ -n "$IMG64_ZIP" ]]; then
    mkdir -p "$(dirname "$IMG64_ZIP")"
fi

echo "EDM2 repo:  $EDM2_REPO"
echo "Split:      $SPLIT"
echo "Source:     $SOURCE"
echo "ImageFolder: $IMG64_DIR"
if [[ -n "$IMG64_ZIP" ]]; then
    echo "EDM2 zip:   $IMG64_ZIP"
fi

if [[ -n "$IMG64_ZIP" ]]; then
    "$PYTHON" "$EDM2_REPO/dataset_tool.py" convert \
        --source="$SOURCE" \
        --dest="$IMG64_ZIP" \
        --resolution=64x64 \
        --transform=center-crop-dhariwal

    echo "Extracting ImageFolder..."
    unzip -q "$IMG64_ZIP" -d "$IMG64_DIR"
else
    "$PYTHON" "$EDM2_REPO/dataset_tool.py" convert \
        --source="$SOURCE" \
        --dest="$IMG64_DIR" \
        --resolution=64x64 \
        --transform=center-crop-dhariwal
fi

COUNT="$(find "$IMG64_DIR" -type f -name '*.png' | wc -l | tr -d ' ')"
echo "Extracted PNG files: $COUNT"
if [[ "$COUNT" -ne "$EXPECTED" ]]; then
    echo "WARNING: expected $EXPECTED images for $SPLIT, got $COUNT." >&2
fi

echo
echo "Done."
echo "ImageFolder: $IMG64_DIR"
if [[ -n "$IMG64_ZIP" ]]; then
    echo "EDM2 zip:   $IMG64_ZIP"
fi
