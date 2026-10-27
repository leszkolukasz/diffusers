#!/usr/bin/env bash
#
#SBATCH --job-name=prosem-generate
#SBATCH --account=g107-2933
#SBATCH --qos=normal
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=volta16G
#SBATCH --output=generate_output.txt

export EDM2_CACHE_DIR="${EDM2_CACHE_DIR:-$HOME/.cache/edm2}"
mkdir -p "$EDM2_CACHE_DIR"

echo "EDM2_CACHE_DIR=$EDM2_CACHE_DIR"

cd "$HOME/diffusers"

apptainer exec ./scripts/container.sif uv sync

srun --cpu-bind=none \
    apptainer exec --nv ./scripts/container.sif \
    uv run --no-sync main.py generate "$@"
