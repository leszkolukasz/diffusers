#!/usr/bin/env bash
#
#SBATCH --job-name=prosem-ays
#SBATCH --account=g107-2933
#SBATCH --qos=normal
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=volta16G
#SBATCH --output=ays_output.txt

JOB_KEY=${SLURM_JOB_ID:-$$}
TASK_KEY=${SLURM_ARRAY_TASK_ID:-0}
export MASTER_PORT=$((20000 + (JOB_KEY + TASK_KEY * 1000) % 40000))
export WORLD_SIZE=${SLURM_NPROCS:-${SLURM_NTASKS:-1}}

echo "NODELIST=${SLURM_NODELIST:-}"
echo "WORLD_SIZE=$WORLD_SIZE"

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR=$MASTER_ADDR"

cd "$HOME/diffusers"

apptainer exec ./scripts/container.sif uv sync

srun --cpu-bind=none \
    apptainer exec --nv ./scripts/container.sif \
    uv run main.py ays "$@"
