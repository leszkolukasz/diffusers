#!/usr/bin/env bash
#
#SBATCH --job-name=prosem
#SBATCH --account=g107-2933
#SBATCH --qos=normal
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=volta32G
#SBATCH --output=output.txt

export MASTER_PORT=12340
export WORLD_SIZE=${SLURM_NPROCS}

echo "NODELIST="${SLURM_NODELIST}
echo "WORLD_SIZE="${SLURM_NPROCS}

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

export WANDB_API_KEY="..."

cd $HOME/diffusers
apptainer exec ./scripts/container.sif uv sync

srun apptainer exec --nv ./scripts/container.sif uv run main.py train "$@"