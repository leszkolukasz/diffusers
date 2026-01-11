#!/usr/bin/env bash
#
#SBATCH --job-name=prosem
#SBATCH --partition=common
#SBATCH --qos=ll438580_common
#SBATCH --time=60
#SBATCH --output=output.txt
#SBATCH --ntasks=4
#SBATCH --gpus=4

export MASTER_PORT=12340
export WORLD_SIZE=${SLURM_NPROCS}

echo "NODELIST="${SLURM_NODELIST}
echo "WORLD_SIZE="${SLURM_NPROCS}

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

source ~/venv/bin/activate

srun python3 main.py train