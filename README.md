# Diffusers

## Setup

Project uses `uv` to manage python dependencies. To install dependencies, run:

```bash
uv sync
```

## Usage

Project contains three commands:
- `train`: Train a diffusion model.
- `generate`: Generate samples from a trained diffusion model.
- `ays`: Find optimal steps using Align Your Steps.

Each command has its own set of arguments. Use `--help` to see them.

## Examples

### MNIST generalized diffusion model

Train with:

```
uv run main.py train --dataset mnist --model unet --schedule linear --eta stochastic
```

This model can be used with both discrete:
```
uv run main.py generate \
    --model unet \
    --schedule linear \
    --sampling-schedule discrete \
    --eta stochastic \
    --equation generalized_discrete \
    --solver discrete \
    --model-path ./models/noise_predictor_unet_mnist.pth
```

and differential diffusion:
```
uv run main.py generate \
    --model unet \
    --schedule linear \
    --sampling-schedule linear \
    --eta stochastic \
    --equation generalized_differential \
    --solver heun \
    --model-path ./models/noise_predictor_unet_mnist.pth
```

### MNIST EDM diffusion model

Train with:

```
 uv run main.py train --dataset mnist --model edm --schedule edm 
```

Generate with:
```
uv run main.py generate \
    --model edm \
    --schedule edm \
    --sampling-schedule edm \
    --equation probability_flow \
    --solver heun \
    --model-path ./models/x0_predictor_edm_mnist.pth
```

### Huggingface DDPM model

Generate with:
```
uv run main.py generate \
    --model huggingface \
    --schedule ddpm \
    --sampling-schedule discrete \
    --eta ddpm \
    --equation generalized_discrete \
    --solver discrete \
    --model-id "1aurent/ddpm-mnist"
```

### Train on ICM

Enter ICM HPC and then Rysy.
Clone this repository then cd into `./scripts` and build container with:

```
apptainer build container.sif container.def
```

Fill `WANDB_API_KEY` in `./scripts/train_icm.sh` and then run:

```
sbatch ./scripts/train_icm.sh
```

You can pass additional arguments e.g.

```
sbatch ./scripts/train_icm.sh --num-epochs 10000
```

## Architecture

The project is fairly modular, one can swap out different components (e.g. signal/noise schedulers, solvers, etc.) by changing command arguments.
Many of the combinations will not work though, like using a DDPM noise scheduler with Euler solver.

`Schedulers` refers to noise ($\alpha$), signal ($\sigma$), eta ($\eta$) and noise-to-signal (NSR) schedulers. In most places, $\sigma$ denotes
noise scheduler and $\sigma_{EDM}$ denotes NSR. There is also `s` scheduler which is an alias for noise scheduler.

There is subset of schedulers called `Sampling schedulers`. This refers to AYS, Entropic, EDM and other time sampling schedulers.

`Equations` model actual mathematical equations for reverse diffusion.

`Solvers` implement various numerical methods for solving differential equations and there is also one for discrete equations.

`Models` are neural networks (mostly UNets). There are a few choices including basic UNet, EDM, EDM2 and model from huggingface.

Many places refer to variable `T`. This usually means that something is meant to operate on [0, T] time interval e.g. `solver_T`
denotes a range on which a solver is meant to operate (sampled time steps should be in [0, T]) and `predictor_T`
denotes that model was trained on range [0, T].