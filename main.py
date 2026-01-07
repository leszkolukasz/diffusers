import os
import sys

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import Tensor
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from tqdm import tqdm

from src.schedule import LinearAlphaSchedule, LinearSigmaSchedule, SchedulePack, CosineAlphaSchedule, CosineSigmaSchedule
from src.denoiser import Denoiser, DiscreteDenoiser, EulerMaruyamaSDEDenoiser, EulerODEDenoiser, HeunODEDenoiser, HeunSDEDenoiser
from src.model import ErrorPredictorUNet, PersistableModule

BATCH_SIZE = 512
EPOCHS = 1000
MAX_T = 1000
NORM_MEAN = 0.5
NORM_STD = 0.5

DENOISER_CONFIGS = {
    "discrete": {
        "denoiser": DiscreteDenoiser,
    },
    "euler": {
        "denoiser": EulerODEDenoiser,
    },
    "heun": {
        "denoiser": HeunODEDenoiser,
    },
    "euler_maruyama": {
        "denoiser": EulerMaruyamaSDEDenoiser,
    },
    "heun_sde": {
        "denoiser": HeunSDEDenoiser,
    },
}
SCHEDULE_CONFIGS = {
    "linear": {
        "alpha_schedule": LinearAlphaSchedule,
        "sigma_schedule": LinearSigmaSchedule,
    },
    "cosine": {
        "alpha_schedule": CosineAlphaSchedule,
        "sigma_schedule": CosineSigmaSchedule,
    },
}

DENOISER_CONFIG_NAME = "discrete"
denoiser_config = DENOISER_CONFIGS[DENOISER_CONFIG_NAME]

SCHEDULE_CONFIG_NAME = "cosine"
schedule_config = SCHEDULE_CONFIGS[SCHEDULE_CONFIG_NAME]

unnormalize = transforms.Normalize((-NORM_MEAN / NORM_STD,), (1.0 / NORM_STD,))

def get_dataloader(batch_size=BATCH_SIZE, shuffle=True, num_workers=2):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((NORM_MEAN,), (NORM_STD,))
    ])

    data = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)

    return loader

# Input: x0 (batch_size, C, H, W), t (batch_size,)
def diffuse(x0: Tensor, t: Tensor, schedules: SchedulePack):
    noise = torch.randn_like(x0)
    alpha = schedules.alpha(t).view(-1, 1, 1, 1)
    sigma = schedules.sigma(t).view(-1, 1, 1, 1)

    return alpha * x0 + sigma * noise, noise
    
def generate(denoiser: Denoiser, n_samples=16):
    x_t = torch.randn(n_samples, 1, 28, 28).cuda()

    # Skip t=T as it results in infinite values
    # SKip t=1 as Heun has problems with it
    for t in reversed(range(2, MAX_T)):
        t_batch = torch.full((n_samples,), t, device=x_t.device, dtype=torch.long)
        x_t = denoiser.denoise(x_t, t_batch)

    x_t = unnormalize(x_t.cpu())

    return x_t

    
def train():
    trainloader = get_dataloader()

    schedules = SchedulePack(
        alpha_schedule=schedule_config["alpha_schedule"](MAX_T),
        sigma_schedule=schedule_config["sigma_schedule"](MAX_T),
    )
        
    model: PersistableModule = ErrorPredictorUNet().cuda()
    model.try_load()

    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    total_steps = EPOCHS * len(trainloader)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        pbar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for batch_idx, (X, _) in enumerate(pbar):
            optimizer.zero_grad()

            X = X.cuda()
            t = torch.randint(1, MAX_T+1, (X.size(0),), device=X.device)

            X_noisy, noise = diffuse(X, t, schedules)
            err_pred = model(X_noisy, timestep=t)

            loss = criterion(err_pred, noise)
            total_loss += loss.item()
            pbar.set_postfix({"loss": total_loss / (batch_idx + 1)})

            loss.backward()
            optimizer.step()
            lr_scheduler.step()

        model.save()

    model.save()

def test():
    if not os.path.exists(f"generated_{DENOISER_CONFIG_NAME}_{SCHEDULE_CONFIG_NAME}"):
        os.makedirs(f"generated_{DENOISER_CONFIG_NAME}_{SCHEDULE_CONFIG_NAME}")

    model: PersistableModule = ErrorPredictorUNet().cuda()
    model.load()
    model.eval()

    schedules = SchedulePack(
        alpha_schedule=schedule_config["alpha_schedule"](MAX_T),
        sigma_schedule=schedule_config["sigma_schedule"](MAX_T),
    )

    denoiser: Denoiser = denoiser_config["denoiser"](
        model=model,
        max_t=MAX_T,
        schedules=schedules,
    )

    n_samples = 16
    generated = generate(denoiser, n_samples=n_samples)

    for i in range(n_samples):
        img = generated[i]
        torchvision.utils.save_image(img, f"generated_{DENOISER_CONFIG_NAME}_{SCHEDULE_CONFIG_NAME}/{i+1}.png")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test()
    else:
        train()