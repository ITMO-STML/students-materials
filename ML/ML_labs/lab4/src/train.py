import csv
import sys
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam

from data import get_dataloaders
from diffusion import Diffusion
from model import UNet

from evaluate import run_evaluation
from report import build_report

OUTPUTS_DIR = Path("/app/outputs")
DATA_DIR = Path("/app/data")

EPOCHS = 12
BATCH_SIZE = 128
LR = 2e-4
T_DIFF = 1000
SEED = 42

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 

def train_ddpm(device: torch.device) -> None:
    train_loader, _ = get_dataloaders(batch_size=BATCH_SIZE)

    model = UNet().to(device)
    diffusion = Diffusion(T=T_DIFF, device=device)
    optimizer = Adam(model.parameters(), lr=LR)

    loss_fn = nn.MSELoss()
    log_path = OUTPUTS_DIR / "loss.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss"])

        for epoch in range(1, EPOCHS + 1):
            model.train()
            running = 0.0
            n_batches = 0

            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                t = torch.randint(0, T_DIFF, (x.size(0),), device=device)
                noise = torch.randn_like(x)
                x_t = diffusion.q_sample(x, t, noise) # зашумлённая картинка

                pred = model(x_t, t, y) # предсказанный шум
                loss = loss_fn(pred, noise)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running += loss.item()
                n_batches += 1

            epoch_loss = running / n_batches
            writer.writerow([epoch, f"{epoch_loss:.6f}"])
            print(f"Epoch {epoch:02d}/{EPOCHS} | loss={epoch_loss:.4f}")

    torch.save(model.state_dict(), OUTPUTS_DIR / "checkpoint.pt")

def main() -> None:

    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)

    train_ddpm(device)

    run_evaluation(device)
    build_report()

if __name__ == "__main__":
    main()