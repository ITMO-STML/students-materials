from pathlib import Path
from typing import Tuple

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

datasets.MNIST.mirrors = ["https://ossci-datasets.s3.amazonaws.com/mnist/"]

DATA_DIR = Path("/app/data")
IMAGE_SIZE = 28
NUM_CLASSES = 10


def get_dataloaders(batch_size: int = 128, num_workers: int = 2) -> Tuple[DataLoader, DataLoader]:

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)), # в [-1, 1]
    ])

    train_ds = datasets.MNIST(root=str(DATA_DIR), train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root=str(DATA_DIR), train=False, download=True, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader