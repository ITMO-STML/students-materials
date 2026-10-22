from pathlib import Path
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import re
from typing import Any, Callable, Optional, Union
import os
import urllib.request
import zipfile
import shutil
import glob

KAGGLE_URL = "https://www.kaggle.com/api/v1/datasets/download/nirmalsankalana/mini-product-image-and-text-dataset"

def download_data_if_needed(data_dir: Path = Path("data")) -> None:
    if (data_dir / "data.csv").exists() and list(data_dir.glob("*.jpg")):
        print("Data already exists. Skipping download.")
        return

    print("Downloading dataset from Kaggle...")
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / "dataset.zip"

    urllib.request.urlretrieve(KAGGLE_URL, zip_path)

    print("Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(data_dir)

    zip_path.unlink()
    
    for jpg in glob.glob(str(data_dir / "**" / "*.jpg"), recursive=True):
        dest = data_dir / Path(jpg).name
        if not dest.exists():
            shutil.move(jpg, dest)
            
    print("Data download complete and flattened.")

image_transform: transforms.Compose = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def extract_gender(display_name: str) -> int:

    if not isinstance(display_name, str):
        return 3
    name = display_name.lower()
    if "unisex" in name:
        return 2
    if re.search(r"\bwomen\b", name) or re.search(r"\bwoman\b", name):
        return 1
    if re.search(r"\bmen\b", name) or re.search(r"\bman\b", name):
        return 0
    return 3

class ProductDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_dir: Union[str, Path], transform: Optional[Callable[[Image.Image], torch.Tensor]] = None) -> None:
        self.df: pd.DataFrame = df.reset_index(drop=True)
        self.image_dir: Path = Path(image_dir)
        self.transform: Optional[Callable[[Image.Image], torch.Tensor]] = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row: pd.Series = self.df.iloc[idx]
        image: Image.Image = Image.open(self.image_dir / row["image"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        text: str = str(row["description"]) if pd.notna(row["description"]) else ""
        
        return {
            "image": image,
            "text": text,
            "label": torch.tensor(row["label"], dtype=torch.long),
            "gender": torch.tensor(row["gender_label"], dtype=torch.long),
        }