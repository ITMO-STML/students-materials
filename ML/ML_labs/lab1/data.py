from pathlib import Path
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


image_transform = transforms.Compose([   # препроцессинг
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


class ProductDataset(Dataset):
    def __init__(self, df, image_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(self.image_dir / row["image"]).convert("RGB")  # ResNet ждёт 3 канала
        if self.transform:
            image = self.transform(image)
        text = str(row["description"]) if pd.notna(row["description"]) else ""
        return {
            "image": image,
            "text": text,
            "label": torch.tensor(row["label"], dtype=torch.long),
        }