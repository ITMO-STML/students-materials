import torch
from abc_encoder import Encoder
from PIL import Image
from torch import nn
from transformers import AutoImageProcessor, AutoModel


class VideoEncoder(nn.Module, Encoder):
    def __init__(self, weight_path: str = "facebook/dinov3-convnext-tiny-pretrain-lvd1689m") -> None:
        super().__init__()
        self.processor = AutoImageProcessor.from_pretrained(weight_path)
        self.model = AutoModel.from_pretrained(
            weight_path,
            device_map="auto",
        )

    def forward(self, images: list[Image.Image]) -> torch.Tensor:
        pass

    def encode(self, images: list[Image.Image]) -> torch.Tensor:
        """Encoder for visual features

        Args:
            images (list[Image.Image]) - list of images

        Returns:
            torch.Tensor: embedding
        """
        return self(images)
