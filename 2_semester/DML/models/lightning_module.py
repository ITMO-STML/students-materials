import pytorch_lightning as pl
import torch
import torchmetrics
from torch import nn, optim
from torch.nn import functional as F

from data import FeatureType

from .audio_encoder import ConformerAttentiveProbe
from .cross_attn import CrossAttnClassifier
from .dummy_net import DummyNet
from .early_mm import EalryClassifier, EalryClassifierv2
from .late_mm import LateClassifier
from .text_encoder import BertAttentiveProbe


class EmoClassifier(pl.LightningModule):
    def __init__(self, net_type: FeatureType, num_classes: int = 1, transfer_learning: bool = True, num_layers: int = 4, num_heads: int = 4) -> None:
        super().__init__()
        net_type = FeatureType(net_type)
        match net_type:
            case FeatureType.TFIDF:
                input_dim = 300
                self.model = DummyNet(input_dim, num_classes)
            case FeatureType.W2V:
                input_dim = 250
                self.model = DummyNet(input_dim, num_classes)
            case FeatureType.MEL:
                input_dim = 128
                self.model = DummyNet(input_dim, num_classes)
            case FeatureType.MFCC:
                input_dim = 40
                self.model = DummyNet(input_dim, num_classes)
            case FeatureType.CONFORMER:
                self.model = ConformerAttentiveProbe(num_classes, num_layers=num_layers, num_heads=num_heads)

                # just a preprocessor
                self.model.proc.eval()
                for param in self.model.proc.parameters():
                    param.requires_grad = False
                if transfer_learning:
                    self.model.enc.eval()
                    for param in self.model.enc.parameters():
                        param.requires_grad = False

            case FeatureType.BERT:
                self.model = BertAttentiveProbe(num_classes=num_classes, num_layers=num_layers, num_heads=num_heads)
                if transfer_learning:
                    self.model.enc.eval()
                    for param in self.model.enc.parameters():
                        param.requires_grad = False

            case FeatureType.EARLY_FUSION:
                self.model = EalryClassifier(num_classes=num_classes)
                self.model.proc.eval()
                if transfer_learning:
                    self.model.text_enc.eval()
                    for param in self.model.text_enc.parameters():
                        param.requires_grad = False

                    self.model.audio_enc.eval()
                    for param in self.model.audio_enc.parameters():
                        param.requires_grad = False

            case FeatureType.EARLY_FUSION_V2:
                self.model = EalryClassifierv2(num_classes=num_classes)
                self.model.proc.eval()
                if transfer_learning:
                    self.model.text_enc.eval()
                    for param in self.model.text_enc.parameters():
                        param.requires_grad = False

                    self.model.audio_enc.eval()
                    for param in self.model.audio_enc.parameters():
                        param.requires_grad = False

            case FeatureType.LATE_FUSION:
                self.model = LateClassifier(num_classes=num_classes)
                self.model.proc.eval()
                if transfer_learning:
                    self.model.text_enc.eval()
                    for param in self.model.text_enc.parameters():
                        param.requires_grad = False

                    self.model.audio_enc.eval()
                    for param in self.model.audio_enc.parameters():
                        param.requires_grad = False

            case FeatureType.CROSS_ATTN:
                self.model = CrossAttnClassifier(num_classes=num_classes)
                self.model.proc.eval()
                if transfer_learning:
                    self.model.text_enc.eval()
                    for param in self.model.text_enc.parameters():
                        param.requires_grad = False

                    self.model.audio_enc.eval()
                    for param in self.model.audio_enc.parameters():
                        param.requires_grad = False

            case _:
                raise NotImplementedError(f"No such faeture option {net_type}")

        self.metrics = nn.ModuleDict(
            {
                "_" + split: torchmetrics.MetricCollection(
                    {
                        "f1": torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average="macro"),
                        "acc": torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, average="macro"),
                        "precision": torchmetrics.Precision(task="multiclass", num_classes=num_classes, average="macro"),
                        "recall": torchmetrics.Recall(task="multiclass", num_classes=num_classes, average="macro"),
                        "roc-auc": torchmetrics.AUROC(task="multiclass", num_classes=num_classes, average="macro"),
                    },
                    prefix=split + "_",
                )
                for split in ["train", "val", "test"]
            }
        )

    def forward(self, features: torch.Tensor, length: list[int] | None, texts: list[str] | None = None) -> torch.Tensor:
        if texts is not None:
            return self.model(features, length, texts)
        if length is not None:
            return self.model(features, length)

        return self.model(features)

    def __shared_step(self, batch: tuple, step_type: str) -> tuple[torch.Tensor, torch.Tensor]:
        length = None
        texts = None

        if len(batch) == 4:
            texts, features, length, labels = batch
        elif len(batch) == 3:
            features, length, labels = batch
        else:
            features, labels = batch
        logits = self(features, length, texts)
        loss = F.cross_entropy(logits, labels)
        m_out = self.metrics["_" + step_type](logits, labels)

        self.log(f"{step_type}_loss", loss, prog_bar=True, on_step=step_type == "train")
        self.log_dict(m_out, on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        return self.__shared_step(batch, "train")

    def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        return self.__shared_step(batch, "val")

    def test_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        return self.__shared_step(batch, "test")

    def configure_optimizers(self) -> optim.AdamW:
        return optim.AdamW(self.model.parameters())
