"""Config composition and forward shape without pretrained weights."""

from pathlib import Path

import hydra
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from torch import nn
from transformers import Wav2Vec2Config, Wav2Vec2Model

from src.model.model import Wav2Vec2AASIST, Wav2Vec2Frontend

CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "configs")


def test_config_and_forward() -> None:
    """Composed settings drive logits and a scalar cross-entropy loss."""
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.3"):
        cfg = compose(config_name="main")

    assert cfg.model.wav2vec2.model_name == "facebook/wav2vec2-xls-r-300m"
    assert cfg.data.model_name == cfg.model.wav2vec2.model_name
    assert cfg.training_arguments.learning_rate == 1e-5
    assert cfg.training_arguments.lr_scheduler_type == "cosine"
    assert cfg.training_arguments.warmup_steps == 0.1
    assert list(cfg.training_arguments.report_to) == ["trackio", "codecarbon"]
    assert cfg.model.wav2vec2.requires_grad is True

    criterion: nn.Module = hydra.utils.instantiate(cfg.model.criterion)
    model = Wav2Vec2AASIST(
        wav2vec2=cfg.model.wav2vec2,
        aasist=cfg.model.aasist,
        criterion=criterion,
        wav2vec=Wav2Vec2Model(Wav2Vec2Config()),
    )
    model.train()
    assert model.frontend.model.training
    assert any(parameter.requires_grad for parameter in model.frontend.model.parameters())
    frozen = Wav2Vec2Frontend("unused", encoder=Wav2Vec2Model(Wav2Vec2Config()), requires_grad=False)
    frozen.train()
    assert frozen.model.training
    assert all(not parameter.requires_grad for parameter in frozen.model.parameters())
    model.eval()
    assert not model.frontend.model.training
    waveforms = torch.randn(2, 16000)
    labels = torch.tensor([0, 1])
    with torch.no_grad():
        outputs = model(waveforms, labels=labels)

    assert outputs["logits"].shape == (2, 2)
    assert outputs["loss"].ndim == 0
