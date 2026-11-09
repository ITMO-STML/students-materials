"""Steering identity, fit, provenance isolation, and attack-score semantics."""

from pathlib import Path
from typing import override

import numpy as np
import pytest
import torch
from torch import Tensor, nn
from transformers import Wav2Vec2Config, Wav2Vec2Model

from src.steering.evaluation import attack_report, dev_threshold
from src.steering.inference import run_model
from src.steering.protocol import Trial, check_disjoint, read_trials, select_trials, speaker_partition
from src.steering.run import admissible, feature_direction
from src.steering.sae import TopKSAE, active_mean, fit_sae


def trial(speaker: str, utterance: str, attack: str, source: str = "-") -> Trial:
    """Build a minimal test record.

    Args:
        speaker: Speaker ID.
        utterance: Recording ID.
        attack: Attack ID or bonafide.
        source: Optional source recording ID.

    Returns:
        Metadata record.
    """
    return Trial(speaker, utterance, attack, int(attack != "bonafide"), "-", source, "/unused")


def test_sae_identity_sparsity_and_residual() -> None:
    """Zero steering is exact; nonzero steering preserves the original residual."""
    torch.manual_seed(3)
    sae = TopKSAE(8, 2, 3)
    hidden = torch.randn(2, 5, 8)
    direction = torch.zeros(16)
    direction[:4] = 1
    latent = sae.encode(hidden)
    assert ((latent != 0).sum(dim=-1) <= 3).all()
    assert (latent >= 0).all()
    assert sae.steer(hidden, direction, 0) is hidden
    expected = hidden - sae(hidden) + sae.decoder(latent * torch.exp(0.5 * direction)) * sae.scale + sae.center
    torch.testing.assert_close(sae.steer(hidden, direction, 0.5), expected)
    assert torch.isfinite(active_mean(torch.zeros(2, 5, 16))).all()
    pooled = active_mean(torch.tensor([[[0.0, 2.0], [4.0, 0.0]]]))
    torch.testing.assert_close(pooled, torch.tensor([[4.0, 2.0]]))


def test_sae_small_fit() -> None:
    """A bounded synthetic fit reduces full-cache reconstruction loss."""
    torch.manual_seed(7)
    torch.set_num_threads(2)
    tokens = torch.randn(64, 8)
    sae = TopKSAE(8, 2, 4)
    initial = float((sae(tokens) - tokens).square().mean().detach())
    losses = fit_sae(sae, tokens, 100, 64, 7)
    final = float((sae(tokens) - tokens).square().mean())
    assert len(losses) == 100
    assert np.isfinite(losses).all()
    assert final < initial
    assert not any(p.requires_grad for p in sae.parameters())


def test_feature_selection_rejects_zero_budget() -> None:
    """A zero feature budget must not silently select the entire dictionary."""
    with pytest.raises(ValueError, match="selected_features"):
        feature_direction(TopKSAE(8, 2, 3), torch.randn(2, 5, 8), np.array([0, 1]), 0)


def test_disjoint_and_source_variant_guard() -> None:
    """Both speaker overlap and codec/source overlap are rejected."""
    support = [trial("s1", "u1", "A21")]
    check_disjoint({"support": support, "test": [trial("s2", "u2", "A22")]})
    with pytest.raises(ValueError, match="Leakage"):
        check_disjoint({"support": support, "test": [trial("s1", "u2", "A22")]})
    with pytest.raises(ValueError, match="Leakage"):
        check_disjoint({"support": support, "test": [trial("s2", "codec_u1", "A22", "u1")]})
    assert speaker_partition("s1", 42) == speaker_partition("s1", 42)


def test_source_dedup_and_deterministic_sampling() -> None:
    """Sampling is independent of protocol ordering and removes codec replicas."""
    records = [trial("s1", "u1", "A21", "original"), trial("s1", "u2", "A21", "original"), trial("s2", "u3", "A21")]
    selected = select_trials(records, ["A21"], 2, 42)
    assert selected == select_trials(list(reversed(records)), ["A21"], 2, 42)
    assert len(selected) == 2
    with pytest.raises(ValueError, match="found 2"):
        select_trials(records, ["A21"], 3, 42)


def test_parse_metadata_layouts(tmp_path: Path) -> None:
    """Attack IDs, labels, source links, and codec fields retain their semantics.

    Args:
        tmp_path: Temporary pytest directory.
    """
    (tmp_path / "last_eval").mkdir()
    (tmp_path / "last_eval/ASVspoof5.eval.track_1.tsv").write_text(
        "E_1 E_2 F C01 1 E_3 AC1 A21 spoof -\nE_4 E_5 M - 0 - - bonafide bonafide -\n",
    )
    (tmp_path / "ASVspoof5.train.metadata.txt").write_text("T_1 T_2 F - A01 spoof\n")
    evaluation = read_trials(tmp_path, "eval")
    assert evaluation[0].source == "E_3"
    assert evaluation[0].attack == "A21"
    assert evaluation[0].codec == "C01"
    assert evaluation[1].label == 0
    assert read_trials(tmp_path, "train")[0].attack == "A01"


def test_per_attack_score_direction_and_fixed_threshold() -> None:
    """Each attack reuses bonafide references; fixed-threshold degradation is visible."""
    records = [
        trial("s1", "b1", "bonafide"),
        trial("s2", "b2", "bonafide"),
        trial("s3", "a1", "A21"),
        trial("s4", "a2", "A22"),
    ]
    scores = np.array([0.9, 0.8, 0.1, 0.85])
    threshold = dev_threshold(records, scores)
    assert np.isfinite(threshold)
    report = attack_report(records, scores, 0.5)
    assert report["A21"]["eer"] == 0
    assert report["A21"]["spoof_acceptance"] == 0
    assert report["A22"]["spoof_acceptance"] == 1
    assert report["A22"]["bonafide_rejection"] == 0
    assert report["A21"]["n_bonafide"] == report["A22"]["n_bonafide"] == 2


class TinyDetector(nn.Module):
    """Small real wav2vec2 encoder to validate installed transformer hook semantics."""

    def __init__(self) -> None:
        super().__init__()
        cfg = Wav2Vec2Config(
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
            conv_dim=(8, 8, 8),
            conv_stride=(5, 2, 2),
            conv_kernel=(10, 3, 3),
            num_conv_pos_embeddings=8,
            num_conv_pos_embedding_groups=2,
            mask_time_prob=0.0,
            do_stable_layer_norm=True,
        )
        self.encoder = Wav2Vec2Model(cfg)
        self.head = nn.Linear(8, 2)

    @override
    def forward(self, input_values: Tensor, attention_mask: Tensor) -> dict[str, Tensor]:
        """Classify mean encoder states.

        Args:
            input_values: Waveforms.
            attention_mask: Valid samples.

        Returns:
            Two-class logits.
        """
        hidden = self.encoder(input_values, attention_mask=attention_mask).last_hidden_state
        return {"logits": self.head(hidden.mean(dim=1))}


def test_real_transformer_hook_identity_and_cleanup() -> None:
    """Capture matches inference; identity is exact and hooks are removed on errors."""
    torch.manual_seed(2)
    model = TinyDetector().eval().requires_grad_(False)
    block = model.encoder.encoder.layers[0]
    waves = torch.randn(2, 800)
    with torch.inference_mode():
        direct = model(waves, torch.ones_like(waves, dtype=torch.long))["logits"].softmax(-1)[:, 0].numpy()
    hooked, hidden = run_model(model, waves, block, 2, capture=True)
    np.testing.assert_array_equal(hooked, direct)
    assert hidden is not None and hidden.shape[0] == 2 and hidden.shape[-1] == 8

    def fail(_hidden: Tensor) -> Tensor:
        message = "intentional failure"
        raise RuntimeError(message)

    with pytest.raises(RuntimeError, match="intentional"):
        run_model(model, waves, block, 2, fail)
    recovered, _ = run_model(model, waves, block, 2)
    np.testing.assert_array_equal(recovered, direct)


def test_retention_guard_rejects_bonafide_damage() -> None:
    """A21 gains cannot override configured bonafide protection."""
    from omegaconf import OmegaConf

    baseline = {"dev_eer": 0.2, "tune_bonafide_rejection": 0.1, "dev_bonafide_rejection": 0.1}
    settings = OmegaConf.create({"max_dev_eer_degradation": 0.02, "max_bonafide_degradation": 0.05})
    assert admissible(baseline, baseline, settings)
    assert not admissible({**baseline, "dev_bonafide_rejection": 0.3}, baseline, settings)
