"""ASVspoof 5 loaders for Wav2Vec2 waveforms."""

from functools import partial
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import Wav2Vec2FeatureExtractor

from .dataset import AudioDataset, WaveformExample, collate_audio


class AudioData:
    """Train, dev, and eval loaders of mean-centered ASVspoof 5 clips.

    Args:
        batch_size: Examples per batch.
        num_workers: Parallel worker processes.
        pin_memory: Copy batches into pinned memory when CUDA is available.
        model_name: Feature-extractor checkpoint. Must match the Wav2Vec2 frontend.
        num_samples: Crop or pad length.
        root: ASVspoof 5 root with train metadata, `flac_T`, `flac_D`, and `last_eval`.
        augmentation: Training compose. Applied only to the training split.
        limit: Maximum trials per split. `None` uses every trial.

    """

    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        model_name: str,
        num_samples: int,
        root: str,
        augmentation: nn.Module,
        limit: int | None = None,
    ) -> None:
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory if torch.cuda.is_available() else False
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.feature_extractor.do_normalize = False
        dataset_root = Path(root)
        self.train_dataset = AudioDataset(
            dataset_root / "ASVspoof5.train.metadata.txt",
            dataset_root / "flac_T",
            num_samples,
            random_crop=True,
            augmentation=augmentation,
            limit=limit,
        )
        self.val_dataset = AudioDataset(
            dataset_root / "ASVspoof5.dev.metadata.txt",
            dataset_root / "flac_D",
            num_samples,
            limit=limit,
        )
        self.test_dataset = AudioDataset(
            dataset_root / "last_eval" / "ASVspoof5.eval.track_1.tsv",
            dataset_root / "last_eval" / "flac_E_eval",
            num_samples,
            track_eval=True,
            limit=limit,
        )

    def train_dataloader(self) -> DataLoader[WaveformExample]:
        """Return the shuffled training loader.

        Returns:
            Loader of batches with `input_values`, `attention_mask`, and `labels`.

        """
        return self._loader(self.train_dataset, shuffle=True, drop_last=True)

    def val_dataloader(self) -> DataLoader[WaveformExample]:
        """Return the dev loader.

        Returns:
            Loader of batches with `input_values`, `attention_mask`, and `labels`.

        """
        return self._loader(self.val_dataset, shuffle=False, drop_last=False)

    def test_dataloader(self) -> DataLoader[WaveformExample]:
        """Return the eval loader.

        Returns:
            Loader of batches with `input_values`, `attention_mask`, and `labels`.

        """
        return self._loader(self.test_dataset, shuffle=False, drop_last=False)

    def _loader(self, dataset: AudioDataset, *, shuffle: bool, drop_last: bool) -> DataLoader[WaveformExample]:
        """Build one loader.

        Args:
            dataset: Split to iterate.
            shuffle: Shuffle examples each epoch.
            drop_last: Drop the final incomplete batch.

        Returns:
            Configured loader.

        """
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            shuffle=shuffle,
            drop_last=drop_last,
            collate_fn=partial(collate_audio, feature_extractor=self.feature_extractor),
        )
