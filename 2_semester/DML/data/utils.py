from enum import Enum

import numpy as np
import pandas as pd
import torch
import torchaudio
from scipy.sparse import load_npz
from torch.utils.data import DataLoader, Dataset


class FeatureType(Enum):
    TFIDF = "tf_idf"
    W2V = "w2v"
    MEL = "mel"
    MFCC = "mfcc"
    CONFORMER = "conformer"
    BERT = "bert"
    EARLY_FUSION = "early_fusion"
    EARLY_FUSION_V2 = "early_fusion_v2"
    LATE_FUSION = "late_fusion"
    CROSS_ATTN = "cross-attn-fusion"


class SplitType(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class SentimentDataset(Dataset):
    def __init__(
        self,
        feature_type: FeatureType,
        split: SplitType,
        mosei_path: str,
        max_length: int = 32000,
    ) -> None:
        suffix = "original" if split == "test" else "modified"
        split_name = split[0].upper() + split[1:]
        df = pd.read_csv(mosei_path + f"/Data_{split_name}_{suffix}.csv")
        self.sentiment = df["sentiment"].to_list()
        self.sent_cls = {"negative": 0, "neutral": 1, "positive": 2}
        self.max_length = max_length
        feature_type = FeatureType(feature_type)

        match feature_type:
            case FeatureType.TFIDF:
                data = load_npz(f"features/{split}_tfidf.npz").toarray()
            case FeatureType.W2V:
                data = np.load(f"features/{split}_w2v.npy")
            case FeatureType.MEL:
                data = np.load(f"features/{split}_mel.npy")
            case FeatureType.MFCC:
                data = np.load(f"features/{split}_mfcc.npy")
            case FeatureType.CONFORMER:
                df = pd.read_csv(mosei_path + f"/Data_{split_name}_{suffix}.csv")
                wav_dir = "/Audio/WAV_16000/"
                data = df["video"].apply(lambda x: mosei_path + wav_dir + x + ".wav").to_list()
            case FeatureType.BERT:
                df = pd.read_csv(mosei_path + f"/Data_{split_name}_{suffix}.csv")
                data = df["text"].to_list()
            case _:
                wrong_type = feature_type
                raise RuntimeError(f"got unexpected type {wrong_type}")
        if feature_type != FeatureType.CONFORMER and feature_type != FeatureType.BERT:
            self.data = torch.from_numpy(data).float().squeeze()
        else:
            self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> tuple:
        features = self.data[index]
        sentiment = self.sentiment[index]
        if sentiment < 0:
            label = self.sent_cls["negative"]
        elif sentiment > 0:
            label = self.sent_cls["positive"]
        else:
            label = self.sent_cls["neutral"]

        # wav_path
        if isinstance(features, str) and features.endswith(".wav"):
            waveform, sr = torchaudio.load(features)
            if sr != 16_000:
                waveform = torchaudio.functional.resample(waveform, sr, 16_000)
            waveform = waveform.squeeze(0)

            # Pad or truncate to max_length
            if waveform.shape[0] > self.max_length:
                waveform = waveform[: self.max_length]
            else:
                pad_len = self.max_length - waveform.shape[0]
                waveform = torch.cat([waveform, torch.zeros(pad_len)])

            features = waveform
            length = min(features.shape[0], self.max_length)
            return features, length, label

        return features, label


class MultiModalSentimentDataset(Dataset):
    def __init__(
        self,
        split: SplitType,
        mosei_path: str,
        max_length: int = 32000,
    ) -> None:
        suffix = "original" if split == "test" else "modified"
        split_name = split[0].upper() + split[1:]
        df = pd.read_csv(mosei_path + f"/Data_{split_name}_{suffix}.csv")
        self.sentiment = df["sentiment"].to_list()
        self.sent_cls = {"negative": 0, "neutral": 1, "positive": 2}
        self.max_length = max_length

        wav_dir = "/Audio/WAV_16000/"
        self.wavs = df["video"].apply(lambda x: mosei_path + wav_dir + x + ".wav").to_list()
        self.texts = df["text"].to_list()

    def __len__(self) -> int:
        return len(self.wavs)

    def __getitem__(self, index: int) -> tuple:
        wav_path = self.wavs[index]
        text = self.texts[index]

        sentiment = self.sentiment[index]
        if sentiment < 0:
            label = self.sent_cls["negative"]
        elif sentiment > 0:
            label = self.sent_cls["positive"]
        else:
            label = self.sent_cls["neutral"]

        # wav_path
        waveform, sr = torchaudio.load(wav_path)
        if sr != 16_000:
            waveform = torchaudio.functional.resample(waveform, sr, 16_000)
        waveform = waveform.squeeze(0)

        # Pad or truncate to max_length
        if waveform.shape[0] > self.max_length:
            waveform = waveform[: self.max_length]
        else:
            pad_len = self.max_length - waveform.shape[0]
            waveform = torch.cat([waveform, torch.zeros(pad_len)])

        length = min(waveform.shape[0], self.max_length)
        return text, waveform, length, label


def collate_fn(batch):
    # multimodal
    if isinstance(batch[0], tuple) and len(batch[0]) == 4:
        texts, waveforms, lengths, labels = zip(*batch)
        waveforms = torch.stack(waveforms)  # [batch, channel, sample]
        lengths = torch.tensor(lengths)
        labels = torch.tensor(labels)
        return texts, waveforms, lengths, labels

    # audio for conformer
    if isinstance(batch[0], tuple) and len(batch[0]) == 3:
        waveforms, lengths, labels = zip(*batch)
        waveforms = torch.stack(waveforms)  # [batch, channel, sample]
        lengths = torch.tensor(lengths)
        labels = torch.tensor(labels)
        return waveforms, lengths, labels

    if isinstance(batch[0], tuple) and isinstance(batch[0][0], str):
        # Text: (text, label)
        texts, labels = zip(*batch)
        labels = torch.tensor(labels)
        return texts, labels

    # precomputed features
    features, labels = zip(*batch)
    features = torch.stack(features)
    labels = torch.tensor(labels)
    return features, labels


def get_dataloader(encoding_type: FeatureType, split: SplitType, mosei_path: str = "/home/ext-yudin-a@ad.speechpro.com/dml/datasets/CMU-MOSEI/") -> DataLoader:
    feature_type = FeatureType(encoding_type)
    if feature_type in (FeatureType.EARLY_FUSION, FeatureType.EARLY_FUSION_V2, FeatureType.LATE_FUSION, FeatureType.CROSS_ATTN):
        ds = MultiModalSentimentDataset(split, mosei_path)
    else:  # unimodal
        ds = SentimentDataset(encoding_type, split, mosei_path)
    return DataLoader(
        ds,
        batch_size=32,
        num_workers=4,
        pin_memory=True,
        shuffle=split == "train",
        drop_last=split == "train",
        collate_fn=collate_fn,
    )
