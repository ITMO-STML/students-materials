import argparse
import re

import gensim
import numpy as np
import pandas as pd
import torchaudio
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from torch import Tensor
from torchaudio import transforms as T


class LogMelSpectrogram(T.MelSpectrogram):
    def __init__(self, eps=1e-8, **kwargs) -> None:
        super().__init__(**kwargs)
        self.eps = eps

    def forward(self, wav_path: str) -> Tensor:
        waveform, sr = torchaudio.load(wav_path, normalize=True)
        if sr != 16_000:
            waveform = torchaudio.functional.resample(waveform, sr, 16_000)
        log_mel_spec = (super().forward(waveform) + self.eps).log()
        return log_mel_spec.mean(dim=-1)


class MFCC(T.MFCC):
    def __init__(self, eps=1e-8, **kwargs) -> None:
        super().__init__(**kwargs)
        self.eps = eps

    def forward(self, wav_path: str) -> Tensor:
        waveform, sr = torchaudio.load(wav_path, normalize=True)
        if sr != 16_000:
            waveform = torchaudio.functional.resample(waveform, sr, 16_000)
        mfcc = super().forward(waveform) + self.eps
        return mfcc.mean(dim=-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", nargs="*", choices=["tf_idf", "w2v", "mel", "mfcc"], type=list[str], required=True)
    parser.add_argument("--mosei_path", type=str, required=True)
    return parser.parse_args()


def clean_text(text: str) -> str:
    with open("stopwords.txt") as file:
        stop_words = set(file.read().splitlines())
    text = text.lower().strip()
    text = re.sub(r"(\w+)'(s|re|t|ll|ve|d|m)\b", r"\1", text, flags=re.IGNORECASE)
    text = re.sub(r"[^\w\s]", "", text)
    words = text.split()
    words = [word.lower() for word in words if word.lower() not in stop_words]
    return " ".join(words)


def tokenize(text: str) -> list[str]:
    return text.split()


def get_sentence_mean_vec(w2v, text: list[str]) -> np.ndarray:
    vecs = []
    for word in text:
        try:
            vec = w2v.get_vector(word)
        except Exception:
            vec = [0] * w2v.vector_size
        vecs.append(vec)
    return np.mean(vecs, axis=0)


def main() -> None:
    args = parse_args()
    train_df = pd.read_csv(args.mosei_path + "/Data_Train_modified.csv")
    val_df = pd.read_csv(args.mosei_path + "/Data_Val_modified.csv")
    test_df = pd.read_csv(args.mosei_path + "/Data_Test_original.csv")

    for vec_type in args.type:
        if vec_type in ("tf_idf", "w2v"):
            train_text = train_df["text"].apply(clean_text).to_list()
            val_text = val_df["text"].apply(clean_text).to_list()
            test_text = test_df["text"].apply(clean_text).to_list()
            if vec_type == "tf_idf":
                tf_idf = TfidfVectorizer(min_df=0.001, max_df=0.999, max_features=300)
                train_tf_idf = tf_idf.fit_transform(train_text)
                val_tf_idf = tf_idf.transform(val_text)
                test_tf_idf = tf_idf.transform(test_text)

                save_npz("features/train_tfidf.npz", train_tf_idf)
                save_npz("features/val_tfidf.npz", val_tf_idf)
                save_npz("features/test_tfidf.npz", test_tf_idf)
            else:
                train_tokenized = [tokenize(text) for text in train_text]
                val_tokenized = [tokenize(text) for text in val_text]
                test_tokenized = [tokenize(text) for text in test_text]

                w2v = gensim.models.Word2Vec(train_tokenized, vector_size=250, min_count=2, window=3)

                train_w2v = [get_sentence_mean_vec(w2v, text) for text in train_tokenized]
                val_w2v = [get_sentence_mean_vec(w2v, text) for text in val_tokenized]
                test_w2v = [get_sentence_mean_vec(w2v, text) for text in test_tokenized]

                np.save("features/train_w2v.npy", np.array(train_w2v))
                np.save("features/val_w2v.npy", np.array(val_w2v))
                np.save("features/test_w2v.npy", np.array(test_w2v))

        else:
            wav_dir = "/Audio/WAV_16000/"
            train_wavs = train_df["video"].apply(lambda x: args.mosei_path + wav_dir + x + ".wav").to_list()
            val_wavs = val_df["video"].apply(lambda x: args.mosei_path + wav_dir + x + ".wav").to_list()
            test_wavs = test_df["video"].apply(lambda x: args.mosei_path + wav_dir + x + ".wav").to_list()
            if vec_type == "mel":
                mel = LogMelSpectrogram(
                    sample_rate=16000,
                    n_fft=2048,
                    hop_length=128,
                    n_mels=128,
                    f_min=40,
                    f_max=8000,
                    mel_scale="slaney",
                )

                train_mel = [mel(wav_path).numpy() for wav_path in train_wavs]
                val_mel = [mel(wav_path).numpy() for wav_path in val_wavs]
                test_mel = [mel(wav_path).numpy() for wav_path in test_wavs]

                np.save("features/train_mel.npy", np.array(train_mel))
                np.save("features/val_mel.npy", np.array(val_mel))
                np.save("features/test_mel.npy", np.array(test_mel))
            else:
                mfcc = MFCC(
                    sample_rate=16_000,
                    n_mfcc=40,
                )

                train_mfcc = [mfcc(wav_path).numpy() for wav_path in train_wavs]
                val_mfcc = [mfcc(wav_path).numpy() for wav_path in val_wavs]
                test_mfcc = [mfcc(wav_path).numpy() for wav_path in test_wavs]

                np.save("features/train_mfcc.npy", np.array(train_mfcc))
                np.save("features/val_mfcc.npy", np.array(val_mfcc))
                np.save("features/test_mfcc.npy", np.array(test_mfcc))


if __name__ == "__main__":
    main()
