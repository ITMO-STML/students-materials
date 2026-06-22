"""
Извлечение признаков для baseline-моделей CMU-MOSEI.

Текст: (1) mean-pooled GloVe — реализовано.
       (2) TBD — зависит от наличия сырых слов в mosei.hdf5, см. примечание в коде вызова.
Аудио: (1) MFCC-блок COVAREP (mean+std pooling).
       (2) Просодика/voice quality (mean+std pooling).

Все признаки кешируются в data/processed/.
"""

import pickle
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Разбиение COVAREP на 2 смысловые группы — обосновано корреляционным
# анализом из EDA (раздел 3.5): два блока избыточности соответствуют
# физической группировке "просодика/voice quality" vs "MFCC".
PROSODY_VOICE_IDX = list(range(0, 36))
MFCC_IDX = list(range(36, 74))

INF_COLUMN = 7  # найдено в EDA: -inf на безголосых кадрах (вероятно log F0)


def get_real_mask(text, audio, vision):
    """(N, 50) bool — True там, где шаг реальный, не паддинг."""
    return ~(
        np.all(text == 0, axis=-1) &
        np.all(audio == 0, axis=-1) &
        np.all(vision == 0, axis=-1)
    )


def fix_inf(audio, col=INF_COLUMN):
    """Заменяет inf/-inf в проблемной колонке на минимальное конечное значение этой же колонки."""
    audio = audio.copy()
    col_vals = audio[..., col]
    finite_vals = col_vals[np.isfinite(col_vals)]
    fallback = finite_vals.min() if len(finite_vals) > 0 else 0.0
    col_vals[~np.isfinite(col_vals)] = fallback
    audio[..., col] = col_vals
    return audio


def masked_mean_pool(seq, mask):
    """seq: (N, T, D), mask: (N, T) bool -> (N, D), среднее по реальным шагам."""
    mask_f = mask.astype(seq.dtype)[..., None]
    summed = (seq * mask_f).sum(axis=1)
    counts = np.clip(mask_f.sum(axis=1), 1e-6, None)
    return summed / counts


def masked_mean_std_pool(seq, mask):
    """Конкатенация [mean, std] по реальным шагам -> (N, 2*D)."""
    mean = masked_mean_pool(seq, mask)
    mask_f = mask.astype(seq.dtype)[..., None]
    counts = np.clip(mask_f.sum(axis=1), 1e-6, None)
    sq_diff = ((seq - mean[:, None, :]) ** 2) * mask_f
    var = sq_diff.sum(axis=1) / counts
    return np.concatenate([mean, np.sqrt(var)], axis=1)


class FeatureScaler:
    """Per-feature z-score: fit только на train, transform на всех сплитах (без утечки данных)."""

    def __init__(self):
        self.mean_, self.std_ = None, None

    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ < 1e-6] = 1.0
        return self

    def transform(self, X):
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def extract_audio_features(pkl_data):
    """{split: {'mfcc': (N, 76), 'prosody': (N, 72)}} — после mean+std pooling и стандартизации."""
    result = {}
    scaler_mfcc, scaler_prosody = FeatureScaler(), FeatureScaler()

    for split in ["train", "valid", "test"]:
        text = pkl_data[split]["text"]
        audio = fix_inf(pkl_data[split]["audio"])
        vision = pkl_data[split]["vision"]
        mask = get_real_mask(text, audio, vision)

        mfcc_pooled = masked_mean_std_pool(audio[..., MFCC_IDX], mask)
        prosody_pooled = masked_mean_std_pool(audio[..., PROSODY_VOICE_IDX], mask)

        if split == "train":
            mfcc_pooled = scaler_mfcc.fit_transform(mfcc_pooled)
            prosody_pooled = scaler_prosody.fit_transform(prosody_pooled)
        else:
            mfcc_pooled = scaler_mfcc.transform(mfcc_pooled)
            prosody_pooled = scaler_prosody.transform(prosody_pooled)

        result[split] = {
            "mfcc": mfcc_pooled.astype(np.float32),
            "prosody": prosody_pooled.astype(np.float32),
        }
    return result


def extract_text_glove_features(pkl_data):
    """Признак 1 текста: mean-pooled GloVe. {split: (N, 300)}"""
    result = {}
    for split in ["train", "valid", "test"]:
        text, audio, vision = pkl_data[split]["text"], pkl_data[split]["audio"], pkl_data[split]["vision"]
        mask = get_real_mask(text, audio, vision)
        result[split] = masked_mean_pool(text, mask).astype(np.float32)
    return result

from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
import h5py


def load_words_index(hdf5_path, verbose=False):
    flat = {}
    with h5py.File(hdf5_path, "r") as f:
        grp = f["words"]
        if verbose:
            sample_key = next(iter(grp.keys()))
            print("[words] пример ключа:", sample_key)
        for key in grp.keys():
            video_id, seg_str = key.rsplit("[", 1)
            idx = int(seg_str.rstrip("]"))
            raw = grp[key]["features"][:]
            words = []
            for w in raw.flatten():
                if isinstance(w, bytes):
                    w = w.decode("utf-8", errors="ignore")
                else:
                    w = str(w)
                words.append(w)
            flat[(video_id, idx)] = words
    return flat


def build_transcripts(ids_array, matched_idx, words_flat, real_length=None):
    """
    Возвращает (transcripts: list[str], word_counts: np.ndarray).
    Слова-маркеры пауз ('sp', пустые строки) отфильтровываются.
    Если real_length передан — печатает проверку согласованности индексации.
    """
    N = len(ids_array)
    transcripts = []
    word_counts = np.zeros(N, dtype=np.int32)

    for i in range(N):
        vid = ids_array[i, 0]
        idx = matched_idx[i]
        words = words_flat.get((vid, idx)) if idx != -1 else None
        if words is None:
            transcripts.append("")
            continue
        clean = [w for w in words if w.strip() and w.lower() != "sp"]
        transcripts.append(" ".join(clean))
        word_counts[i] = len(clean)

    if real_length is not None:
        valid = matched_idx != -1
        corr = np.corrcoef(word_counts[valid], real_length[valid])[0, 1]
        diff = np.abs(word_counts[valid] - real_length[valid])
        print(f"\n[проверка индексации] корреляция числа слов и real_length: {corr:.3f}")
        print(f"[проверка индексации] медиана |разницы| в словах: {np.median(diff):.1f}")
        print(f"[проверка индексации] пустых транскриптов (нет соответствия): "
              f"{(np.array([t == '' for t in transcripts])).sum()} ({100*(~valid).mean():.1f}%)")

    return transcripts, word_counts


def extract_text_tfidf_features(transcripts_by_split, max_features=2000, min_df=2):
    """Fit TF-IDF только на train, transform на valid/test (без утечки данных)."""
    vectorizer = TfidfVectorizer(max_features=max_features, min_df=min_df)
    result = {"train": vectorizer.fit_transform(transcripts_by_split["train"]).toarray().astype(np.float32)}
    for split in ["valid", "test"]:
        result[split] = vectorizer.transform(transcripts_by_split[split]).toarray().astype(np.float32)
    return result, vectorizer

def extract_text_tfidf_pipeline(pkl_data, label_cache, hdf5_path, max_features=2000, min_df=2):
    """Полный пайплайн: hdf5 -> слова -> транскрипты -> TF-IDF (fit только на train)."""
    words_flat = load_words_index(hdf5_path)

    transcripts_by_split = {}
    for split in ["train", "valid", "test"]:
        ids = label_cache[split]["ids"]
        matched_idx = label_cache[split]["hdf5_idx"]
        transcripts, _ = build_transcripts(ids, matched_idx, words_flat)
        transcripts_by_split[split] = transcripts

    tfidf_features, vectorizer = extract_text_tfidf_features(
        transcripts_by_split, max_features=max_features, min_df=min_df
    )
    return tfidf_features, vectorizer, transcripts_by_split


def build_feature_cache(pkl_data, label_cache, hdf5_path, force_rebuild=False):
    cache_path = PROCESSED_DIR / "baseline_features_cache.pkl"
    if cache_path.exists() and not force_rebuild:
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    tfidf_features, vectorizer, _ = extract_text_tfidf_pipeline(pkl_data, label_cache, hdf5_path)

    cache = {
        "text_glove": extract_text_glove_features(pkl_data),
        "text_tfidf": tfidf_features,
        "audio": extract_audio_features(pkl_data),
    }
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(cache, f)
    return cache

def fix_inf_sequence(audio_seq, col=INF_COLUMN):
    """Замена inf в проблемной колонке для последовательностей (N, T, D)."""
    out = audio_seq.copy()
    col_vals = out[..., col]
    finite_mask = np.isfinite(col_vals)
    if (~finite_mask).any():
        fallback = col_vals[finite_mask].min() if finite_mask.any() else 0.0
        col_vals[~finite_mask] = fallback
        out[..., col] = col_vals
    return out


class SequenceFeatureScaler:
    """Per-feature z-score для последовательностей. Параметры подгоняются по реальным (не паддинг) шагам train."""

    def __init__(self):
        self.mean_, self.std_ = None, None

    def fit(self, X, mask):
        valid = X[mask]  # (total_valid_steps, D)
        self.mean_ = valid.mean(axis=0)
        self.std_ = valid.std(axis=0)
        self.std_[self.std_ < 1e-6] = 1.0
        return self

    def transform(self, X):
        return (X - self.mean_) / self.std_

    def fit_transform(self, X, mask):
        return self.fit(X, mask).transform(X)


def extract_sequence_features(pkl_data):
    """
    Возвращает {modality: {split: {'X': (N, T, D), 'mask': (N, T)}}}.
    Используется для CNN/RNN/Transformer (разделы 5+ курсовой).
    """
    result = {"text": {}, "audio": {}}
    text_scaler = SequenceFeatureScaler()
    audio_scaler = SequenceFeatureScaler()

    for split in ["train", "valid", "test"]:
        text = pkl_data[split]["text"]
        audio = fix_inf_sequence(pkl_data[split]["audio"])
        vision = pkl_data[split]["vision"]
        mask = get_real_mask(text, audio, vision)

        if split == "train":
            text_scaled = text_scaler.fit_transform(text, mask)
            audio_scaled = audio_scaler.fit_transform(audio, mask)
        else:
            text_scaled = text_scaler.transform(text)
            audio_scaled = audio_scaler.transform(audio)

        text_scaled = (text_scaled * mask[..., None]).astype(np.float32)
        audio_scaled = (audio_scaled * mask[..., None]).astype(np.float32)

        result["text"][split] = {"X": text_scaled, "mask": mask}
        result["audio"][split] = {"X": audio_scaled, "mask": mask}

    return result