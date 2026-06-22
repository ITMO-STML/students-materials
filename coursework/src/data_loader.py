# src/data_loader.py
"""
Загрузка CMU-MOSEI и сшивка sentiment (из pkl) с эмоциями (из hdf5).

Источники меток не выровнены позиционно: hdf5 хранит индекс сегмента
как порядок извлечения из видео, не как хронологический порядок по времени.
Сопоставление делается через scipy.linear_sum_assignment, минимизируя
расхождение sentiment между сегментами одного video_id (sentiment есть
в обоих источниках => служит проверяемым "общим ключом").
Эмпирически на train: ~94.8% сэмплов сматчены, 0% с расхождением sentiment
выше порога. Несматченные сэмплы помечаются mask=False и сохраняют NaN
в emotion-полях — они остаются валидными для sentiment-задачи.
"""

import pickle
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
from scipy.optimize import linear_sum_assignment

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PKL_PATH = PROJECT_ROOT / "data" / "raw" / "mosei_senti_data.pkl"
DEFAULT_HDF5_PATH = PROJECT_ROOT / "data" / "raw" / "mosei.hdf5"
DEFAULT_CACHE_PATH = PROJECT_ROOT / "data" / "processed" / "mosei_labels_cache.pkl"


def load_pkl(path=DEFAULT_PKL_PATH):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_hdf5_emotion_index(path=DEFAULT_HDF5_PATH):
    """video_id -> [(seg_idx, feats[7]), ...], feats = [sent, happy, sad, anger, surprise, disgust, fear]"""
    by_video = defaultdict(list)
    with h5py.File(path, "r") as f:
        grp = f["All Labels"]
        for key in grp.keys():
            video_id, seg_str = key.rsplit("[", 1)
            idx = int(seg_str.rstrip("]"))
            feats = grp[key]["features"][0]
            by_video[video_id].append((idx, feats))
    return by_video


def match_emotions(ids_array, sentiment_array, hdf5_by_video, tol=0.05):
    """
    Возвращает:
      y_emotion (N,6) — с NaN на несматченных строках
      mask (N,) bool
      matched_idx (N,) int — индекс сегмента в hdf5 (video_id[idx]), -1 если не сматчено.
      Этот идентификатор используется повторно для поиска слов в группе 'words'
      того же hdf5-файла, в предположении общей индексации сегментов внутри файла.
    """
    N = len(ids_array)
    y_emotion = np.full((N, 6), np.nan, dtype=np.float32)
    mask = np.zeros(N, dtype=bool)
    matched_idx = np.full(N, -1, dtype=np.int64)

    pkl_by_video = defaultdict(list)
    for i in range(N):
        pkl_by_video[ids_array[i, 0]].append(i)

    for vid, idx_list in pkl_by_video.items():
        hdf5_items = hdf5_by_video.get(vid)
        if hdf5_items is None or len(idx_list) != len(hdf5_items):
            continue

        n = len(idx_list)
        cost = np.zeros((n, n))
        for a, i in enumerate(idx_list):
            for b, (_, feats) in enumerate(hdf5_items):
                cost[a, b] = abs(feats[0] - sentiment_array[i])

        row_ind, col_ind = linear_sum_assignment(cost)
        for a, b in zip(row_ind, col_ind):
            if cost[a, b] <= tol:
                i = idx_list[a]
                y_emotion[i] = hdf5_items[b][1][1:]
                mask[i] = True
                matched_idx[i] = hdf5_items[b][0]   # новое: сохраняем сегментный индекс

    return y_emotion, mask, matched_idx


def binarize_sentiment(scores):
    """scores: float array [-3,3] -> int classes {0:negative, 1:neutral, 2:positive}"""
    scores = np.asarray(scores).reshape(-1)
    classes = np.where(scores < 0, 0, np.where(scores == 0, 1, 2))
    return classes.astype(np.int64)


def binarize_emotion_presence(emotion_raw):
    """emotion_raw: (N,6) интенсивности [0,3] (могут содержать NaN) -> (N,6) presence {0,1}, NaN сохраняется."""
    out = np.where(np.isnan(emotion_raw), np.nan, (emotion_raw > 0).astype(np.float32))
    return out


def build_label_cache(pkl_path=DEFAULT_PKL_PATH, hdf5_path=DEFAULT_HDF5_PATH):
    pkl_data = load_pkl(pkl_path)
    hdf5_by_video = load_hdf5_emotion_index(hdf5_path)

    cache = {}
    for split in ("train", "valid", "test"):
        ids = pkl_data[split]["id"]
        sent_raw = pkl_data[split]["labels"].squeeze()

        emotion_raw, mask, matched_idx = match_emotions(ids, sent_raw, hdf5_by_video)

        cache[split] = {
            "ids": ids,
            "sentiment_raw": sent_raw.astype(np.float32),
            "sentiment_class": binarize_sentiment(sent_raw),
            "emotion_raw": emotion_raw,
            "emotion_binary": binarize_emotion_presence(emotion_raw),
            "mask": mask,
            "hdf5_idx": matched_idx,
        }
        print(f"[{split}] N={len(ids)}, эмоции сматчены: {mask.sum()} ({100*mask.mean():.1f}%)")

    return cache

def load_or_build_cache(force_rebuild=False, cache_path=DEFAULT_CACHE_PATH,
                         pkl_path=DEFAULT_PKL_PATH, hdf5_path=DEFAULT_HDF5_PATH):
    cache_path = Path(cache_path)
    if cache_path.exists() and not force_rebuild:
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    cache = build_label_cache(pkl_path, hdf5_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(cache, f)
    return cache


if __name__ == "__main__":
    load_or_build_cache(force_rebuild=True)
    