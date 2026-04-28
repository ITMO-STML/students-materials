import panphon
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv

from typing import List, Dict
from Levenshtein import editops

def symbols_dicts():
    return {
    'VOWEL_BASE': {"a": "a",
                    "e": "e",
                    "i": "i",
                    "o": "o",
                    "u": "u",
                    "y": "ɨ",
                   },

    "REDUCTION_BY_VOWEL": { "a": {"1": "ʌ", "2": "ʌ", "4": "ə"},
                            "o": {"1": "ʌ", "2": "ʌ", "4": "ə"},
                            "e": {"1": "e", "2": "e", "4": "ə"},
                            "i": {"1": "ɪ", "2": "ɪ", "4": "ɪ"},
                            "u": {"1": "ʊ", "2": "ʊ", "4": "ʊ"},
                            "y": {"1": "ɨ", "2": "ɨ", "4": "ɨ"},},

    "CONSONANT_BASE": {
                        "b":  "b",
                        "p":  "p",
                        "d":  "d",
                        "t":  "t",
                        "g":  "ɡ",
                        "k":  "k",
                        "v":  "v",
                        "f":  "f",
                        "z":  "z",
                        "s":  "s",
                        "zh": "ʐ",
                        "sh": "ʂ",
                        "sc": "ɕ",      # вместо "ɕː" — если нужна долгота, можно "ɕː"
                        "c":  "t͡s",
                        "ch": "t͡ɕ",
                        "ch_": "tʃ",    # если хочешь оставить аффрикату без палатализации
                        "m":  "m",
                        "n":  "n",
                        "l":  "l",
                        "r":  "r",
                        "j":  "j",
                        "h":  "x",
                        'sil': 'sil'
                    },

    "SPECIAL_UPPERCASE": {  "C": "t͡s",
                            "CH": "t͡ɕ",
                            "H": "x",
                            "SC": "ɕː",},
    "FEATURES": {
    "b": dict(type="consonant", voice="voiced", place="bilabial", manner="stop"),
    "p": dict(type="consonant", voice="voiceless", place="bilabial", manner="stop"),
    "d": dict(type="consonant", voice="voiced", place="dental", manner="stop"),
    "t": dict(type="consonant", voice="voiceless", place="dental", manner="stop"),
    "g": dict(type="consonant", voice="voiced", place="velar", manner="stop"),
    "k": dict(type="consonant", voice="voiceless", place="velar", manner="stop"),
    "v": dict(type="consonant", voice="voiced", place="labiodental", manner="fricative"),
    "f": dict(type="consonant", voice="voiceless", place="labiodental", manner="fricative"),
    "z": dict(type="consonant", voice="voiced", place="alveolar", manner="fricative"),
    "s": dict(type="consonant", voice="voiceless", place="alveolar", manner="fricative"),
    "zh": dict(type="consonant", voice="voiced", place="postalveolar", manner="fricative"),
    "sh": dict(type="consonant", voice="voiceless", place="postalveolar", manner="fricative"),
    "sc": dict(type="consonant", voice="voiceless", place="alveolo-palatal", manner="fricative"),
    "c": dict(type="consonant", voice="voiceless", place="alveolar", manner="affricate"),
    "ch": dict(type="consonant", voice="voiceless", place="alveolo-palatal", manner="affricate"),
    "ch_": dict(type="consonant", voice="voiceless", place="postalveolar", manner="affricate"),
    "m": dict(type="consonant", voice="voiced", place="bilabial", manner="nasal"),
    "n": dict(type="consonant", voice="voiced", place="dental", manner="nasal"),
    "l": dict(type="consonant", voice="voiced", place="alveolar", manner="lateral"),
    "r": dict(type="consonant", voice="voiced", place="alveolar", manner="trill"),
    "j": dict(type="consonant", voice="voiced", place="palatal", manner="approximant"),
    "h": dict(type="consonant", voice="voiceless", place="velar", manner="fricative"),

# vowels (base)
    "a": dict(type="vowel", height="open", backness="central", rounded=False),
    "e": dict(type="vowel", height="mid", backness="front", rounded=False),
    "i": dict(type="vowel", height="close", backness="front", rounded=False),
    "o": dict(type="vowel", height="mid", backness="back", rounded=True),
    "u": dict(type="vowel", height="close", backness="back", rounded=True),
    "y": dict(type="vowel", height="close", backness="central", rounded=False),
}}



def corpres_to_ipa_symbol(symbol: str) -> str:
    VOWEL_BASE = symbols_dicts()['VOWEL_BASE']

    REDUCTION_BY_VOWEL = symbols_dicts()['REDUCTION_BY_VOWEL']

    CONSONANT_BASE = symbols_dicts()['CONSONANT_BASE']

    SPECIAL_UPPERCASE = symbols_dicts()['SPECIAL_UPPERCASE']

    # Uppercase normalization
    if len(symbol.split(' ')) > 1:
        symbol = symbol.split(' ')[1]
    if symbol in SPECIAL_UPPERCASE:
        return SPECIAL_UPPERCASE[symbol]

    # Palatalization
    palatalized = symbol.endswith("'")
    base = symbol[:-1] if palatalized else symbol

    # Vowels with reduction
    if len(base) >= 2 and base[0] in VOWEL_BASE:
        vowel = base[0]
        idx = base[1]

        ipa_vowel = VOWEL_BASE[vowel]
        reduced = REDUCTION_BY_VOWEL.get(vowel, {}).get(idx)

        if reduced:
            ipa_vowel = reduced


        return ipa_vowel

    # Consonants
    ipa = CONSONANT_BASE.get(base, f"[UNK:{symbol}]")

    if palatalized:
        ipa += "ʲ"

    return ipa


def corpres2ipa(seq: List[str]) -> str:
    """Convert sequence of corpres symbols → IPA string"""
    return " ".join(corpres_to_ipa_symbol(s) for s in seq)


def corpres_to_features(symbol: str) -> Dict:
    VOWEL_BASE = symbols_dicts()['VOWEL_BASE']
    FEATURES = symbols_dicts()['FEATURES']

    palatalized = symbol.endswith("'")
    base = symbol[:-1] if palatalized else symbol

    # vowels with index
    if len(base) >= 2 and base[0] in VOWEL_BASE:
        vowel = base[0]
        idx = base[1]

        feat = FEATURES[vowel].copy()
        feat["stress"] = idx == "1"
        feat["reduction"] = idx
        return feat

    feat = FEATURES.get(base, {"type": "unknown"}).copy()
    feat["palatalized"] = palatalized
    return feat


def corpres2features(seq: List[str]) -> List[Dict]:
    """Convert sequence → list of feature dicts"""
    return [corpres_to_features(s) for s in seq]

import pandas as pd

def visualize_consonant_features(features_dict, sort=True):

    rows = []

    for symbol, feats in features_dict.items():
        if feats.get("type") == "consonant":
            row = {"symbol": symbol}
            row.update(feats)
            rows.append(row)

    df = pd.DataFrame(rows)

    if sort:
        sort_cols = [c for c in ["manner", "place", "voice"] if c in df.columns]
        df = df.sort_values(by=sort_cols)

    rows = []
    for sym, feats in features_dict.items():
        if feats.get("type") == "consonant":
            row = {"symbol": sym}
            row.update(feats)
            rows.append(row)

    df = pd.DataFrame(rows).set_index("symbol")

    # One-hot кодирование категориальных признаков
    df_encoded = pd.get_dummies(df[["voice", "place", "manner"]])

    plt.figure()
    plt.imshow(df_encoded.values)
    plt.xticks(range(len(df_encoded.columns)), df_encoded.columns, rotation=90)
    plt.yticks(range(len(df_encoded.index)), df_encoded.index)
    plt.title("Consonant Feature Heatmap")
    plt.tight_layout()
    plt.show()

    return df

def panphon_features(word):
    ft = panphon.FeatureTable()
    fts = ft.word_fts(word)
    return fts  #returns list of features for all the phonemes


ft = panphon.FeatureTable()  # создаем объект PanPhon один раз


def phonetic_features(seq: List[str]) -> np.ndarray:
    """
    Преобразует последовательность символов (в IPA) в матрицу признаков.
    Возвращает np.ndarray размерности [num_phonemes, num_features]
    """
    feats_list = []
    for sym in seq:
        # word_fts возвращает список списков признаков (обычно 1 список для 1 символа)
        fts = ft.word_fts(sym)
        if fts:
            feats_list.append(fts[0])
        else:
            # если символ не распознан, используем вектор нулей
            feats_list.append([0] * 24)  # 24 — стандартная длина признаков в PanPhon

    return np.array(feats_list, dtype=int)


def compute_PFER(true_seq: List[str], pred_seq: List[str]) -> float:
    """
    Вычисляет Phonetic Feature Error Rate (PFER)
    true_seq и pred_seq — списки символов в исходной разметке (corpres)
    """
    # Конвертация в IPA
    true_ipa = corpres2ipa(true_seq).split()
    pred_ipa = corpres2ipa(pred_seq).split()

    # Преобразуем в матрицы признаков
    true_feats = phonetic_features(true_ipa)
    pred_feats = phonetic_features(pred_ipa)

    # Выравнивание по длине (если sequences разной длины)
    min_len = min(len(true_feats), len(pred_feats))
    true_feats = true_feats[:min_len]
    pred_feats = pred_feats[:min_len]

    # PFER: количество несовпадающих признаков / всего признаков
    errors = np.sum(true_feats != pred_feats)
    total = true_feats.size

    return errors / total

def compute_PFER_levenshtein(true_seq: List[str], pred_seq: List[str]) -> float:
    """
    PFER с учетом выравнивания Левенштейна
    """
    # 1. Конвертация в IPA
    true_ipa = corpres2ipa(true_seq).split()
    pred_ipa = corpres2ipa(pred_seq).split()

    # 2. Получение признаков
    true_feats = phonetic_features(true_ipa)
    pred_feats = phonetic_features(pred_ipa)

    # 3. Получаем список операций редактирования
    ops = editops(' '.join(true_ipa), ' '.join(pred_ipa))

    # 4. Выравниваем sequences по операциям
    aligned_true = []
    aligned_pred = []

    i_true = i_pred = 0
    for op, i, j in ops:
        if op == 'replace':
            aligned_true.append(true_feats[i_true])
            aligned_pred.append(pred_feats[i_pred])
            i_true += 1
            i_pred += 1
        elif op == 'delete':
            aligned_true.append(true_feats[i_true])
            aligned_pred.append(np.zeros_like(true_feats[i_true]))
            i_true += 1
        elif op == 'insert':
            aligned_true.append(np.zeros_like(pred_feats[i_pred]))
            aligned_pred.append(pred_feats[i_pred])
            i_pred += 1

    # добавляем оставшиеся элементы (если есть)
    while i_true < len(true_feats):
        aligned_true.append(true_feats[i_true])
        aligned_pred.append(np.zeros_like(true_feats[i_true]))
        i_true += 1
    while i_pred < len(pred_feats):
        aligned_true.append(np.zeros_like(pred_feats[i_pred]))
        aligned_pred.append(pred_feats[i_pred])
        i_pred += 1

    aligned_true = np.array(aligned_true)
    aligned_pred = np.array(aligned_pred)

    # 5. PFER = число несовпадающих признаков / всего признаков
    errors = np.sum(aligned_true != aligned_pred)
    total = aligned_true.size

    return errors / total

# sample = ["b", "a1", "t'", "sh", "i0"]
#
# print("CORPRES:", sample)
# print("IPA:", corpres2ipa(sample))
# print("FEATURES:")
# for f in corpres2features(sample):
#     print(f)
#
# #visualize_consonant_features(symbols_dicts()['FEATURES'])



