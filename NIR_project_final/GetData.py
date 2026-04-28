import torch.nn as nn
import os
import numpy as np
from collections import Counter

from torch.utils.data import DataLoader, ConcatDataset

import torch
import random
from typing import Dict, Any, List

import time
def load_selected_embeddings(root_dir, phoneme_list, samples_of_ph, speakers, triplet_variation='one border'):
    info = []
    selected_embeddings = []
    selected_phonemes = []

    for speaker in speakers:
        speaker_root_dir = os.path.join(root_dir, speaker)
        speaker_phonemes = []

        for root, _, files in os.walk(speaker_root_dir):
            if all(Counter(speaker_phonemes)[ph] >= samples_of_ph for ph in phoneme_list):
                break

            for file in files:
                if file == 'ata1355_no_av_ph.txt':
                    pass
                else:

                    if not file.endswith("_no_av_ph.txt"):
                        continue

                    name = file.replace("_no_av_ph.txt", "")
                    phoneme_file_path = os.path.join(root, file)
                    emb_file_path = os.path.join(root, name + "_no_av_embs.npy")

                    if not os.path.exists(emb_file_path):
                        print(f"Нет эмбеддингов для {phoneme_file_path}")
                        continue

                    with open(phoneme_file_path, "r", encoding="utf-8") as f:
                        phonemes = [line.strip() for line in f.readlines()]

                    emb_array = np.load(emb_file_path, allow_pickle=True)  # shape (N, D)

                    if len(phonemes) != len(emb_array):
                        print(f"Несовпадение эмбеддингов и меток в {file}")
                        continue

                    for phoneme, emb in zip(phonemes, emb_array):
                        base_ph = phoneme.split('_')[1]
                        start_ph = phoneme.split('_')[0]
                        end_ph = phoneme.split('_')[2]

                        def add_sample():
                            label = phoneme.split('_')
                            if len(label) != 3:
                                print(file)
                                print(label, file)
                            selected_embeddings.append({
                                'label': label,
                                'embedding': torch.tensor(emb, dtype=torch.float32),
                                'place': file
                            })
                            selected_phonemes.append(phoneme)
                            speaker_phonemes.append(base_ph)

                        if base_ph in phoneme_list and start_ph in phoneme_list and end_ph in phoneme_list:
                            if Counter(speaker_phonemes)[base_ph] < samples_of_ph:
                                if triplet_variation == 'one border':
                                    if start_ph != end_ph or start_ph != base_ph or end_ph != base_ph:
                                        add_sample()

                                elif triplet_variation == 'two borders':
                                    if len({start_ph, base_ph, end_ph}) == 3:  # все разные
                                        add_sample()

                                elif triplet_variation == 'no borders':
                                    if start_ph == end_ph == base_ph:
                                        add_sample()

                                elif triplet_variation == 'all':
                                    add_sample()

                    if all(Counter(speaker_phonemes)[ph] >= samples_of_ph for ph in phoneme_list):
                        break

        print(f"{speaker}: {Counter(speaker_phonemes)}")
        info.append({speaker: Counter(speaker_phonemes)})

    return selected_embeddings, selected_phonemes, info


def load_phrases(
        root_dir,
        speakers,
        max_sequences_per_speaker=None
):
    info = []
    selected_embeddings = []
    selected_phonemes = []

    for speaker in speakers:
        speaker_root_dir = os.path.join(root_dir, speaker)
        speaker_counter = Counter()
        speaker_seq_count = 0

        for root, _, files in os.walk(speaker_root_dir):
            if max_sequences_per_speaker is not None and speaker_seq_count >= max_sequences_per_speaker:
                break

            for file in files:
                if max_sequences_per_speaker is not None and speaker_seq_count >= max_sequences_per_speaker:
                    break

                if not file.endswith("_no_av_ph.txt"):
                    continue

                name = file.replace("_no_av_ph.txt", "")
                phoneme_file_path = os.path.join(root, file)
                emb_file_path = os.path.join(root, name + "_no_av_embs.npy")

                if not os.path.exists(emb_file_path):
                    continue

                with open(phoneme_file_path, "r", encoding="utf-8") as f:
                    phonemes = [line.strip() for line in f.readlines()]

                emb_array = np.load(emb_file_path)

                if len(phonemes) != len(emb_array):
                    continue

                emb_sequence = []
                phoneme_sequence = []

                for phoneme, emb in zip(phonemes, emb_array):
                    parts = phoneme.split('_')
                    if len(parts) != 3:
                        continue

                    base_ph = parts[1]
                    # if base_ph not in phoneme_list:
                    #     continue

                    emb_sequence.append(emb)
                    phoneme_sequence.append(base_ph)
                    speaker_counter[base_ph] += 1

                if len(phoneme_sequence) > 0:
                    selected_embeddings.append(
                        torch.tensor(np.stack(emb_sequence), dtype=torch.float32)
                    )
                    selected_phonemes.append({file: phoneme_sequence})
                    speaker_seq_count += 1

        info.append({
            speaker: {
                "num_sequences": speaker_seq_count,
                "phoneme_distribution": speaker_counter
            }
        })

        print(f"{speaker}: {speaker_seq_count} sequences")

    return selected_embeddings, selected_phonemes, info